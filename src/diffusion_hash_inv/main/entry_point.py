"""
Entry point for the diffusion hash inversion process.
This module initializes the necessary components.
This module starts the hash inversion process based on the provided configurations.
"""
import sys
from typing import Optional, Any

from diffusion_hash_inv.logger import Logs, Metadata, BaseLogs, StepLogs
from diffusion_hash_inv.config \
    import (MainConfig, MessageConfig, HashConfig, OutputConfig, Byte2RGBConfig)
from diffusion_hash_inv.generator import NBitsGenerator
from diffusion_hash_inv.main.context import RuntimeState, RuntimeConfig
from diffusion_hash_inv.utils import FileIO, RGBImgMaker
from diffusion_hash_inv.utils.progress import progress
from diffusion_hash_inv.validation import validate
from diffusion_hash_inv import hashing


class MainEP:
    """
    Entry point for hash generation and validation
    """

    def __init__(self, runtime_config: RuntimeConfig, file_controller: Optional[FileIO] = None):
        self.program_start_time = Logs.get_current_timestamp()

        self.runtime_cfg: RuntimeConfig = runtime_config
        self.main_cfg: MainConfig = runtime_config.main
        self.output_cfg: OutputConfig = runtime_config.output
        self.rgb_cfg: Byte2RGBConfig = runtime_config.rgb
        print("Main Entry Point Initialized.")
        print(f"Program Start Time: {self.program_start_time}")
        print(f"Hash Algorithm: {self.runtime_cfg.hash.hash_alg_upper}")
        print(f"Message Length: {self.runtime_cfg.message.length}")
        print(f"Data Directory: {self.output_cfg.data_dir}")
        print(f"Output Directory: {self.output_cfg.output_dir}")


        self.io_controller = file_controller \
            if file_controller is not None else FileIO(self.main_cfg, self.output_cfg)

    def message_generator(self,
                        msg_config: MessageConfig,
                        **kwargs) -> bytes:
        """
        Generate random message for hashing
        """
        generator = None
        message = kwargs.pop("message", None)

        if not msg_config.message_flag:
            generator = NBitsGenerator(self.runtime_cfg,
                                    self.io_controller,
                                    self.program_start_time)
        else:
            raise NotImplementedError\
                ("Message generation for arbitrary messages is not implemented yet.")

        assert generator is not None, "Generator is not initialized."

        _msg = generator.main(value=message)
        return _msg


    def get_hash_alg(self, steplogs: StepLogs):
        """
        Get the hashing algorithm instance based on name
        """
        alg_name = self.runtime_cfg.hash.hash_alg_upper
        try:
            algo = getattr(hashing, alg_name.upper())\
                (self.runtime_cfg.hash, steplogs, self.main_cfg.verbose_flag)
        except AttributeError as e:
            raise ValueError(f"Unsupported algo: {alg_name}") from e

        return algo

    def rgb_image_maker(self):
        """
        Create RGB image from hash outputs
        """
        rgb_encoder = RGBImgMaker(
                                self.runtime_cfg,
                                self.io_controller,
                                self.rgb_cfg)
        rgb_encoder.main()

    def _make_img_perf(self):
        """
        Create RGB image from hash outputs
        """
        print("RGB Image Maker Module Loaded.")
        _img_make_start = Logs.perftimer_start()

        self.rgb_image_maker()

        _img_make_end = Logs.perftimer_end(_img_make_start)
        img_process_time = Logs.perftimer_str(_img_make_end)
        print("RGB Image Maker execution time:", img_process_time)
        print("RGB Image Maker process completed.")
        print("=========================")
        print()

        return _img_make_end

    def _loop_preprocess(self) -> RuntimeState:
        """
        Preprocessing steps before main loop
        """
        msg_cfg: MessageConfig = self.runtime_cfg.message
        hash_cfg: HashConfig = self.runtime_cfg.hash

        _metadata: Metadata = Metadata(hash_alg=hash_cfg.hash_alg_upper,
                                    is_message=msg_cfg.message_flag,
                                    input_bits_len=msg_cfg.length,
                                    started_at=self.program_start_time)
        _metadata.hash_property(byteorder=hash_cfg.byteorder,
                            hierarchy=hash_cfg.hierarchy)
        _baselogs = BaseLogs()
        _steplogs = StepLogs(wordsize=hash_cfg.ws_bits, byteorder=hash_cfg.byteorder, \
                    hierarchy=hash_cfg.hierarchy)
        _algo = self.get_hash_alg(_steplogs)

        runtime_state = \
            RuntimeState(metadata=_metadata, baselogs=_baselogs, steplogs=_steplogs, algo=_algo)

        return runtime_state

    def _loop_main(self, state: RuntimeState, **kwargs) -> RuntimeState:
        """
        Main loop for hash generation and validation
        """

        msg_cfg: MessageConfig = self.runtime_cfg.message
        hash_cfg: HashConfig = self.runtime_cfg.hash

        updated_state = state.copy()
        assert updated_state.algo is not None, "Hash algorithm instance is not initialized."
        updated_state.algo.reset()

        Logs.clear(baselogs=updated_state.baselogs, steplogs=updated_state.steplogs)

        _loop_timer_start = Logs.perftimer_start()

        input_msg = self.message_generator(msg_cfg, **kwargs)

        generated_hash = updated_state.algo.digest(input_msg)

        valid, right_hash = \
            validate(generated_hash,
                    input_msg,
                    hash_cfg.hash_alg_upper,
                    self.main_cfg.verbose_flag)

        _loop_timer_end = Logs.perftimer_end(_loop_timer_start)

        if not valid:
            print(f"Generated Hash: {Logs.bytes_to_str(generated_hash)}")
            print(f"Correct   Hash: {Logs.bytes_to_str(right_hash)}")
            raise RuntimeError("Hash validation failed.")

        updated_state.metadata.time_logger(_loop_timer_end)

        updated_state.baselogs.update(message=input_msg, \
                        generated_hash=generated_hash, \
                        correct_hash=right_hash, \
                        is_message=msg_cfg.message_flag)

        return updated_state

    def main(self, iteration: int = -1, **kwargs):
        """
        Main entry point for hash generation and validation
        """
        runtime_state = self._loop_preprocess()
        assert iteration >= 0, "Iteration count must be non-negative integer."

        with progress(
            range(iteration),
            desc="Hash Generation Progress",
            unit="iteration",
        ) as pbar:
            pbar.set_postfix(
                {"Hash Algorithm": self.runtime_cfg.hash.hash_alg_upper,
                "Message Length": self.runtime_cfg.message.length}
                )

            for _i in pbar:
                kwargs.update({"message": _i})
                    # For sequential mode, generate messages based on iteration index
                json_file_name = \
                    Logs.json_file_namer(
                        self.runtime_cfg.hash.hash_alg_upper,
                        self.runtime_cfg.message.length,
                        self.program_start_time,
                        _i, iteration)

                runtime_state = self._loop_main(runtime_state, **kwargs)

                self.io_controller.file_writer(
                    filename=json_file_name,
                    content={"metadata": runtime_state.metadata,
                        "baselogs": runtime_state.baselogs, "steplogs": runtime_state.steplogs},
                    length=self.runtime_cfg.message.length,
                    path_infix=f"{self.program_start_time}/{_i}")
        sys.stdout.flush()

    def run(self,
            iteration: Optional[int] = None,
            **kwargs):
        """
        Run the main process
        """
        if iteration is None:
            mode = kwargs.get("mode", "default")
            if mode == "sequential":
                iteration = 2 ** self.runtime_cfg.message.length
                print(f"Running in sequential mode.\nIteration count set to: {iteration}")
            else:
                raise ValueError("Iteration count must be specified for non-sequential modes.\n"
                                "Use --iteration or -i flag to specify the number of iterations.\n")
        if iteration < 0:
            raise ValueError("Iteration count must be a positive integer.\n"
                            "Use --iteration or -i flag to specify the number of iterations.\n"
                            f"Currently set to: {iteration}\n")

        _start_total = Logs.perftimer_start()

        self.main(iteration, **kwargs)

        _end_total = Logs.perftimer_end(_start_total)
        elapsed_time_str = Logs.perftimer_str(_end_total)
        print(f"Hash Calculation time: {elapsed_time_str}")
        print("Process completed.")
        print("=========================\n")

        _img_make_end = 0

        if self.main_cfg.make_image_flag:
            _img_make_end = self._make_img_perf()

        print(f"Total Execution Time: {Logs.perftimer_str(_end_total + _img_make_end)}\n")
