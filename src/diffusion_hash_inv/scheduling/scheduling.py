"""
Beta scheduling utilities derived from hash intermediate statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Iterator

import numpy as np
from numpy.typing import ArrayLike, NDArray

from diffusion_hash_inv.logger import Logs

if TYPE_CHECKING:
    from diffusion_hash_inv.analyze import BetaScheduleSummary


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MultiplyRescaleResult:
    """
    Candidate beta schedules for Approach 1.
    """

    base_betas: FloatArray
    raw_candidate: FloatArray
    rescaled_candidate: FloatArray


@dataclass(frozen=True)
class LinearMappingResult:
    """
    Candidate beta schedule and coefficients for Approach 2.
    """

    candidate: FloatArray
    slope: float
    sn_min: float
    sn_max: float


class BetaScheduler:
    """
    Build candidate beta schedules from summarized SN values.
    """

    def __init__(
        self,
        beta_min: float,
        beta_max: float,
        sn_array: ArrayLike | None = None,
        *,
        dtype: np.dtype | type = np.float64,
    ) -> None:
        if beta_max < beta_min:
            raise ValueError("beta_max must be greater than or equal to beta_min.")

        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.dtype = np.dtype(dtype)
        self.sn_array = None if sn_array is None else self._as_1d_array(
            sn_array,
            name="sn_array",
            dtype=self.dtype,
        )

    @staticmethod
    def _as_1d_array(
        values: ArrayLike,
        *,
        name: str,
        dtype: np.dtype | type = np.float64,
    ) -> FloatArray:
        arr = np.asarray(values, dtype=dtype)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError(f"{name} must be a non-empty one-dimensional array.")
        return arr

    @staticmethod
    def _validate_beta_range(beta_min: float, beta_max: float) -> tuple[float, float]:
        beta_min = float(beta_min)
        beta_max = float(beta_max)
        if beta_max < beta_min:
            raise ValueError("beta_max must be greater than or equal to beta_min.")
        return beta_min, beta_max

    def base_betas(self, length: int) -> FloatArray:
        """
        Return the original linear beta schedule used as the base schedule.
        """
        if length <= 0:
            raise ValueError("length must be positive.")
        return np.linspace(
            start=self.beta_min,
            stop=self.beta_max,
            num=int(length),
            dtype=self.dtype,
        )

    def _resolve_mean_sn(self, mean_sn_values: ArrayLike | None) -> FloatArray:
        if mean_sn_values is not None:
            return self._as_1d_array(mean_sn_values, name="mean_sn_values", dtype=self.dtype)
        if self.sn_array is None:
            raise ValueError("mean_sn_values must be provided when sn_array is not configured.")
        return self.sn_array

    @staticmethod
    def rescale_betas(
        mean_sn_values: ArrayLike,
        beta_min: float,
        beta_max: float,
        *,
        dtype: np.dtype | type = np.float64,
    ) -> FloatArray:
        """
        Rescale values to the beta range using min-max normalization.
        """
        beta_min, beta_max = BetaScheduler._validate_beta_range(beta_min, beta_max)
        values = BetaScheduler._as_1d_array(
            mean_sn_values,
            name="mean_sn_values",
            dtype=dtype,
        )
        value_min = float(np.min(values))
        value_max = float(np.max(values))
        midpoint = (beta_min + beta_max) / 2

        if value_max == value_min:
            return np.full(values.shape, midpoint, dtype=dtype)

        rescaled = (values - value_min) * (beta_max - beta_min) / (value_max - value_min)
        rescaled = rescaled + beta_min
        return np.clip(rescaled, beta_min, beta_max).astype(dtype, copy=False)

    def multiply_with_base(
        self,
        mean_sn_values: ArrayLike | None = None,
        *,
        base_betas: ArrayLike | None = None,
    ) -> FloatArray:
        """
        Multiply mean SN values by the base beta schedule.
        """
        mean_sn = self._resolve_mean_sn(mean_sn_values)
        base = self.base_betas(mean_sn.size) if base_betas is None else self._as_1d_array(
            base_betas,
            name="base_betas",
            dtype=self.dtype,
        )

        if base.shape != mean_sn.shape:
            raise ValueError(
                f"base_betas shape mismatch: expected {mean_sn.shape}, got {base.shape}."
            )

        return np.multiply(mean_sn, base).astype(self.dtype, copy=False)

    def approach1(
        self,
        mean_sn_values: ArrayLike | None = None,
        *,
        base_betas: ArrayLike | None = None,
    ) -> MultiplyRescaleResult:
        """
        Approach 1: multiply mean SN values by base betas, then rescale.
        """
        mean_sn = self._resolve_mean_sn(mean_sn_values)
        base = self.base_betas(mean_sn.size) if base_betas is None else self._as_1d_array(
            base_betas,
            name="base_betas",
            dtype=self.dtype,
        )
        raw_candidate = self.multiply_with_base(mean_sn, base_betas=base)
        rescaled_candidate = self.rescale_betas(
            raw_candidate,
            self.beta_min,
            self.beta_max,
            dtype=self.dtype,
        )
        return MultiplyRescaleResult(
            base_betas=base,
            raw_candidate=raw_candidate,
            rescaled_candidate=rescaled_candidate,
        )

    def approach2(self, mean_sn_values: ArrayLike | None = None) -> LinearMappingResult:
        """
        Approach 2: linearly map mean SN values into the beta range.
        """
        mean_sn = self._resolve_mean_sn(mean_sn_values)
        sn_min = float(np.min(mean_sn))
        sn_max = float(np.max(mean_sn))

        if sn_max == sn_min:
            slope = 0.0
            candidate = np.full(
                mean_sn.shape,
                (self.beta_min + self.beta_max) / 2,
                dtype=self.dtype,
            )
        else:
            slope = (self.beta_max - self.beta_min) / (sn_max - sn_min)
            candidate = slope * (mean_sn - sn_min) + self.beta_min
            candidate = np.clip(candidate, self.beta_min, self.beta_max).astype(
                self.dtype,
                copy=False,
            )

        return LinearMappingResult(
            candidate=candidate,
            slope=float(slope),
            sn_min=sn_min,
            sn_max=sn_max,
        )

    def summarize_approaches(
        self,
        summary: BetaScheduleSummary,
    ) -> tuple[MultiplyRescaleResult, LinearMappingResult]:
        """
        Build both notebook approaches from a BetaScheduleSummary.
        """
        return self.approach1(summary.mean), self.approach2(summary.mean)

    @staticmethod
    def get_step4(io_controller: Any, runtime_cfg: Any) -> list[dict[str, Any]]:
        """
        Get Step 4 logs from the latest hash logs.
        """
        logs = Logs.iter_logs(io_controller, runtime_cfg.hash, runtime_cfg.main)
        return list(Logs.iter_step_logs(logs, step_name="4th Step"))

    @staticmethod
    def iter_step4(logs: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        """
        Stream Step 4 logs from raw or wrapped log dictionaries.
        """
        return Logs.iter_step_logs(logs, step_name="4th Step")


__all__ = [
    "BetaScheduler",
    "LinearMappingResult",
    "MultiplyRescaleResult",
]
