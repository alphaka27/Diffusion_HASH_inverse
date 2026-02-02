"""
Metaclass to freeze class variables after initialization.
"""

from typing import ClassVar, List

class FreezeClassVar(type):
    """
    Metaclass to freeze class variables after initialization.
    """
    _is_locked: ClassVar[bool] = False

    def __setattr__(cls, key, value):
        if cls._is_locked:
            raise AttributeError("Cannot modify class variable after initialization.")

        if isinstance(value, List):
            value = tuple(value)

        return super().__setattr__(key, value)

    def __delattr__(cls, name):
        if cls._is_locked:
            raise AttributeError("Cannot delete class variable after initialization.")
        return super().__delattr__(name)

    def lock(cls):
        """
        Lock the class to prevent further modifications to class variables.
        """
        type.__setattr__(cls, '_is_locked', True)

    def unlock(cls):
        """
        Unlock the class to allow modifications to class variables.
        """
        type.__setattr__(cls, '_is_locked', False)

    def is_locked(cls) -> bool:
        """
        Check if the class is locked.
        """
        return cls._is_locked
