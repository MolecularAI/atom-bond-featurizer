"""Set and reset environment variables."""

import os
from typing import Optional


class Environment:
    """Set and reset environment variables.

    Attributes
    ----------
    **kwargs : Optional[str]
        Arbitrary keyword arguments that represent environment variables and their values.
    _env_cache : Dict[str, str]
        A cache of the original environment variables at the time of instantiation.
    """

    def __init__(self, **kwargs: Optional[str]) -> None:
        for attr_name, value in kwargs.items():
            setattr(self, attr_name, value)

        # Save the original state of the environment
        self._env_cache = dict(os.environ.items())

    def set_environment(self) -> None:
        """Set the environment variables based on the instance attributes.

        Returns
        -------
        None
        """
        for var, value in self.__dict__.items():
            if var == "_env_cache":
                continue
            if value is not None:
                os.environ[var] = str(value)

    def reset_environment(self) -> None:
        """Reset the environment to its original state.

        Returns
        -------
        None
        """
        os.environ.clear()
        os.environ.update(self._env_cache)
