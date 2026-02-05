# TODO: Update this import path to match your Pipe location
from .pipe_api import Pipe

_pipe = None
_setup_hooks = []


def register_setup_hook(fn):
    """Register a function to be called before Pipe is instantiated.

    The function receives the Pipe class and can modify it.

    Example:
        def my_hook(pipe_class):
            pipe_class.some_method = patched_method

        register_setup_hook(my_hook)
    """
    if _pipe is not None:
        raise RuntimeError("Cannot register hook after Pipe is already instantiated")
    _setup_hooks.append(fn)


def get_pipe():
    """Get the global Pipe instance (lazy initialization)."""
    global _pipe
    if _pipe is None:
        # Run all setup hooks before instantiation
        for hook in _setup_hooks:
            hook(Pipe)
        _pipe = Pipe()
    return _pipe


def reset_pipe():
    """Reset the global Pipe instance (mainly for testing)."""
    global _pipe
    _pipe = None
