from .pipe_api import Pipe

_pipe = None


def get_pipe():
    """Get the global Pipe instance (lazy initialization)."""
    global _pipe
    if _pipe is None:
        _pipe = Pipe()
    return _pipe
