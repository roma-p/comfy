"""
Utils module - Shared utilities for ReadNode
"""

from .file_utils import (
    strip_path,
    calculate_file_hash,
    escape_ffmpeg_path,
    validate_size_string,
    IMG_EXTENSIONS,
    EXR_EXTENSIONS,
    ALL_EXTENSIONS,
)

from .sequence_utils import (
    has_sequence_pattern,
    find_sequence_files,
    detect_sequences,
    resolve_sequence_files,
    detect_file_type_from_path,
    SequencePattern,
)

from .preview_utils import (
    generate_preview_animated,
    generate_preview_static,
)

__all__ = [
    # File utils
    'strip_path',
    'calculate_file_hash',
    'escape_ffmpeg_path',
    'validate_size_string',
    'IMG_EXTENSIONS',
    'EXR_EXTENSIONS',
    'ALL_EXTENSIONS',
    # Sequence utils
    'has_sequence_pattern',
    'find_sequence_files',
    'detect_sequences',
    'resolve_sequence_files',
    'detect_file_type_from_path',
    'SequencePattern',
    # Preview utils
    'generate_preview_animated',
    'generate_preview_static',
]
