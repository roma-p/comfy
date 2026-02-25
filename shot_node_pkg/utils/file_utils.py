"""
File utilities for ReadNode
"""

import os
import re
import hashlib
from enum import Enum
from typing import List, Optional, Set


# Supported file extensions
IMG_EXTENSIONS: Set[str] = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif', '.webp'}
EXR_EXTENSIONS: Set[str] = {'.exr'}
ALL_EXTENSIONS: Set[str] = IMG_EXTENSIONS | EXR_EXTENSIONS


class FileType(Enum):
    """Detected file type."""
    STANDARD = "standard"  # PNG, JPG, etc.
    EXR = "exr"
    UNKNOWN = "unknown"


def strip_path(path: Optional[str]) -> Optional[str]:
    """Strip whitespace and quotes from path."""
    if path is None:
        return None
    path = path.strip()
    if path.startswith('"'):
        path = path[1:]
    if path.endswith('"'):
        path = path[:-1]
    return path


def get_extension(path: str) -> str:
    """
    Get lowercase file extension without the dot.

    Args:
        path: File path or filename

    Returns:
        Extension without dot, lowercase (e.g., "png", "exr")
    """
    return os.path.splitext(path)[1].lower().lstrip('.')


def has_extension(path: str, extensions: Optional[Set[str]]) -> bool:
    """
    Check if file has one of the given extensions.

    Args:
        path: File path or filename
        extensions: Set of valid extensions (with or without dot, any case)
                   If None, returns True (all extensions valid)

    Returns:
        True if extension matches or extensions is None
    """
    if extensions is None:
        return True
    ext = get_extension(path)
    # Normalize extensions set (remove dots, lowercase)
    normalized = {e.lower().lstrip('.') for e in extensions}
    return ext in normalized


def escape_ffmpeg_path(path: str) -> str:
    """
    Escape a file path for use in ffmpeg concat file.

    ffmpeg concat format requires:
    - Single quotes around paths
    - Single quotes escaped by doubling them
    - Backslashes escaped

    Args:
        path: File path to escape

    Returns:
        Escaped path safe for ffmpeg concat file
    """
    # Escape backslashes first, then single quotes
    escaped = path.replace("\\", "\\\\").replace("'", "'\\''")
    return escaped


def validate_size_string(size_str: str) -> bool:
    """
    Validate a size string for ffmpeg scale filter.

    Valid formats:
    - "512x512" (both dimensions specified)
    - "512x?" (width specified, height auto)
    - "?x512" (height specified, width auto)

    Args:
        size_str: Size string to validate

    Returns:
        True if valid format
    """
    if not size_str or size_str == "Disabled":
        return True

    # Pattern: digits or ? for each dimension, separated by x
    pattern = r'^(\d+|\?)[xX](\d+|\?)$'
    return bool(re.match(pattern, size_str))


def calculate_file_hash(filename: str) -> str:
    """Calculate hash based on filename and modification time."""
    h = hashlib.sha256()
    h.update(filename.encode())
    h.update(str(os.path.getmtime(filename)).encode())
    return h.hexdigest()


def get_sorted_dir_files_from_directory(
    directory: str,
    skip_first_images: int = 0,
    select_every_nth: int = 1,
    extensions: Optional[Set[str]] = None
) -> List[str]:
    """
    Get sorted list of image files from directory.

    Args:
        directory: Path to directory
        skip_first_images: Number of images to skip from start
        select_every_nth: Select every Nth image (1 = all)
        extensions: Set of valid extensions (with dot, lowercase)

    Returns:
        List of full file paths, sorted alphabetically
    """
    directory = strip_path(directory)
    if not directory or not os.path.isdir(directory):
        return []

    if extensions is None:
        extensions = ALL_EXTENSIONS

    dir_files = os.listdir(directory)
    dir_files = sorted(dir_files)
    dir_files = [os.path.join(directory, x) for x in dir_files]
    dir_files = [f for f in dir_files if os.path.isfile(f)]

    # Filter by extension
    filtered_files = [f for f in dir_files if has_extension(f, extensions)]

    # Apply skip and step
    filtered_files = filtered_files[skip_first_images:]
    filtered_files = filtered_files[0::select_every_nth]

    return filtered_files


def detect_file_type(directory: str) -> FileType:
    """
    Detect the predominant file type in a directory.

    Args:
        directory: Path to directory

    Returns:
        FileType enum: STANDARD, EXR, or UNKNOWN
    """
    directory = strip_path(directory)
    if not directory or not os.path.isdir(directory):
        return FileType.UNKNOWN

    has_standard = False
    has_exr = False

    try:
        for item in os.scandir(directory):
            if not item.is_file():
                continue

            if has_extension(item.name, IMG_EXTENSIONS):
                has_standard = True
            elif has_extension(item.name, EXR_EXTENSIONS):
                has_exr = True

            # Early exit if we know enough
            if has_standard and has_exr:
                break
    except (PermissionError, OSError):
        return FileType.UNKNOWN

    # Prefer EXR if present, otherwise standard
    if has_exr:
        return FileType.EXR
    elif has_standard:
        return FileType.STANDARD

    return FileType.UNKNOWN


