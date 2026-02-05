"""
File utilities for ReadNode
"""

import os
import hashlib
from enum import Enum
from typing import List, Optional, Set


# Supported file extensions
IMG_EXTENSIONS: Set[str] = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif', '.webp'}
EXR_EXTENSIONS: Set[str] = {'.exr'}
ALL_EXTENSIONS: Set[str] = IMG_EXTENSIONS | EXR_EXTENSIONS


class FileType(Enum):
    """Detected file type for a directory."""
    STANDARD = "standard"  # PNG, JPG, etc.
    EXR = "exr"
    MIXED = "mixed"  # Contains both types
    EMPTY = "empty"  # No valid files
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
    filtered_files = []
    for filepath in dir_files:
        ext = os.path.splitext(filepath)[1].lower()
        if ext in extensions:
            filtered_files.append(filepath)

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
        FileType enum indicating what types of files are present
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

            ext = os.path.splitext(item.name)[1].lower()
            if ext in IMG_EXTENSIONS:
                has_standard = True
            elif ext in EXR_EXTENSIONS:
                has_exr = True

            # Early exit if we've found both types
            if has_standard and has_exr:
                return FileType.MIXED
    except (PermissionError, OSError):
        return FileType.UNKNOWN

    if has_exr and not has_standard:
        return FileType.EXR
    elif has_standard and not has_exr:
        return FileType.STANDARD
    elif not has_standard and not has_exr:
        return FileType.EMPTY

    return FileType.MIXED


def get_first_valid_file(directory: str, extensions: Optional[Set[str]] = None) -> Optional[str]:
    """
    Get the first valid file from a directory.

    Args:
        directory: Path to directory
        extensions: Set of valid extensions (defaults to ALL_EXTENSIONS)

    Returns:
        Path to first valid file, or None if none found
    """
    files = get_sorted_dir_files_from_directory(directory, extensions=extensions)
    return files[0] if files else None
