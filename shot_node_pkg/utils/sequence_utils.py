"""
Sequence pattern utilities for ReadNode
Supports multiple padding formats: %04d (printf), $F4 (Houdini), #### (hash)
"""

import os
import re
from enum import Enum
from typing import Optional, Tuple, List, Set
from dataclasses import dataclass, field

from .file_utils import has_extension


class PatternType(Enum):
    PRINTF = "printf"    # %04d, %03d, etc.
    HOUDINI = "houdini"  # $F4, $F3, $F, etc.
    HASH = "hash"        # ####, ###, etc.
    NONE = "none"        # No pattern detected


@dataclass
class SequencePattern:
    """
    Represents an image sequence pattern.

    Sortable by (directory, prefix, suffix) for consistent ordering.
    """
    pattern_type: PatternType
    padding: int
    prefix: str      # Part before the pattern (e.g., "image.")
    suffix: str      # Part after the pattern (e.g., ".exr")
    directory: str   # Directory containing the sequence
    original: str = ""  # Original pattern string (e.g., "####" or "%04d")
    first_frame: int = 0
    last_frame: int = 0
    frame_count: int = 0
    files: List[str] = field(default_factory=list)

    def __lt__(self, other: "SequencePattern") -> bool:
        """Sort by directory, then prefix, then suffix."""
        return (self.directory, self.prefix, self.suffix) < (other.directory, other.prefix, other.suffix)

    def to_regex(self) -> str:
        """Generate regex to match sequence files."""
        prefix_escaped = re.escape(self.prefix)
        suffix_escaped = re.escape(self.suffix)
        return f"^{prefix_escaped}(\\d{{{self.padding},}}){suffix_escaped}$"

    def to_pattern_string(self) -> str:
        """Generate pattern filename (e.g., 'image.####.exr')."""
        return f"{self.prefix}{'#' * self.padding}{self.suffix}"

    def to_full_path(self) -> str:
        """Generate full pattern path (e.g., '/path/to/image.####.exr')."""
        return os.path.join(self.directory, self.to_pattern_string())


# Regex patterns for each format
PRINTF_PATTERN = re.compile(r'%0?(\d+)d')      # %04d, %4d, %03d
HOUDINI_PATTERN = re.compile(r'\$F(\d*)')       # $F4, $F3, $F (default 4)
HASH_PATTERN = re.compile(r'(#+)')              # ####, ###, ##


def _detect_pattern_type(path: str) -> PatternType:
    """Detect which pattern type is used in the path."""
    if PRINTF_PATTERN.search(path):
        return PatternType.PRINTF
    elif HOUDINI_PATTERN.search(path):
        return PatternType.HOUDINI
    elif HASH_PATTERN.search(path):
        return PatternType.HASH
    return PatternType.NONE


def _parse_pattern(path: str) -> Optional[SequencePattern]:
    """
    Parse a sequence pattern path into components.

    Args:
        path: Sequence path like "/path/to/image.####.exr" or "/path/to/render.$F4.png"

    Returns:
        SequencePattern object or None if no pattern found
    """
    if not path:
        return None

    path = path.strip().strip('"')
    directory = os.path.dirname(path)
    filename = os.path.basename(path)

    if not filename:
        return None

    pattern_type = _detect_pattern_type(filename)

    if pattern_type == PatternType.PRINTF:
        match = PRINTF_PATTERN.search(filename)
        if match:
            padding = int(match.group(1)) if match.group(1) else 1
            return SequencePattern(
                pattern_type=pattern_type,
                padding=padding,
                prefix=filename[:match.start()],
                suffix=filename[match.end():],
                original=match.group(0),
                directory=directory or "."
            )

    elif pattern_type == PatternType.HOUDINI:
        match = HOUDINI_PATTERN.search(filename)
        if match:
            padding = int(match.group(1)) if match.group(1) else 4
            return SequencePattern(
                pattern_type=pattern_type,
                padding=padding,
                prefix=filename[:match.start()],
                suffix=filename[match.end():],
                original=match.group(0),
                directory=directory or "."
            )

    elif pattern_type == PatternType.HASH:
        match = HASH_PATTERN.search(filename)
        if match:
            return SequencePattern(
                pattern_type=pattern_type,
                padding=len(match.group(1)),
                prefix=filename[:match.start()],
                suffix=filename[match.end():],
                original=match.group(0),
                directory=directory or "."
            )

    return None


def has_sequence_pattern(path: str) -> bool:
    return _detect_pattern_type(path) != PatternType.NONE


def _find_sequence_files(pattern_path: str) -> List[str]:
    """
    Find all files matching the sequence pattern.

    Args:
        pattern_path: Path with sequence pattern

    Returns:
        Sorted list of matching file paths
    """
    pattern = _parse_pattern(pattern_path)
    if not pattern:
        return []

    if not os.path.isdir(pattern.directory):
        return []

    regex = re.compile(pattern.to_regex())
    matching_files = []

    try:
        for filename in os.listdir(pattern.directory):
            if regex.match(filename):
                full_path = os.path.join(pattern.directory, filename)
                if os.path.isfile(full_path):
                    matching_files.append(full_path)
    except (PermissionError, OSError):
        return []

    return sorted(matching_files)


def detect_sequences(
    directory: str,
    extensions: Optional[Set[str]] = None,
    min_frames: int = 2
) -> List[SequencePattern]:
    """
    Detect all image sequences in a directory.

    Scans the directory for numbered files, groups them by pattern,
    and returns a sorted list of SequencePattern objects.

    Args:
        directory: Directory path to scan
        extensions: Optional set of valid extensions (with dot, lowercase).
                   If None, uses all supported image extensions.
        min_frames: Minimum number of frames to be considered a sequence (default: 2)

    Returns:
        List of SequencePattern objects, sorted by (directory, prefix, suffix)
    """
    from .file_utils import ALL_EXTENSIONS

    if not directory or not os.path.isdir(directory):
        return []

    if extensions is None:
        extensions = ALL_EXTENSIONS

    # Pattern to detect numbered files: prefix + digits + suffix
    # Matches: image.0001.exr, render_001.png, frame0001.jpg
    number_pattern = re.compile(r'^(.+?)(\d{2,})(\.[^.]+)$')

    # Group files by (prefix, padding, suffix)
    sequences: dict = {}

    try:
        for item in os.scandir(directory):
            if not item.is_file():
                continue

            # Check extension
            if not has_extension(item.name, extensions):
                continue

            match = number_pattern.match(item.name)
            if not match:
                continue

            prefix, digits, suffix = match.groups()
            padding = len(digits)
            frame_num = int(digits)
            key = (prefix, padding, suffix)

            if key not in sequences:
                sequences[key] = {
                    "frames": [],
                    "first": frame_num,
                    "last": frame_num,
                }

            full_path = os.path.join(directory, item.name)
            sequences[key]["frames"].append(full_path)
            sequences[key]["first"] = min(sequences[key]["first"], frame_num)
            sequences[key]["last"] = max(sequences[key]["last"], frame_num)

    except (PermissionError, OSError):
        return []

    # Build SequencePattern objects
    result = []
    for (prefix, padding, suffix), info in sequences.items():
        if len(info["frames"]) >= min_frames:
            pattern = SequencePattern(
                pattern_type=PatternType.HASH,
                padding=padding,
                prefix=prefix,
                suffix=suffix,
                directory=directory,
                original="#" * padding,
                first_frame=info["first"],
                last_frame=info["last"],
                frame_count=len(info["frames"]),
                files=sorted(info["frames"]),
            )
            result.append(pattern)

    return sorted(result)


def resolve_sequence_files(
    path: str,
    skip_first: int = 0,
    select_every_nth: int = 1,
    extensions: Optional[Set[str]] = None,
    limit: int = 0
) -> Tuple[List[str], str]:
    """
    Resolve files from sequence pattern or directory path.

    This is the unified entry point for getting files - handles both
    sequence patterns (/path/image.####.exr) and directory paths (/path/to/dir/).

    Args:
        path: Sequence pattern or directory path
        skip_first: Number of files to skip from start
        select_every_nth: Select every Nth file (1 = all)
        extensions: Optional set of valid extensions (with dot, lowercase).
                   If None, determines from path or uses all supported.
        limit: Maximum number of files to return (0 = no limit)

    Returns:
        Tuple of (file_list, file_type_string)
        file_type_string is one of: "standard", "exr", "unknown"
    """
    from .file_utils import (
        IMG_EXTENSIONS, EXR_EXTENSIONS, ALL_EXTENSIONS,
        _get_sorted_dir_files_from_directory, _detect_file_type, strip_path, FileType
    )

    path = strip_path(path)
    if not path:
        return [], "unknown"

    files = []
    file_type = "unknown"

    if has_sequence_pattern(path):
        files = _find_sequence_files(path)

        if extensions is not None:
            files = [f for f in files if has_extension(f, extensions)]

        if files:
            if has_extension(files[0], EXR_EXTENSIONS):
                file_type = "exr"
            elif has_extension(files[0], IMG_EXTENSIONS):
                file_type = "standard"
    else:
        if not os.path.isdir(path):
            return [], "unknown"

        detected = _detect_file_type(path)
        file_type = detected.value

        if extensions is not None:
            use_extensions = extensions
        elif detected == FileType.EXR:
            use_extensions = EXR_EXTENSIONS
        elif detected == FileType.STANDARD:
            use_extensions = IMG_EXTENSIONS
        else:
            use_extensions = ALL_EXTENSIONS

        files = _get_sorted_dir_files_from_directory(path, extensions=use_extensions)

    if skip_first > 0:
        files = files[skip_first:]

    if select_every_nth > 1:
        files = files[::select_every_nth]

    if limit > 0:
        files = files[:limit]

    return files, file_type


def detect_file_type_from_path(path: str) -> str:
    """
    Detect file type from sequence pattern or directory.

    Args:
        path: Sequence pattern or directory path

    Returns:
        File type string: "standard", "exr", "unknown"
    """
    from .file_utils import IMG_EXTENSIONS, EXR_EXTENSIONS, _detect_file_type, strip_path

    path = strip_path(path)
    if not path:
        return "unknown"

    if has_sequence_pattern(path):
        files = _find_sequence_files(path)
        if files:
            if has_extension(files[0], EXR_EXTENSIONS):
                return "exr"
            elif has_extension(files[0], IMG_EXTENSIONS):
                return "standard"
        return "unknown"
    else:
        return _detect_file_type(path).value
