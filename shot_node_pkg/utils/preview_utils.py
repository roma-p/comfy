"""
Preview utilities for ReadNode.
Handles preview generation for node display (ffmpeg video previews, etc.)
"""

import os
import shutil
import uuid
import asyncio
import subprocess
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
import folder_paths
from aiohttp import web

from .file_utils import escape_ffmpeg_path, validate_size_string

ffmpeg_path = None
try:
    from imageio_ffmpeg import get_ffmpeg_exe
    ffmpeg_path = get_ffmpeg_exe()
except Exception:
    ffmpeg_path = shutil.which("ffmpeg")


async def generate_preview_animated(
    files: List[str],
    request: web.Request,
    force_size: Optional[str] = None,
    frame_duration: float = 0.125,
) -> web.StreamResponse:
    """
    Generate animated webm preview from image sequence via ffmpeg.

    Creates a concat file, builds ffmpeg command, and streams the output
    directly to the HTTP response.

    Args:
        files: List of image file paths to include in preview
        request: aiohttp web request to stream response to
        force_size: Optional size constraint like "512x?", "?x512", or "512x512"
        frame_duration: Duration per frame in seconds (default: 0.125 = 8fps)

    Returns:
        StreamResponse on success, or error Response on failure
    """
    if not files:
        return web.Response(status=204, text="No files to preview")

    if ffmpeg_path is None:
        return web.Response(status=500, text="ffmpeg not available")

    # Create concat file in temp directory
    temp_dir = folder_paths.get_temp_directory()
    os.makedirs(temp_dir, exist_ok=True)
    concat_file = os.path.join(temp_dir, f"preview_concat_{uuid.uuid4().hex[:8]}.txt")

    # Write concat file with properly escaped paths
    with open(concat_file, "w") as f:
        f.write("ffconcat version 1.0\n")
        for path in files:
            escaped_path = escape_ffmpeg_path(os.path.abspath(path))
            f.write(f"file '{escaped_path}'\n")
            f.write(f"duration {frame_duration}\n")

    # Build ffmpeg args
    args = [ffmpeg_path, "-v", "error", "-safe", "0", "-i", concat_file]

    # Add scaling filter if size specified and valid
    if force_size and force_size != "Disabled" and validate_size_string(force_size):
        size = force_size.lower().split("x")
        if len(size) == 2:
            # Use safe integer conversion for dimensions
            width = "-2" if size[0] == "?" else f"'min({int(size[0])},iw)'"
            height = "-2" if size[1] == "?" else f"'min({int(size[1])},ih)'"
            args += ["-vf", f"scale={width}:{height}"]

    # Output settings: VP9 webm, optimized for real-time streaming
    args += ["-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "8", "-f", "webm", "-"]

    # Stream ffmpeg output
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=subprocess.PIPE,
            stdin=subprocess.DEVNULL
        )

        resp = web.StreamResponse()
        resp.content_type = "video/webm"
        await resp.prepare(request)

        while True:
            bytes_read = await proc.stdout.read(2**20)
            if not bytes_read:
                break
            await resp.write(bytes_read)

        await proc.wait()
        return resp

    except Exception as e:
        return web.Response(status=500, text=str(e))


def generate_preview_static(
    image_tensor: torch.Tensor,
    source_path: str = "",
    max_size: int = 512
) -> List[dict]:
    """
    Generate static PNG preview from image tensor.

    Creates a PNG preview image from the first frame of the tensor,
    suitable for displaying in ComfyUI node UI.

    Args:
        image_tensor: Tensor [B, H, W, C] - uses first frame
        source_path: Source path for unique filename generation
        max_size: Maximum preview dimension

    Returns:
        List of preview dicts for ComfyUI UI, or empty list on failure
    """
    try:
        # Get first frame
        if image_tensor.dim() == 4 and image_tensor.shape[0] > 0:
            frame = image_tensor[0]
        elif image_tensor.dim() == 3:
            frame = image_tensor
        else:
            return []

        # Convert to numpy, clip to 0-1 range (tone map HDR)
        img_array = frame.cpu().numpy()
        img_array = np.clip(img_array, 0, 1)
        img_array = (img_array * 255).astype(np.uint8)

        # Create PIL image
        if img_array.shape[2] == 3:
            pil_img = Image.fromarray(img_array, mode='RGB')
        elif img_array.shape[2] == 4:
            pil_img = Image.fromarray(img_array, mode='RGBA')
        else:
            pil_img = Image.fromarray(img_array[:, :, :3], mode='RGB')

        # Resize for preview
        if pil_img.width > max_size or pil_img.height > max_size:
            ratio = min(max_size / pil_img.width, max_size / pil_img.height)
            new_size = (int(pil_img.width * ratio), int(pil_img.height * ratio))
            pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)

        # Save to temp directory
        temp_dir = folder_paths.get_temp_directory()
        file_hash = abs(hash(source_path)) % 1000000
        unique_id = uuid.uuid4().hex[:8]
        filename = f"read_preview_{file_hash}_{unique_id}.png"
        filepath = os.path.join(temp_dir, filename)

        pil_img.save(filepath, format='PNG')

        return [{
            "filename": filename,
            "subfolder": "",
            "type": "temp"
        }]

    except Exception as e:
        print(f"EXR preview generation failed: {e}")
        return []
