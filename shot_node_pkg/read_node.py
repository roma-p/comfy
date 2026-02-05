"""
Read Node - Unified image sequence loader with EXR support
Supports standard images (PNG, JPG, etc.) and multi-layer EXR files
"""

import os
import json
import hashlib
import shutil
import subprocess
import asyncio
import uuid
from io import BytesIO
import numpy as np
import torch
from PIL import Image, ImageOps

import folder_paths
from comfy.utils import common_upscale, ProgressBar
from aiohttp import web
from server import PromptServer

# Import from local modules
from .utils.file_utils import (
    strip_path, calculate_file_hash, get_sorted_dir_files_from_directory,
    detect_file_type, IMG_EXTENSIONS, EXR_EXTENSIONS, ALL_EXTENSIONS, FileType
)
from .loaders import ImageLoader, EXR_AVAILABLE
if EXR_AVAILABLE:
    from .loaders import ExrLoader


# Constants
BIGMAX = (2**53 - 1)

# Find ffmpeg (using imageio_ffmpeg like VHS does)
ffmpeg_path = None
try:
    from imageio_ffmpeg import get_ffmpeg_exe
    ffmpeg_path = get_ffmpeg_exe()
except:
    ffmpeg_path = shutil.which("ffmpeg")


def get_loader(file_path: str):
    """Get appropriate loader for file type."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext in EXR_EXTENSIONS:
        if not EXR_AVAILABLE:
            raise ImportError(
                "OpenImageIO is required for EXR files but not available. "
                "Install with: pip install OpenImageIO"
            )
        return ExrLoader()
    else:
        return ImageLoader()


# =============================================================================
# Preview Generation
# =============================================================================

def generate_preview(image_tensor: torch.Tensor, source_path: str = "",
                     max_size: int = 512) -> list:
    """
    Generate execution preview from loaded tensor.

    Args:
        image_tensor: Tensor [B, H, W, C] - uses first frame
        source_path: Source path for unique filename
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
        print(f"Preview generation failed: {e}")
        return []


# =============================================================================
# API Endpoints
# =============================================================================

@PromptServer.instance.routes.get("/read_node/getpath")
async def get_path(request):
    """Return directory listing for path autocomplete."""
    query = request.rel_url.query
    if "path" not in query:
        return web.json_response([])

    path = query["path"].strip().strip('"')
    if not path:
        path = "/"

    path = os.path.abspath(path)

    if not os.path.exists(path):
        return web.json_response([])

    valid_extensions = query.get("extensions")
    valid_items = []

    try:
        for item in os.scandir(path):
            try:
                if item.is_dir():
                    valid_items.append(item.name + "/")
                    continue
                ext = os.path.splitext(item.name)[1].lower()
                if valid_extensions is None or ext.lstrip('.') in valid_extensions:
                    valid_items.append(item.name)
            except OSError:
                pass
    except PermissionError:
        return web.json_response([])

    valid_items.sort()
    return web.json_response(valid_items)


@PromptServer.instance.routes.get("/read_node/detect_type")
async def detect_type_endpoint(request):
    """Detect file type in directory (standard vs EXR)."""
    query = request.rel_url.query
    directory = query.get("path", "").strip().strip('"')

    if not directory:
        return web.json_response({"type": "unknown", "exr_available": EXR_AVAILABLE})

    directory = strip_path(directory)
    file_type = detect_file_type(directory)

    return web.json_response({
        "type": file_type.value,
        "exr_available": EXR_AVAILABLE,
    })


@PromptServer.instance.routes.get("/read_node/preview")
async def get_preview(request):
    """Return first image from directory as preview."""
    directory = request.query.get("directory", "")

    if not directory or not os.path.isdir(directory):
        return web.Response(status=404, text="Directory not found")

    files = get_sorted_dir_files_from_directory(directory, extensions=IMG_EXTENSIONS)

    if not files:
        return web.Response(status=404, text="No images found")

    first_file = files[0]

    try:
        img = Image.open(first_file)
        img = ImageOps.exif_transpose(img)
        img.thumbnail((512, 512), Image.Resampling.LANCZOS)
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")

        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)

        return web.Response(
            body=buffer.read(),
            content_type="image/jpeg",
            headers={"Cache-Control": "no-cache"}
        )
    except Exception as e:
        return web.Response(status=500, text=str(e))


@PromptServer.instance.routes.get("/read_node/viewvideo")
async def view_video(request):
    """Generate animated preview from image sequence using ffmpeg."""
    query = request.rel_url.query
    directory = query.get("filename", "").strip().strip('"')

    if not directory or not os.path.isdir(directory):
        return web.Response(status=404)

    if ffmpeg_path is None:
        return web.Response(status=500, text="ffmpeg not found")

    skip_first = int(query.get("skip_first_images", 0))
    select_nth = int(query.get("select_every_nth", 1)) or 1

    # Only use standard image extensions for video preview
    valid_images = get_sorted_dir_files_from_directory(
        directory, skip_first, select_nth, IMG_EXTENSIONS
    )

    if not valid_images:
        return web.Response(status=204)

    # Create unique concat file to avoid race conditions
    os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
    concat_file = os.path.join(
        folder_paths.get_temp_directory(),
        f"read_node_preview_{hash(directory) & 0xFFFFFFFF}.txt"
    )

    with open(concat_file, "w") as f:
        f.write("ffconcat version 1.0\n")
        for path in valid_images:
            f.write(f"file '{os.path.abspath(path)}'\n")
            f.write("duration 0.125\n")

    args = [ffmpeg_path, "-v", "error", "-safe", "0", "-i", concat_file]

    force_size = query.get("force_size", "")
    if force_size and force_size != "Disabled":
        size = force_size.split("x")
        if len(size) == 2:
            size[0] = "-2" if size[0] == "?" else f"'min({size[0]},iw)'"
            size[1] = "-2" if size[1] == "?" else f"'min({size[1]},ih)'"
            args += ["-vf", f"scale={size[0]}:{size[1]}"]

    args += ["-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "8", "-f", "webm", "-"]

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


# =============================================================================
# Image Loading Functions
# =============================================================================

def is_changed_load_images(directory: str, image_load_cap: int = 0, skip_first_images: int = 0,
                           select_every_nth: int = 1, normalize: bool = False, **kwargs):
    """Check if inputs have changed for ComfyUI caching."""
    directory = strip_path(directory)
    if not directory or not os.path.isdir(directory):
        return False

    dir_files = get_sorted_dir_files_from_directory(
        directory, skip_first_images, select_every_nth, ALL_EXTENSIONS
    )
    if image_load_cap > 0:
        dir_files = dir_files[:image_load_cap]

    m = hashlib.sha256()
    m.update(f"normalize={normalize}".encode())
    for filepath in dir_files:
        m.update(calculate_file_hash(filepath).encode())
    return m.digest().hex()


def validate_load_images(directory: str):
    """Validate directory input."""
    directory = strip_path(directory)
    if not directory or not os.path.isdir(directory):
        return f"Directory '{directory}' cannot be found."

    dir_files = get_sorted_dir_files_from_directory(directory, extensions=ALL_EXTENSIONS)
    if len(dir_files) == 0:
        return f"No valid image files in directory '{directory}'."
    return True


def load_images_with_layers(directory: str, image_load_cap: int = 0, skip_first_images: int = 0,
                            select_every_nth: int = 1, normalize: bool = False,
                            meta_batch=None, unique_id=None):
    """
    Load images from directory with full layer support.

    Returns:
        Tuple of (images, masks, frame_count, layers, cryptomatte, metadata)
    """
    directory = strip_path(directory)

    # Get files based on detected type
    file_type = detect_file_type(directory)

    if file_type == FileType.EXR:
        extensions = EXR_EXTENSIONS
    elif file_type == FileType.STANDARD:
        extensions = IMG_EXTENSIONS
    else:
        # Mixed or unknown - use all extensions
        extensions = ALL_EXTENSIONS

    dir_files = get_sorted_dir_files_from_directory(
        directory, skip_first_images, select_every_nth, extensions
    )

    if not dir_files:
        raise FileNotFoundError(f"No valid files in directory '{directory}'")

    if image_load_cap > 0:
        dir_files = dir_files[:image_load_cap]

    # Determine loader from first file
    first_file = dir_files[0]
    loader = get_loader(first_file)
    is_exr = isinstance(loader, ExrLoader) if EXR_AVAILABLE else False

    # Load first image to get dimensions
    first_result = loader.load_image(first_file, normalize=normalize)
    target_width = first_result.width
    target_height = first_result.height

    # Prepare batch lists
    image_list = [first_result.image]
    mask_list = [first_result.mask]

    # For EXR, track layers and cryptomatte
    all_layers = {}
    all_cryptomatte = {}

    if is_exr:
        for layer_name, layer_tensor in first_result.layers.items():
            all_layers[layer_name] = [layer_tensor]
        for crypto_name, crypto_tensor in first_result.cryptomatte.items():
            all_cryptomatte[crypto_name] = [crypto_tensor]

    # Load remaining images
    pbar = ProgressBar(len(dir_files))

    for i, file_path in enumerate(dir_files[1:], start=1):
        try:
            result = loader.load_image(file_path, normalize=normalize)

            # Resize if needed
            if result.width != target_width or result.height != target_height:
                result.image = loader.resize_to_match(
                    result.image, target_width, target_height
                )
                if result.mask.shape[-2:] != (target_height, target_width):
                    result.mask = loader.resize_to_match(
                        result.mask.unsqueeze(-1), target_width, target_height
                    ).squeeze(-1)

            image_list.append(result.image)
            mask_list.append(result.mask)

            # Collect layers for EXR
            if is_exr:
                for layer_name, layer_tensor in result.layers.items():
                    if layer_name in all_layers:
                        all_layers[layer_name].append(layer_tensor)
                for crypto_name, crypto_tensor in result.cryptomatte.items():
                    if crypto_name in all_cryptomatte:
                        all_cryptomatte[crypto_name].append(crypto_tensor)

        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
            # Use white placeholder
            image_list.append(torch.ones_like(first_result.image))
            mask_list.append(torch.zeros_like(first_result.mask))

        pbar.update_absolute(i, len(dir_files))

    # Stack into batches
    images = torch.cat(image_list, dim=0)
    masks = torch.cat(mask_list, dim=0)

    # Stack layers
    layers_dict = {}
    for layer_name, tensor_list in all_layers.items():
        if tensor_list:
            layers_dict[layer_name] = torch.cat(tensor_list, dim=0)

    cryptomatte_dict = {}
    for crypto_name, tensor_list in all_cryptomatte.items():
        if tensor_list:
            cryptomatte_dict[crypto_name] = torch.cat(tensor_list, dim=0)

    # Build metadata
    metadata = {
        "file_type": file_type.value,
        "frame_count": len(dir_files),
        "width": target_width,
        "height": target_height,
        "is_exr": is_exr,
        "layer_names": list(layers_dict.keys()),
        "cryptomatte_names": list(cryptomatte_dict.keys()),
    }

    return (
        images,
        masks,
        len(dir_files),
        layers_dict,
        cryptomatte_dict,
        json.dumps(metadata),
    )


# =============================================================================
# Node Class
# =============================================================================

class ReadNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {
                    "default": "",
                    "placeholder": "X://path/to/images",
                    "vhs_path_extensions": []
                }),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                "skip_first_images": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": BIGMAX, "step": 1}),
                "normalize": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "DICT", "DICT", "STRING")
    RETURN_NAMES = ("images", "masks", "frame_count", "layers", "cryptomatte", "metadata")
    FUNCTION = "load_images"
    CATEGORY = "project"

    def load_images(self, directory: str, image_load_cap: int = 0, skip_first_images: int = 0,
                    select_every_nth: int = 1, normalize: bool = False, unique_id=None, **kwargs):
        directory = strip_path(directory)
        validation = validate_load_images(directory)
        if validation != True:
            raise Exception(validation)

        result = load_images_with_layers(
            directory,
            image_load_cap=image_load_cap,
            skip_first_images=skip_first_images,
            select_every_nth=select_every_nth,
            normalize=normalize,
            unique_id=unique_id,
        )

        # Generate execution preview from first frame
        images = result[0]
        preview = generate_preview(images, directory)

        if preview:
            return {"ui": {"images": preview}, "result": result}
        return result

    @classmethod
    def IS_CHANGED(s, directory: str, **kwargs):
        if directory is None:
            return "input"
        return is_changed_load_images(directory, **kwargs)

    @classmethod
    def VALIDATE_INPUTS(s, directory: str, **kwargs):
        if directory is None:
            return True
        return validate_load_images(directory)


NODE_CLASS_MAPPINGS = {"ReadNode": ReadNode}
NODE_DISPLAY_NAME_MAPPINGS = {"ReadNode": "Read"}
