import os
import json
import hashlib
from io import BytesIO
import torch
from PIL import Image, ImageOps

from comfy.utils import ProgressBar
from aiohttp import web
from server import PromptServer

# Import from local modules
from .utils.file_utils import (
    strip_path, get_extension, has_extension, calculate_file_hash,
    IMG_EXTENSIONS, EXR_EXTENSIONS, ALL_EXTENSIONS
)
from .utils.sequence_utils import (
    has_sequence_pattern, detect_sequences,
    resolve_sequence_files, detect_file_type_from_path
)
from .utils.preview_utils import generate_preview_animated, generate_preview_static
from .loaders import EXR_AVAILABLE
if EXR_AVAILABLE:
    from .loaders import ExrLoader


BIGMAX = (2**53 - 1)


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

    # Parse extensions filter from query (comma-separated or None for all)
    ext_param = query.get("extensions")
    extensions = set(ext_param.split(',')) if ext_param else None

    valid_items = []
    try:
        for item in os.scandir(path):
            try:
                if item.is_dir():
                    valid_items.append(item.name + "/")
                elif has_extension(item.name, extensions):
                    valid_items.append(item.name)
            except OSError:
                pass
    except PermissionError:
        return web.json_response([])

    valid_items.sort()
    return web.json_response(valid_items)


@PromptServer.instance.routes.get("/read_node/detect_type")
async def detect_type_endpoint(request):
    """Detect file type from sequence pattern or directory (standard vs EXR)."""
    query = request.rel_url.query
    path = query.get("path", "").strip().strip('"')

    if not path:
        return web.json_response({"type": "unknown", "exr_available": EXR_AVAILABLE})

    file_type = detect_file_type_from_path(path)

    return web.json_response({
        "type": file_type,
        "exr_available": EXR_AVAILABLE,
    })


@PromptServer.instance.routes.get("/read_node/detect_sequences")
async def detect_sequences_endpoint(request):
    """Detect image sequences in a directory and return suggested patterns."""
    query = request.rel_url.query
    directory = query.get("path", "").strip().strip('"')

    sequences = detect_sequences(directory)

    # Convert to JSON-serializable format, sorted by frame count (most frames first)
    result = sorted(
        [
            {
                "pattern": seq.to_pattern_string(),
                "full_path": seq.to_full_path(),
                "frame_count": seq.frame_count,
                "first_frame": seq.first_frame,
                "last_frame": seq.last_frame,
                "padding": seq.padding,
            }
            for seq in sequences
        ],
        key=lambda x: -x["frame_count"]
    )

    return web.json_response({"sequences": result})


@PromptServer.instance.routes.get("/read_node/preview")
async def resolve_preview_exr(request):
    """Return first image from EXR sequence as static JPEG preview."""
    sequence_path = request.query.get("sequence_path", "")

    if not sequence_path:
        return web.Response(status=404, text="No sequence path provided")

    # Get first file only (limit=1)
    files, _ = resolve_sequence_files(sequence_path, extensions=IMG_EXTENSIONS, limit=1)

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


@PromptServer.instance.routes.get("/read_node/animated_preview")
async def resolve_animated_preview(request):
    """Generate animated webm preview from standard image sequence using ffmpeg."""
    query = request.rel_url.query
    sequence_path = query.get("filename", "").strip().strip('"')

    if not sequence_path:
        return web.Response(status=404)

    skip_first = int(query.get("skip_first_images", 0))
    select_nth = int(query.get("select_every_nth", 1)) or 1

    # Get images (only standard formats for video preview)
    valid_images, _ = resolve_sequence_files(
        sequence_path,
        skip_first=skip_first,
        select_every_nth=select_nth,
        extensions=IMG_EXTENSIONS
    )

    force_size = query.get("force_size", "") or None
    return await generate_preview_animated(valid_images, request, force_size=force_size)


# =============================================================================
# Image Loading Functions
# =============================================================================

def is_changed_load_images(sequence_path: str, image_load_cap: int = 0, skip_first_images: int = 0,
                           select_every_nth: int = 1, normalize: bool = False, **kwargs):
    """Check if inputs have changed for ComfyUI caching."""
    if not sequence_path:
        return False

    all_files, _ = resolve_sequence_files(
        sequence_path,
        skip_first=skip_first_images,
        select_every_nth=select_every_nth,
        limit=image_load_cap
    )

    if not all_files:
        return False

    m = hashlib.sha256()
    m.update(f"normalize={normalize}".encode())
    for filepath in all_files:
        m.update(calculate_file_hash(filepath).encode())
    return m.digest().hex()


def validate_load_images(sequence_path: str):
    """Validate sequence path input."""
    if not sequence_path or not strip_path(sequence_path):
        return "No sequence path provided."

    files, file_type = resolve_sequence_files(sequence_path, limit=1)

    if file_type == "unknown" and not has_sequence_pattern(sequence_path):
        return f"Path '{sequence_path}' is not a valid directory or sequence pattern."

    if not files:
        return f"No valid files found for '{sequence_path}'."

    return True


def load_images_with_layers(sequence_path: str, image_load_cap: int = 0, skip_first_images: int = 0,
                            select_every_nth: int = 1, normalize: bool = False):
    """
    Load images from sequence pattern or directory with full layer support.

    Args:
        sequence_path: Sequence pattern (e.g., /path/to/image.####.exr) or directory path

    Returns:
        Tuple of (images, masks, frame_count, layers, cryptomatte, metadata)
    """
    # Resolve all files using unified helper
    dir_files, file_type = resolve_sequence_files(
        sequence_path,
        skip_first=skip_first_images,
        select_every_nth=select_every_nth,
        limit=image_load_cap
    )

    if not dir_files:
        raise FileNotFoundError(f"No valid files found for '{sequence_path}'")

    # Determine loader from first file
    first_file = dir_files[0]
    loader = resolve_loader(first_file)
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

    # Build metadata (Option B: flat with type discriminator)
    if is_exr:
        # EXR format: CoCoTools-compatible
        metadata = {
            "type": "exr",
            "width": target_width,
            "height": target_height,
            "frame_count": len(dir_files),
            # EXR-specific fields from loader
            "subimages": first_result.metadata.get("subimages", []),
            "is_multipart": first_result.metadata.get("is_multipart", False),
            "subimage_count": first_result.metadata.get("subimage_count", 1),
            "layer_names": list(layers_dict.keys()),
            "cryptomatte_names": list(cryptomatte_dict.keys()),
        }
    else:
        # Standard format: PNG/JPG/etc
        metadata = {
            "type": "standard",
            "width": target_width,
            "height": target_height,
            "frame_count": len(dir_files),
            # Standard-specific fields from loader
            "format": first_result.metadata.get("format", "unknown"),
            "mode": first_result.metadata.get("mode", "RGB"),
            "bit_depth": first_result.metadata.get("bit_depth", 8),
            "color_profile": first_result.metadata.get("color_profile"),
            "has_alpha": first_result.has_alpha,
        }

    return (
        images,
        masks,
        len(dir_files),
        json.dumps(metadata),
        layers_dict,
        cryptomatte_dict,
    )


# =============================================================================
# Node Class
# =============================================================================

class ReadNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sequence_path": ("STRING", {
                    "default": "",
                    "placeholder": "/path/to/image.####.exr",
                    "vhs_path_extensions": []
                }),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                "skip_first_images": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": BIGMAX, "step": 1}),
                "normalize": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "STRING", "DICT", "DICT")
    RETURN_NAMES = ("images", "masks", "frame_count", "metadata", "layers", "cryptomatte")
    FUNCTION = "load_images"
    CATEGORY = "project"

    def load_images(self, sequence_path: str, image_load_cap: int = 0, skip_first_images: int = 0,
                    select_every_nth: int = 1, normalize: bool = False, **kwargs):
        validation = validate_load_images(sequence_path)
        if validation != True:
            raise Exception(validation)

        result = load_images_with_layers(
            sequence_path,
            image_load_cap=image_load_cap,
            skip_first_images=skip_first_images,
            select_every_nth=select_every_nth,
            normalize=normalize,
        )

        # Generate execution preview only for EXR (standard images have animated preview)
        metadata = json.loads(result[3])

        if metadata.get("type") == "exr":
            images = result[0]
            preview = generate_preview_static(images, sequence_path)
            if preview:
                return {"ui": {"images": preview}, "result": result}

        return result

    @classmethod
    def IS_CHANGED(cls, sequence_path: str, **kwargs):
        if sequence_path is None:
            return "input"
        return is_changed_load_images(sequence_path, **kwargs)

    @classmethod
    def VALIDATE_INPUTS(cls, sequence_path: str, **kwargs):
        if sequence_path is None:
            return True
        return validate_load_images(sequence_path)


NODE_CLASS_MAPPINGS = {"ReadNode": ReadNode}
NODE_DISPLAY_NAME_MAPPINGS = {"ReadNode": "Read"}
