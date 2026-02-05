"""
Read Node - Self-contained image sequence loader
Based on VHS Load Images but with no external dependencies
"""

import os
import hashlib
import shutil
import subprocess
import asyncio
from io import BytesIO
import numpy as np
import torch
from PIL import Image, ImageOps
import itertools

import folder_paths
from comfy.utils import common_upscale, ProgressBar
from aiohttp import web
from server import PromptServer

# Constants
BIGMAX = (2**53-1)
IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif', '.webp'}

# Find ffmpeg (using imageio_ffmpeg like VHS does)
ffmpeg_path = None
try:
    from imageio_ffmpeg import get_ffmpeg_exe
    ffmpeg_path = get_ffmpeg_exe()
except:
    ffmpeg_path = shutil.which("ffmpeg")


# API endpoint for path autocomplete
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
                if valid_extensions is None or item.name.split(".")[-1].lower() in valid_extensions:
                    valid_items.append(item.name)
            except OSError:
                pass
    except PermissionError:
        return web.json_response([])

    valid_items.sort()
    return web.json_response(valid_items)


# API endpoint for preview image (static)
@PromptServer.instance.routes.get("/read_node/preview")
async def get_preview(request):
    """Return first image from directory as preview."""
    directory = request.query.get("directory", "")

    if not directory or not os.path.isdir(directory):
        return web.Response(status=404, text="Directory not found")

    files = []
    for f in sorted(os.listdir(directory)):
        ext = os.path.splitext(f)[1].lower()
        if ext in IMG_EXTENSIONS:
            files.append(os.path.join(directory, f))

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


# API endpoint for animated video preview
@PromptServer.instance.routes.get("/read_node/viewvideo")
async def view_video(request):
    """Generate animated preview from image sequence using ffmpeg."""
    query = request.rel_url.query
    directory = query.get("filename", "").strip().strip('"')

    if not directory or not os.path.isdir(directory):
        return web.Response(status=404)

    if ffmpeg_path is None:
        return web.Response(status=500, text="ffmpeg not found")

    # Get sorted image files
    skip_first = int(query.get("skip_first_images", 0))
    select_nth = int(query.get("select_every_nth", 1)) or 1

    valid_images = get_sorted_dir_files_from_directory(directory, skip_first, select_nth, IMG_EXTENSIONS)

    if not valid_images:
        return web.Response(status=204)

    # Create concat file for ffmpeg
    os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
    concat_file = os.path.join(folder_paths.get_temp_directory(), "read_node_preview.txt")

    with open(concat_file, "w") as f:
        f.write("ffconcat version 1.0\n")
        for path in valid_images:
            f.write(f"file '{os.path.abspath(path)}'\n")
            f.write("duration 0.125\n")

    # Build ffmpeg command
    args = [ffmpeg_path, "-v", "error", "-safe", "0", "-i", concat_file]

    # Handle resize
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


def strip_path(path):
    """Strip whitespace and quotes from path."""
    if path is None:
        return None
    path = path.strip()
    if path.startswith("\""):
        path = path[1:]
    if path.endswith("\""):
        path = path[:-1]
    return path


def calculate_file_hash(filename: str):
    """Calculate hash based on filename and modification time."""
    h = hashlib.sha256()
    h.update(filename.encode())
    h.update(str(os.path.getmtime(filename)).encode())
    return h.hexdigest()


def get_sorted_dir_files_from_directory(directory: str, skip_first_images: int = 0, select_every_nth: int = 1, extensions=None):
    """Get sorted list of image files from directory."""
    directory = strip_path(directory)
    if not os.path.isdir(directory):
        return []

    dir_files = os.listdir(directory)
    dir_files = sorted(dir_files)
    dir_files = [os.path.join(directory, x) for x in dir_files]
    dir_files = list(filter(lambda filepath: os.path.isfile(filepath), dir_files))

    if extensions is not None:
        extensions = list(extensions)
        new_dir_files = []
        for filepath in dir_files:
            ext = "." + filepath.split(".")[-1]
            if ext.lower() in extensions:
                new_dir_files.append(filepath)
        dir_files = new_dir_files

    dir_files = dir_files[skip_first_images:]
    dir_files = dir_files[0::select_every_nth]
    return dir_files


def is_changed_load_images(directory: str, image_load_cap: int = 0, skip_first_images: int = 0, select_every_nth: int = 1, **kwargs):
    """Check if inputs have changed for ComfyUI caching."""
    if not os.path.isdir(directory):
        return False

    dir_files = get_sorted_dir_files_from_directory(directory, skip_first_images, select_every_nth, IMG_EXTENSIONS)
    if image_load_cap != 0:
        dir_files = dir_files[:image_load_cap]

    m = hashlib.sha256()
    for filepath in dir_files:
        m.update(calculate_file_hash(filepath).encode())
    return m.digest().hex()


def validate_load_images(directory: str):
    """Validate directory input."""
    if not os.path.isdir(directory):
        return f"Directory '{directory}' cannot be found."
    dir_files = os.listdir(directory)
    if len(dir_files) == 0:
        return f"No files in directory '{directory}'."
    return True


def images_generator(directory: str, image_load_cap: int = 0, skip_first_images: int = 0, select_every_nth: int = 1, meta_batch=None, unique_id=None):
    """Generator that yields images one by one."""
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory '{directory}' cannot be found.")

    dir_files = get_sorted_dir_files_from_directory(directory, skip_first_images, select_every_nth, IMG_EXTENSIONS)

    if len(dir_files) == 0:
        raise FileNotFoundError(f"No files in directory '{directory}'.")

    if image_load_cap > 0:
        dir_files = dir_files[:image_load_cap]

    sizes = {}
    has_alpha = False
    for image_path in dir_files:
        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        has_alpha |= 'A' in i.getbands()
        count = sizes.get(i.size, 0)
        sizes[i.size] = count + 1

    size = max(sizes.items(), key=lambda x: x[1])[0]

    yield size[0], size[1], has_alpha
    if meta_batch is not None:
        yield min(image_load_cap, len(dir_files)) or len(dir_files)

    iformat = "RGBA" if has_alpha else "RGB"

    def load_image(file_path):
        i = Image.open(file_path)
        i = ImageOps.exif_transpose(i)
        i = i.convert(iformat)
        i = np.array(i, dtype=np.float32)
        i /= 255.0

        if i.shape[0] != size[1] or i.shape[1] != size[0]:
            t = torch.from_numpy(i).movedim(-1, 0).unsqueeze(0)
            t = common_upscale(t, size[0], size[1], "lanczos", "center")
            i = t.squeeze(0).movedim(0, -1).numpy()

        if has_alpha:
            i[:, :, -1] = 1 - i[:, :, -1]
        return i

    total_images = len(dir_files)
    processed_images = 0
    pbar = ProgressBar(total_images)
    images = map(load_image, dir_files)

    try:
        prev_image = next(images)
        while True:
            next_image = next(images)
            yield prev_image
            processed_images += 1
            pbar.update_absolute(processed_images, total_images)
            prev_image = next_image
    except StopIteration:
        pass

    if meta_batch is not None:
        meta_batch.inputs.pop(unique_id)
        meta_batch.has_closed_inputs = True

    if prev_image is not None:
        yield prev_image


def load_images(directory: str, image_load_cap: int = 0, skip_first_images: int = 0, select_every_nth: int = 1, meta_batch=None, unique_id=None):
    """Load images from directory and return batched tensors."""
    if meta_batch is None or unique_id not in meta_batch.inputs:
        gen = images_generator(directory, image_load_cap, skip_first_images, select_every_nth, meta_batch, unique_id)
        (width, height, has_alpha) = next(gen)
        if meta_batch is not None:
            meta_batch.inputs[unique_id] = (gen, width, height, has_alpha)
            meta_batch.total_frames = min(meta_batch.total_frames, next(gen))
    else:
        gen, width, height, has_alpha = meta_batch.inputs[unique_id]

    if meta_batch is not None:
        gen = itertools.islice(gen, meta_batch.frames_per_batch)

    images = torch.from_numpy(np.fromiter(gen, np.dtype((np.float32, (height, width, 3 + has_alpha)))))

    if has_alpha:
        masks = images[:, :, :, 3]
        images = images[:, :, :, :3]
    else:
        masks = torch.zeros((images.size(0), 64, 64), dtype=torch.float32, device="cpu")

    if len(images) == 0:
        raise FileNotFoundError(f"No images could be loaded from directory '{directory}'.")

    return images, masks, images.size(0)


class ReadNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": "", "placeholder": "X://path/to/images", "vhs_path_extensions": []}),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                "skip_first_images": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": BIGMAX, "step": 1}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    RETURN_NAMES = ("images", "masks", "frame_count")
    FUNCTION = "load_images"
    CATEGORY = "project"

    def load_images(self, directory: str, **kwargs):
        directory = strip_path(directory)
        if directory is None or validate_load_images(directory) != True:
            raise Exception("Directory is not valid: " + str(directory))
        return load_images(directory, **kwargs)

    @classmethod
    def IS_CHANGED(s, directory: str, **kwargs):
        if directory is None:
            return "input"
        return is_changed_load_images(strip_path(directory), **kwargs)

    @classmethod
    def VALIDATE_INPUTS(s, directory: str, **kwargs):
        if directory is None:
            return True
        return validate_load_images(strip_path(directory))


NODE_CLASS_MAPPINGS = {"ReadNode": ReadNode}
NODE_DISPLAY_NAME_MAPPINGS = {"ReadNode": "Read"}
