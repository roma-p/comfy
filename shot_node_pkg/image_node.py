from aiohttp import web
from server import PromptServer
from .pipe_globals import get_pipe

PLACEHOLDER = "-- Select --"


@PromptServer.instance.routes.get("/image_node/fields")
async def get_all_fields(request):
    return web.json_response(list(get_pipe().list_all_image_fields()))


@PromptServer.instance.routes.get("/image_node/visible_fields/{image_id}")
async def get_visible_fields(request):
    image_id = request.match_info.get("image_id")
    visible = get_pipe().get_image_id_dict(image_id)
    return web.json_response(list(visible))


class ImageNode:
    @classmethod
    def INPUT_TYPES(cls):
        image_ids = [PLACEHOLDER] + list(get_pipe().list_image_id())
        all_fields = get_pipe().list_all_image_fields()
        optional = {field: ("STRING", {"default": ""}) for field in all_fields}
        return {
            "required": {
                "image_id": (image_ids,),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("folder_path",)
    FUNCTION = "execute"
    CATEGORY = "project"

    @classmethod
    def VALIDATE_INPUTS(cls, image_id, **kwargs):
        if image_id == PLACEHOLDER:
            return "Please select an image_id"
        visible_fields = get_pipe().get_image_id_dict(image_id)
        for field in visible_fields:
            if not kwargs.get(field):
                return f"Field '{field}' is required for image_id '{image_id}'"
        return True

    def execute(self, image_id, **kwargs):
        visible_fields = get_pipe().get_image_id_dict(image_id)
        resolved_kwargs = {k: v for k, v in kwargs.items() if k in visible_fields}
        folder_path = get_pipe().resolve_image_folder(image_id, **resolved_kwargs)
        return (folder_path,)


NODE_CLASS_MAPPINGS = {"ImageNode": ImageNode}
NODE_DISPLAY_NAME_MAPPINGS = {"ImageNode": "Image"}
