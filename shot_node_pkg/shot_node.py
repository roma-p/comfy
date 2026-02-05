from aiohttp import web
from server import PromptServer
from .pipe_globals import get_pipe

PLACEHOLDER = "-- Select --"


@PromptServer.instance.routes.get("/shot_node/sequences")
async def get_sequences(request):
    return web.json_response(list(get_pipe().get_seq_tuple()))


@PromptServer.instance.routes.get("/shot_node/shots/{sequence}")
async def get_shots(request):
    sequence = request.match_info.get("sequence")
    try:
        return web.json_response(list(get_pipe().get_shot_tuple(sequence)))
    except KeyError:
        return web.json_response({"error": f"Unknown sequence: {sequence}"}, status=404)


class ShotNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sequence": ([PLACEHOLDER] + list(get_pipe().get_seq_tuple()),),
                "shot": ([PLACEHOLDER],),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = "project"

    @classmethod
    def VALIDATE_INPUTS(cls, sequence, shot):
        if sequence == PLACEHOLDER:
            return "Please select a sequence"
        if shot == PLACEHOLDER:
            return "Please select a shot"
        # Validate against actual Pipe data
        valid_shots = get_pipe().get_shot_tuple(sequence)
        if shot not in valid_shots:
            return f"Invalid shot '{shot}' for sequence '{sequence}'"
        return True

    def execute(self, sequence, shot):
        print(f"Selected: {sequence} / {shot}")
        return {}


NODE_CLASS_MAPPINGS = {"ShotNode": ShotNode}
NODE_DISPLAY_NAME_MAPPINGS = {"ShotNode": "Shot"}
