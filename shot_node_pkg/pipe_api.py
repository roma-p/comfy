class Pipe:
    def __init__(self):
        self.shot_dict = {"seq_0": ("sh_0_0", "sh_0_1")}
        self.image_id_dict = {
            "3d": {"task", "camera", "aov", "colorspace"},
            "2d": {"task", "write", "aov", "colorspace"},
            "proxy": {"task", "colorspace"},
        }

    def get_seq_tuple(self):
        i = len(self.shot_dict.keys())
        self.shot_dict[f"seq_{i}"] = (f"sh_{i}_0", f"sh_{i}_1")
        return tuple(self.shot_dict.keys())

    def get_shot_tuple(self, seq):
        return self.shot_dict[seq]

    def list_image_id(self):
        return tuple(self.image_id_dict.keys())

    def get_image_id_dict(self, image_id):
        return self.image_id_dict.get(image_id, set())

    def list_all_image_fields(self):
        return set().union(*self.image_id_dict.values())

    def resolve_image_folder(self, *args, **kwargs):
        # Mock implementation - returns ComfyUI input folder for testing
        # Real implementation would resolve based on image_id, task, camera, etc.
        import folder_paths
        print(f"resolve_image_folder called with args={args}, kwargs={kwargs}")
        return folder_paths.get_input_directory()
