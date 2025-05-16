# node_florence_tile_captioner.py

import torch
from PIL import Image
import torchvision.transforms as T

# Replace with the actual Florence import path if needed
from nodes import NODE_CLASS_MAPPINGS as florence_nodes

class FlorenceTileCaptioner:
    CATEGORY = "Florence/Utility"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "tile_height": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "overlap": ("INT", {"default": 32, "min": 0, "max": 256}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("tile_images", "tile_captions",)

    FUNCTION = "process"

    def process(self, image, tile_width, tile_height, overlap):
        B, C, H, W = image.shape
        tiles = []
        captions = []
        to_pil = T.ToPILImage()

        # Use FlorenceCaption from installed ComfyUI_Florence
        florence_caption = florence_nodes["Florence Caption"]()

        for y in range(0, H - tile_height + 1, tile_height - overlap):
            for x in range(0, W - tile_width + 1, tile_width - overlap):
                tile = image[:, :, y:y+tile_height, x:x+tile_width]
                tile_img = to_pil(tile[0].cpu())
                caption = florence_caption.caption_image(tile_img)
                tiles.append(tile)
                captions.append(caption)

        tile_stack = torch.cat(tiles, dim=0)
        return (tile_stack, captions)

NODE_CLASS_MAPPINGS = {
    "FlorenceTileCaptioner": FlorenceTileCaptioner
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlorenceTileCaptioner": "Florence Tile Captioner"
}
