import torch
from PIL import Image
import torchvision.transforms as T
# Transformers library is needed for AutoModelForCausalLM, AutoProcessor
# Make sure it's available in your ComfyUI's Python environment.
# If using comfyui_Florence2_nodes by huchenlei, it should already be a dependency.

class FlorenceTileCaptioner_v2:
    CATEGORY = "Florence/Experimental Tiling"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "fl2_model": ("FL2_MODEL",),  # Input for the loaded Florence model
                "fl2_processor": ("FL2_PROCESSOR",),  # Input for the loaded Florence processor
                "task_prompt_for_tiles": ("STRING", {"default": "<CAPTION>", "multiline": False, "dynamicPrompts": False,
                                                     "tooltip": "Florence-2 task prompt for each tile, e.g., <CAPTION>, <DETAILED_CAPTION>"}),
                "tile_width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "tile_height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 32,
                                    "tooltip": "Overlap between tiles. Helps maintain context but increases processing."}),
                "max_new_tokens": ("INT", {"default": 128, "min": 1, "max": 1024}),
                "num_beams": ("INT", {"default": 3, "min": 1, "max": 10}),
                "batch_size_florence": ("INT", {"default": 1, "min": 1, "max": 16, 
                                                "tooltip": "How many tiles to send to Florence-2 at once. Increase if you have VRAM."})
            },
            "optional": {
                "text_input_for_task": ("STRING", {"default": "", "multiline": False, "dynamicPrompts": False,
                                                   "tooltip": "Optional text input if the task_prompt requires it (e.g., for OVS 'an object a, an object b')."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LIST_STRING",) # LIST_STRING will be a Python list of strings
    RETURN_NAMES = ("tile_images_batch", "tile_captions_list",)
    FUNCTION = "generate_tile_captions"

    def _generate_captions_batched(self, model, processor, pil_images, task_prompt, text_input, max_new_tokens, num_beams):
        if not pil_images:
            return []

        if task_prompt == "<OCR_WITH_REGION>" or (text_input and task_prompt != "<MORE_DETAILED_CAPTION>"):
            # Tasks that require specific text input formatting per image
            # For simplicity in this batched version, we'll assume text_input is generic or we handle it carefully
            # This part might need refinement if those tasks are commonly used for tiles.
            # For now, we assume a general captioning task or that text_input applies to all.
            processed_inputs = [processor(text=f"{task_prompt}{text_input if text_input else ''}", images=img, return_tensors="pt") for img in pil_images]
            input_ids_list = [inp["input_ids"].to(model.device) for inp in processed_inputs]
            pixel_values_list = [inp["pixel_values"].to(model.device) for inp in processed_inputs]
            
            # Pad if necessary for batching (simplification: assuming processor handles padding or we use batch_size=1 for these tasks)
            # Proper batching for heterogeneous inputs requires careful padding with processor.
            # For now, let's assume they are processed one by one if tasks are complex, or processor handles it.
            # This example will send them one by one if the task is complex.
            if len(pil_images) > 1 and (task_prompt == "<OCR_WITH_REGION>" or (text_input and task_prompt != "<MORE_DETAILED_CAPTION>")): # Crude check
                print(f"Warning: Batching complex tasks like {task_prompt} with different text inputs per image is tricky. Processing one by one for safety.")
                all_captions = []
                for img in pil_images:
                     all_captions.extend(self._generate_captions_batched(model, processor, [img], task_prompt, text_input, max_new_tokens, num_beams))
                return all_captions


            inputs = processor(text=[f"{task_prompt}{text_input if text_input else ''}"] * len(pil_images), images=pil_images, return_tensors="pt", padding=True, truncation=True).to(model.device)

        else: # Standard captioning tasks
            inputs = processor(text=[task_prompt] * len(pil_images), images=pil_images, return_tensors="pt", padding=True, truncation=True).to(model.device)

        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False # Usually False for consistent captioning
        )
        
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=False)
        
        final_captions = []
        for i, text in enumerate(generated_texts):
            # The post_process_generation function expects image_size for some tasks.
            original_pil_image = pil_images[i]
            image_size = (original_pil_image.width, original_pil_image.height)
            
            # This is the key part: using the processor's designated post-processing
            # For many captioning tasks, this should extract the clean caption.
            # The task string must exactly match what Florence-2 expects for its internal routing.
            try:
                parsed_output = processor.post_process_generation(text, task=task_prompt, image_size=image_size)
                
                # Extract the caption string from the parsed output. This varies by task.
                # For "<CAPTION>", it's usually directly the string or in a dict like {'<CAPTION>': 'caption text'}
                # For "Dense Region Caption", it returns a dict like {'<DENSE_REGION_CAPTION>': [['caption1', [box1]], ...]}
                # We want a single representative string caption per tile.
                if task_prompt in ["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"]:
                    if isinstance(parsed_output, dict) and task_prompt in parsed_output:
                        final_captions.append(str(parsed_output[task_prompt]))
                    elif isinstance(parsed_output, str):
                        final_captions.append(parsed_output)
                    else: # Fallback
                        final_captions.append(str(parsed_output)) # Fallback, might need cleaning
                elif task_prompt == "<DENSE_REGION_CAPTION>": # Returns list of captions/boxes
                     # For a single tile, concatenate or take the most prominent? Let's concatenate.
                    if isinstance(parsed_output, dict) and task_prompt in parsed_output:
                        all_region_captions = [item[0] for item in parsed_output[task_prompt]]
                        final_captions.append(". ".join(all_region_captions))
                    else:
                        final_captions.append(str(parsed_output)) # Fallback
                elif task_prompt == "<OCR>" or task_prompt == "<OCR_WITH_REGION>":
                    if isinstance(parsed_output, dict) and task_prompt in parsed_output:
                         # OCR output can be complex, e.g., list of text lines and their boxes.
                         # For a single prompt, concatenate recognized text.
                        ocr_texts = []
                        for item_list in parsed_output[task_prompt]: # item_list might be [[box1, text1], [box2, text2]]
                            for item in item_list: # item is [box, text] or just text
                                if isinstance(item, list) and len(item) == 2:
                                    ocr_texts.append(item[1]) # Text is the second element
                                elif isinstance(item, str):
                                    ocr_texts.append(item)
                        final_captions.append(" ".join(ocr_texts) if ocr_texts else "No text found.")
                    else:
                        final_captions.append(str(parsed_output))
                else: # For other tasks or fallback
                    final_captions.append(str(parsed_output)) # May require manual parsing by user later
            except Exception as e:
                print(f"Error during Florence-2 post_process_generation for task '{task_prompt}': {e}")
                print(f"Raw generated text was: {text}")
                final_captions.append(text) # Fallback to raw text

        return final_captions

    def generate_tile_captions(self, image: torch.Tensor, fl2_model, fl2_processor, task_prompt_for_tiles,
                               tile_width, tile_height, overlap,
                               max_new_tokens, num_beams, batch_size_florence, text_input_for_task=None):
        
        # Ensure image is B,C,H,W and on CPU for tiling logic, then move tiles to model device for processing
        if image.dim() == 3: # H, W, C (from LoadImage)
            image = image.unsqueeze(0) # B, H, W, C
        
        image_bchw = image.permute(0, 3, 1, 2).cpu() # B, C, H, W on CPU

        if image_bchw.shape[0] > 1:
            print(f"Warning: FlorenceTileCaptioner_v2 received batch of {image_bchw.shape[0]} images. Processing only the first one.")
            image_bchw = image_bchw[0].unsqueeze(0)

        single_image_chw = image_bchw[0] # C, H, W
        C, H, W = single_image_chw.shape

        output_tile_tensors_chw = [] # Store original tile tensors (CHW)
        pil_tiles_for_florence = []  # Store PIL images for Florence batch

        # T.ToPILImage() expects CHW tensor
        pil_transform = T.ToPILImage()

        step_y = tile_height - overlap
        step_x = tile_width - overlap
        if step_y <= 0: step_y = tile_height # Avoid issues if overlap is too large
        if step_x <= 0: step_x = tile_width

        for y in range(0, H, step_y):
            for x in range(0, W, step_x):
                # Define the tile boundaries
                y1, x1 = y, x
                y2, x2 = min(y + tile_height, H), min(x + tile_width, W)
                
                # Ensure minimum tile size (e.g. if edge tile is too small)
                # This might be better handled by padding the original image.
                # For now, we just take what's there.
                if (y2 - y1) < 32 or (x2 - x1) < 32: # Skip very small slivers
                    continue

                tile_tensor_chw = single_image_chw[:, y1:y2, x1:x2]
                output_tile_tensors_chw.append(tile_tensor_chw.unsqueeze(0)) # Add batch dim for later cat

                tile_pil = pil_transform(tile_tensor_chw)
                if tile_pil.mode != "RGB":
                    tile_pil = tile_pil.convert("RGB")
                pil_tiles_for_florence.append(tile_pil)

        if not output_tile_tensors_chw: # Image was too small for any tiles
            print("Warning: Image too small for tiling or no valid tiles found. Captioning whole image.")
            tile_pil = pil_transform(single_image_chw)
            if tile_pil.mode != "RGB":
                tile_pil = tile_pil.convert("RGB")
            pil_tiles_for_florence.append(tile_pil)
            output_tile_tensors_chw.append(single_image_chw.unsqueeze(0))

        # Batch process PIL images with Florence-2
        all_captions = []
        for i in range(0, len(pil_tiles_for_florence), batch_size_florence):
            batch_pil_images = pil_tiles_for_florence[i:i + batch_size_florence]
            captions_batch = self._generate_captions_batched(fl2_model, fl2_processor, batch_pil_images,
                                                             task_prompt_for_tiles, text_input_for_task,
                                                             max_new_tokens, num_beams)
            all_captions.extend(captions_batch)
            print(f"Processed tile batch {i//batch_size_florence + 1}/{(len(pil_tiles_for_florence)-1)//batch_size_florence + 1}. Captions: {captions_batch}")


        # Concatenate all tile tensors (CHW) into a single batch tensor (B,C,H,W)
        # Tiles might have different sizes if original image wasn't perfectly divisible.
        # KSampler batches usually require same-sized inputs. This could be an issue.
        # For now, we output them as is. Downstream nodes must handle potentially varied sizes or use a mode that pads.
        # However, typical tile processing aims for same-sized tiles.
        # The tiling logic above extracts tiles that can be smaller at edges.
        # To ensure same size, padding original image or resizing tiles would be needed.
        # Let's assume for now that downstream can handle it, or user sets tile_width/height that divides image.
        # Or, a more robust tiling would pad the source image first.

        # For typical ComfyUI KSampler input, images are B H W C
        # Our output_tile_tensors_chw are list of [1,C,H,W]. Cat them to [N,C,H,W]
        # then permute to [N,H,W,C] for typical image outputs.
        batched_tiles_bchw = torch.cat(output_tile_tensors_chw, dim=0)
        batched_tiles_bhwc = batched_tiles_bchw.permute(0, 2, 3, 1) # N, H, W, C

        return (batched_tiles_bhwc, all_captions) # all_captions is a Python list of strings

NODE_CLASS_MAPPINGS = {
    "FlorenceTileCaptioner_v2": FlorenceTileCaptioner_v2
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FlorenceTileCaptioner_v2": "Florence Tile Captioner v2"
}