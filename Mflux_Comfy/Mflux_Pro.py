import os
from PIL import Image, ImageOps
import folder_paths
import numpy as np
import torch

# Try to import ControlnetUtil from mflux; if unavailable or missing helpers, provide a local fallback
try:
    from mflux.controlnet.controlnet_util import ControlnetUtil  # type: ignore
    if not hasattr(ControlnetUtil, "preprocess_canny") or not hasattr(ControlnetUtil, "scale_image"):
        raise AttributeError("ControlnetUtil missing expected helpers")
except Exception:
    from PIL import ImageFilter, ImageOps

    class ControlnetUtil:  # type: ignore
        @staticmethod
        def preprocess_canny(img):
            gray = img.convert("L")
            edges = gray.filter(ImageFilter.FIND_EDGES)
            return ImageOps.invert(edges)

        @staticmethod
        def scale_image(h, w, img):
            return img.resize((w, h), Image.BICUBIC)

class MfluxImg2ImgPipeline:
    def __init__(self, image_path, image_strength):
        self.image_path = image_path
        self.image_strength = image_strength

    def clear_cache(self):
        self.image_path = None
        self.image_strength = None


def _make_oriented_copy(image_path: str) -> tuple[str, int, int]:
    """Load an image, apply EXIF-based transpose if needed, and save an oriented copy.

    Returns a tuple of (new_path, width, height). The new file is saved next to the
    original with a suffix "_oriented" to ensure it's readable by the backend later
    during generation.
    """
    try:
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img)
            # Ensure a consistent mode for saving
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")
            width, height = img.size
            base = os.path.splitext(os.path.basename(image_path))[0]
            # Prefer PNG to avoid JPEG EXIF complexities in the copy
            out_dir = os.path.dirname(image_path)
            candidate = os.path.join(out_dir, f"{base}_oriented.png")
            # Avoid accidental overwrite of an existing oriented copy
            if os.path.exists(candidate):
                i = 1
                while True:
                    alt = os.path.join(out_dir, f"{base}_oriented_{i}.png")
                    if not os.path.exists(alt):
                        candidate = alt
                        break
                    i += 1
            img.save(candidate)
            return candidate, width, height
    except Exception:
        # On any failure, fall back to the original path and size
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception:
            width = height = 0
        return image_path, width, height

class MfluxImg2Img:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

        return {
            "required": {
                "image": (sorted(files), {"image_upload": True, "tooltip": "Choose the starting image for img2img."}),
                "image_strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "How strongly to follow the input image. Lower = more creative, Higher = closer to input."}),
            },
            "optional": {
                "resize_mode": ([
                    "Original",
                    "Fit (keep ratio)",
                    "Fill (crop, keep ratio)",
                    "Long side",
                    "Short side",
                    "Exact (stretch)",
                ], {"default": "Original", "tooltip": "How to adapt the input image to target size before generation."}),
                "target_width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8, "tooltip": "Target width (used by Fit/Fill/Exact; by Long side as scale if selected)."}),
                "target_height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8, "tooltip": "Target height (used by Fit/Fill/Exact; by Short side as scale if selected)."}),
                "round_multiple": ("INT", {"default": 8, "min": 1, "max": 64, "tooltip": "Round final size to nearest multiple. Use 8 for best backend performance."}),
                "crop_anchor": ([
                    "Center","Top","Bottom","Left","Right",
                    "Top-Left","Top-Right","Bottom-Left","Bottom-Right"
                ], {"default": "Center", "tooltip": "Where to crop when using Fill (cover)."}),
            }
        }

    CATEGORY = "MFlux/Pro"
    RETURN_TYPES = ("MfluxImg2ImgPipeline", "INT", "INT")
    RETURN_NAMES = ("img2img", "width", "height")
    FUNCTION = "load_and_process"

    def load_and_process(self, image, image_strength, resize_mode="Original", target_width=512, target_height=512, round_multiple=8, crop_anchor="Center"):
        image_path = folder_paths.get_annotated_filepath(image)
        # Normalize orientation using EXIF and use a stable copy as the base
        oriented_path, width, height = _make_oriented_copy(image_path)

        def _round_dim(v: int, m: int) -> int:
            if m <= 1:
                return int(v)
            # round to nearest multiple of m
            return max(m, int(round(v / m) * m))

        def _crop_to_anchor(img: Image.Image, tw: int, th: int, anchor: str) -> Image.Image:
            w, h = img.size
            left = (w - tw) // 2
            top = (h - th) // 2
            if anchor in ("Top", "Top-Left", "Top-Right"):
                top = 0
            if anchor in ("Bottom", "Bottom-Left", "Bottom-Right"):
                top = h - th
            if anchor in ("Left", "Top-Left", "Bottom-Left"):
                left = 0
            if anchor in ("Right", "Top-Right", "Bottom-Right"):
                left = w - tw
            left = max(0, min(left, w - tw))
            top = max(0, min(top, h - th))
            return img.crop((left, top, left + tw, top + th))

        # Compute resize if requested
        mode = str(resize_mode or "Original")
        resized_path = oriented_path
        out_w, out_h = width, height
        try:
            if mode != "Original":
                with Image.open(oriented_path) as im:
                    im = ImageOps.exif_transpose(im)
                    ow, oh = im.size
                    if mode == "Fit (keep ratio)":
                        scale = min(target_width / ow, target_height / oh)
                        nw, nh = int(ow * scale), int(oh * scale)
                        nw, nh = _round_dim(nw, round_multiple), _round_dim(nh, round_multiple)
                        im = im.resize((max(1, nw), max(1, nh)), Image.BICUBIC)
                        out_w, out_h = im.size
                    elif mode == "Fill (crop, keep ratio)":
                        scale = max(target_width / ow, target_height / oh)
                        nw, nh = int(ow * scale), int(oh * scale)
                        im = im.resize((max(1, nw), max(1, nh)), Image.BICUBIC)
                        tw, th = _round_dim(target_width, round_multiple), _round_dim(target_height, round_multiple)
                        im = _crop_to_anchor(im, tw, th, crop_anchor)
                        out_w, out_h = im.size
                    elif mode == "Long side":
                        long_target = max(1, int(target_width))
                        if ow >= oh:
                            scale = long_target / ow
                        else:
                            scale = long_target / oh
                        nw, nh = _round_dim(int(ow * scale), round_multiple), _round_dim(int(oh * scale), round_multiple)
                        im = im.resize((max(1, nw), max(1, nh)), Image.BICUBIC)
                        out_w, out_h = im.size
                    elif mode == "Short side":
                        short_target = max(1, int(target_height))
                        if ow <= oh:
                            scale = short_target / ow
                        else:
                            scale = short_target / oh
                        nw, nh = _round_dim(int(ow * scale), round_multiple), _round_dim(int(oh * scale), round_multiple)
                        im = im.resize((max(1, nw), max(1, nh)), Image.BICUBIC)
                        out_w, out_h = im.size
                    elif mode == "Exact (stretch)":
                        tw, th = _round_dim(target_width, round_multiple), _round_dim(target_height, round_multiple)
                        im = im.resize((max(1, tw), max(1, th)), Image.BICUBIC)
                        out_w, out_h = im.size

                    base = os.path.splitext(os.path.basename(oriented_path))[0]
                    out_dir = os.path.dirname(oriented_path)
                    suffix_map = {
                        "Fit (keep ratio)": "fit",
                        "Fill (crop, keep ratio)": "fill",
                        "Long side": "long",
                        "Short side": "short",
                        "Exact (stretch)": "exact",
                    }
                    tag = suffix_map.get(mode, "resized")
                    candidate = os.path.join(out_dir, f"{base}_{tag}.png")
                    if os.path.exists(candidate):
                        i = 1
                        while True:
                            alt = os.path.join(out_dir, f"{base}_{tag}_{i}.png")
                            if not os.path.exists(alt):
                                candidate = alt
                                break
                            i += 1
                    im.save(candidate)
                    resized_path = candidate
        except Exception as e:
            # On any error, fall back to oriented image without resizing
            print(f"[MFlux-ComfyUI] Img2Img resize failed: {e}")
            resized_path = oriented_path
            out_w, out_h = width, height

        return MfluxImg2ImgPipeline(resized_path, image_strength), int(out_w), int(out_h)

    @classmethod
    def IS_CHANGED(cls, image, image_strength, resize_mode="Original", target_width=512, target_height=512, round_multiple=8, crop_anchor="Center"):
        image_hash = hash(image)
        strength_rounded = round(float(image_strength), 2)
        key = (image_hash, strength_rounded, str(resize_mode), int(target_width), int(target_height), int(round_multiple), str(crop_anchor))
        return key

    @classmethod
    def VALIDATE_INPUTS(cls, image, image_strength, resize_mode="Original", target_width=512, target_height=512, round_multiple=8, crop_anchor="Center"):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        if not isinstance(image_strength, (int, float)):
            return "Strength must be a number"

        try:
            if not (0.0 <= float(image_strength) <= 1.0):
                return "image_strength must be between 0.0 and 1.0"
        except Exception:
            return "Invalid image_strength value"

        # Basic validation for resize params
        try:
            if int(target_width) < 1 or int(target_height) < 1:
                return "target_width/target_height must be positive"
            if int(round_multiple) < 1:
                return "round_multiple must be >= 1"
        except Exception:
            return "Invalid resize parameters"

        return True

class MfluxLorasPipeline:
    def __init__(self, lora_paths, lora_scales):
        self.lora_paths = lora_paths
        self.lora_scales = lora_scales

    def clear_cache(self):
        self.lora_paths = []
        self.lora_scales = []

class MfluxLorasLoader:
    @classmethod
    def INPUT_TYPES(cls):
        lora_base_path = folder_paths.models_dir
        loras_relative = ["None"] + folder_paths.get_filename_list("loras")

        inputs = {
            "required": {
                "Lora1": (loras_relative, {"tooltip": "Select first LoRA (optional)."}),
                "scale1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Blend strength for Lora1 (0-1)."}),
                "Lora2": (loras_relative, {"tooltip": "Select second LoRA (optional)."}),
                "scale2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Blend strength for Lora2 (0-1)."}),
                "Lora3": (loras_relative, {"tooltip": "Select third LoRA (optional)."}),
                "scale3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Blend strength for Lora3 (0-1)."}),
            },
            "optional": {
                "Loras": ("MfluxLorasPipeline", {"tooltip": "Optionally chain previously selected LoRAs here."})
            }
        }

        return inputs

    RETURN_TYPES = ("MfluxLorasPipeline",)
    RETURN_NAMES = ("Loras",)
    FUNCTION = "lora_stacker"
    CATEGORY = "MFlux/Pro"

    def lora_stacker(self, Loras=None, **kwargs):
        lora_base_path = folder_paths.models_dir
        lora_models = [
            (os.path.join(lora_base_path, "loras", kwargs.get(f"Lora{i}")), kwargs.get(f"scale{i}"))
            for i in range(1, 4) if kwargs.get(f"Lora{i}") != "None"
        ]
        
        if Loras is not None and isinstance(Loras, MfluxLorasPipeline):
            lora_paths = Loras.lora_paths
            lora_scales = Loras.lora_scales

            if lora_paths and lora_scales:
                lora_models.extend(zip(lora_paths, lora_scales))

        if lora_models:
            lora_paths, lora_scales = zip(*lora_models)
        else:
            lora_paths, lora_scales = [], []

        return (MfluxLorasPipeline(list(lora_paths), list(lora_scales)),)

class MfluxControlNetPipeline:
    def __init__(self, model_selection, control_image_path, control_strength, save_canny=False):
        self.model_selection = model_selection
        self.control_image_path = control_image_path
        self.control_strength = control_strength
        self.save_canny = save_canny
 

    def clear_cache(self):
        self.model_selection = None
        self.control_image_path = None
        self.control_strength = None
        self.save_canny = False


class MfluxControlNetLoader:
    @classmethod
    def INPUT_TYPES(cls):
        controlnet_models = [
            "InstantX/FLUX.1-dev-Controlnet-Canny",
        ]

        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

        return {
            "required": {
                "model_selection": (controlnet_models, {"default": "InstantX/FLUX.1-dev-Controlnet-Canny", "tooltip": "Choose a ControlNet model. Canny works best for clear edges."}),
                "image": (sorted(files), {"image_upload": True, "tooltip": "Upload or select the image to extract edges from."}),
                "control_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "How strongly to apply edge guidance (0-1)."}),
                "save_canny": ("BOOLEAN", {"default": False, "label_on": "Save preview", "label_off": "Don't save", "tooltip": "Save the detected edges image to outputs/MFlux for reference."}),
            }

        }

    CATEGORY = "MFlux/Pro"
    RETURN_TYPES = ("MfluxControlNetPipeline", "INT", "INT", "IMAGE",)
    RETURN_NAMES = ("ControlNet", "width", "height", "preprocessed_image")
    FUNCTION = "load_and_select"

    def load_and_select(self, model_selection, image, control_strength, save_canny):
        
        control_image_path = folder_paths.get_annotated_filepath(image)
        # Use oriented version for consistent sizing and preprocessing
        oriented_path, width, height = _make_oriented_copy(control_image_path)

        with Image.open(oriented_path) as img:
            # Create a preview image that PreviewImage can handle reliably
            canny_image = ControlnetUtil.preprocess_canny(img)
            canny_image_np = np.array(canny_image).astype(np.float32)  # in 0..255
            # Ensure 3-channel [H, W, 3]
            if canny_image_np.ndim == 2:
                canny_image_np = np.stack([canny_image_np] * 3, axis=-1)
            elif canny_image_np.ndim == 3 and canny_image_np.shape[-1] == 1:
                canny_image_np = np.repeat(canny_image_np, 3, axis=-1)
            # Normalize to 0..1 float32 and add batch dim to match ComfyUI IMAGE tensor [B,H,W,C]
            if canny_image_np.max() > 1.0:
                canny_image_np = canny_image_np / 255.0
            canny_preview = torch.from_numpy(canny_image_np.astype(np.float32)).unsqueeze(0)

        # Optionally save canny preview to output directory for reference
        if save_canny:
            try:
                output_dir = folder_paths.get_output_directory()
                mflux_output_folder = os.path.join(output_dir, "MFlux")
                os.makedirs(mflux_output_folder, exist_ok=True)
                base = os.path.splitext(os.path.basename(control_image_path))[0]
                canny_file = os.path.join(mflux_output_folder, f"{base}_canny.png")
                canny_image.convert("L").save(canny_file)
            except Exception as e:
                print(f"[MFlux-ComfyUI] Failed to save canny preview: {e}")
        # Return the oriented path so downstream consumers (if any) get the corrected image
        return MfluxControlNetPipeline(model_selection, oriented_path, control_strength, save_canny), width, height, canny_preview

    @classmethod
    def IS_CHANGED(cls, model_selection, image, control_strength, save_canny):
        control_image_path = folder_paths.get_annotated_filepath(image)
        control_strength = round(float(control_strength), 2)
        control_image_hash = hash(image)
        return str(control_image_hash) + str(model_selection) + str(control_strength) + str(save_canny)

    @classmethod
    def VALIDATE_INPUTS(cls, model_selection, image, control_strength, save_canny):

        available_models = [
            "InstantX/FLUX.1-dev-Controlnet-Canny",
        ]
        if model_selection not in available_models:
            return f"Invalid model selection: {model_selection}"

        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid control image file: {}".format(image)

        if not isinstance(control_strength, (int, float)):
            return "Strength must be a number"
        try:
            if not (0.0 <= float(control_strength) <= 1.0):
                return "control_strength must be between 0.0 and 1.0"
        except Exception:
            return "Invalid control_strength value"

        if not isinstance(save_canny, bool):
            return "save_canny must be a boolean value"

        return True
