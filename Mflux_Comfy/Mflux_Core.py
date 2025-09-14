import random
import json
import os
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import comfy.utils as utils
import folder_paths

try:
    import mlx.core as mx  # type: ignore
except Exception:
    mx = None  # noqa: N816

try:
    from mflux.flux.flux import Flux1  # type: ignore
    from mflux.config.config import Config  # type: ignore
except Exception as e:
    raise ImportError("[MFlux-ComfyUI] mflux>=0.10.0 is required. Activate your ComfyUI venv and install with: pip install 'mflux==0.10.0'") from e

# ModelConfig is optional in some mflux builds; handle gracefully
try:
    from mflux.config.model_config import ModelConfig  # type: ignore
except Exception:
    ModelConfig = None  # type: ignore

try:
    from mflux.controlnet.controlnet_util import ControlnetUtil  # type: ignore
    # Some mflux builds lack helpers; fall back to local implementation in that case
    if not hasattr(ControlnetUtil, "preprocess_canny") or not hasattr(ControlnetUtil, "scale_image"):
        raise AttributeError("ControlnetUtil missing expected helpers")
except Exception:
    # Minimal fallback for canny preprocessing and scaling
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
from .Mflux_Pro import MfluxControlNetPipeline

# -------------------------------
# Phase 1 shims & helpers (0.10.0 backend; keep ComfyUI graph compatibility)
# -------------------------------

def _get_mflux_version() -> str:
    try:
        import mflux  # noqa: F401
        return getattr(__import__('mflux'), '__version__', 'unknown')
    except Exception:
        return 'unknown'


_printed_mlx_info = False


def warn_if_mlx_old():
    """Detect and report MLX version, printing once per session.

    Always prints the detected version (or lack thereof) the first time it is called,
    and warns if the version is older than 0.27.0.
    """
    global _printed_mlx_info
    try:
        ver = None
        try:
            import mlx  # type: ignore
            ver = getattr(mlx, "__version__", None)
        except Exception:
            ver = None
        if ver is None:
            try:
                # Some builds expose version via core
                import mlx.core as _mx  # type: ignore
                ver = getattr(_mx, "__version__", None)
            except Exception:
                ver = None
        if ver is None:
            try:
                from importlib.metadata import version as _pkg_version
                ver = _pkg_version("mlx")
            except Exception:
                ver = None

        if not ver:
            if not _printed_mlx_info:
                print("[MFlux-ComfyUI] MLX: not found (mflux requires MLX on Apple Silicon)")
                try:
                    import os as _os
                    if _os.environ.get("MLX_FORCE_CPU"):
                        print(f"[MFlux-ComfyUI] Env MLX_FORCE_CPU={_os.environ.get('MLX_FORCE_CPU')} (forces CPU)")
                except Exception:
                    pass
                _printed_mlx_info = True
            return

        if not _printed_mlx_info:
            print(f"[MFlux-ComfyUI] MLX detected: {ver}")
            _printed_mlx_info = True
        # Coarse warning for older versions
        if str(ver) < '0.27.0':
            print(f"[MFlux-ComfyUI] Warning: MLX {ver} may be incompatible with mflux>=0.10.0. Please upgrade to >=0.27.0.")
        # Also show MLX_FORCE_CPU if set
        try:
            import os as _os
            if _os.environ.get("MLX_FORCE_CPU"):
                print(f"[MFlux-ComfyUI] Env MLX_FORCE_CPU={_os.environ.get('MLX_FORCE_CPU')} (forces CPU)")
        except Exception:
            pass
    except Exception:
        if not _printed_mlx_info:
            print("[MFlux-ComfyUI] Warning: Unable to determine MLX version.")
            _printed_mlx_info = True


def is_third_party_model(model_name: str) -> bool:
    prefixes = [
        "filipstrand/",
        "akx/",
        "Freepik/",
        "shuttleai/",
    ]
    return any(model_name.startswith(p) for p in prefixes)


def migrate_legacy_parameters(**kwargs):
    """Translate legacy init_* params to new image_* params.

    Always set defaults for base_model and low_ram to preserve workflow stability.
    """
    migrated = {}
    mapping = {
        'init_image_path': 'image_path',
        'init_image_strength': 'image_strength',
    }
    for old_key, new_key in mapping.items():
        if old_key in kwargs and kwargs[old_key] is not None:
            migrated[new_key] = kwargs[old_key]
        elif new_key in kwargs and kwargs[new_key] is not None:
            migrated[new_key] = kwargs[new_key]

    migrated.setdefault('base_model', 'dev')
    migrated.setdefault('low_ram', False)
    return migrated

flux_cache = {}

def infer_quant_bits(name: str | None) -> int | None:
    """Infer quantization bits from a model path or name.

    Matches patterns like '8-bit', '6-bit', '5-bit', '4-bit', '3-bit',
    as well as '-mflux-4bit', '4bit', '6bit', etc.
    Returns an int in {3,4,5,6,8} or None if not found.
    """
    if not name:
        return None
    s = str(name).lower()
    # Explicit '-X-bit' patterns
    for b in (8, 6, 5, 4, 3):
        if f"{b}-bit" in s:
            return b
    # Common compact patterns
    for b in (8, 6, 5, 4, 3):
        if f"{b}bit" in s:
            return b
    # mflux-4bit and similar
    if "mflux-4bit" in s:
        return 4
    return None

def load_or_create_flux(model, quantize, path, lora_paths, lora_scales, base_model="dev", respect_ui_quant: bool = False):
    """Create or fetch a cached Flux1 model.

    Always use the constructor with ModelConfig so we can pass lora_paths/lora_scales.
    from_name() in mflux 0.10.0 does NOT accept LoRA args.
    """
    # If a local model path is supplied, infer precision from folder name and
    # prefer it over the UI quantize value to avoid mismatches.
    q_inferred = infer_quant_bits(path)
    if path:
        # Always defer to the saved precision when a local model is provided.
        # Passing a quantize value here can cause double-quantization mismatches.
        q_effective = None
    elif respect_ui_quant:
        q_effective = quantize
    else:
        q_effective = quantize

    key = (model, q_effective, path, tuple(lora_paths), tuple(lora_scales), base_model)
    if key not in flux_cache:
        flux_cache.clear()

        # Determine which base_model hint to pass to ModelConfig
        base_for_config = base_model if (is_third_party_model(model) or ("/" in str(model))) else None

        if ModelConfig is None:
            # Fallback path: can't construct with LoRAs; warn and use from_name()
            if lora_paths:
                print("[MFlux-ComfyUI] Warning: ModelConfig not available; LoRAs will be ignored for this run.")
            flux = Flux1.from_name(model_name=model, quantize=q_effective)
            flux_cache[key] = flux
        else:
            # Preferred path: build ModelConfig then construct Flux1 with LoRAs and optional local_path
            model_config = ModelConfig.from_name(model_name=model, base_model=base_for_config)
            flux = Flux1(
                model_config=model_config,
                quantize=q_effective,
                local_path=path,
                lora_paths=lora_paths,
                lora_scales=lora_scales,
            )
            flux_cache[key] = flux

        if path:
            if q_inferred is not None:
                print(f"[MFlux-ComfyUI] Local model path provided; detected {q_inferred}-bit precision. Using saved precision and ignoring UI quantize.")
            else:
                print("[MFlux-ComfyUI] Local model path provided; using model's saved precision (ignoring UI quantize).")

    # This code is licensed under the Apache 2.0 License.
    # Portions of this code are derived from the work of CharafChnioune at https://github.com/CharafChnioune/MFLUX-WEBUI
    return flux_cache[key]

def get_lora_info(Loras):
    if Loras:
        return Loras.lora_paths, Loras.lora_scales
    return [], []

def generate_image(prompt, model, seed, width, height, steps, guidance, quantize="None", metadata=True, Local_model="", image=None, Loras=None, ControlNet=None, base_model="dev", low_ram=False):
    warn_if_mlx_old()

    model_resolved = "dev" if "dev" in str(Local_model).lower() else ("schnell" if "schnell" in str(Local_model).lower() else model)
    print(f"Using model: {model_resolved}")

    migrated = migrate_legacy_parameters(
        init_image_path=image.image_path if image else None,
        init_image_strength=image.image_strength if image else None,
        base_model=base_model,
        low_ram=low_ram,
    )
    image_path = migrated.get('image_path')
    image_strength = migrated.get('image_strength')
    base_model = migrated.get('base_model', 'dev')

    lora_paths, lora_scales = get_lora_info(Loras)
    if Loras:
        print(f"LoRA paths: {lora_paths}")
        print(f"LoRA scales: {lora_scales}")
        if quantize not in (None, "None") and int(quantize) < 8:
            print("[MFlux-ComfyUI] Warning: LoRAs with quantize < 8 may be unsupported.")

    q_val = None if quantize in (None, "None") else int(quantize)
    if q_val is not None and q_val not in (3, 4, 5, 6, 8):
        raise ValueError("Quantize must be one of 3,4,5,6,8 or None")

    seed_val = random.randint(0, 0xffffffffffffffff) if seed == -1 else int(seed)
    print(f"Using seed: {seed_val}")
    try:
        cn_on = ControlNet is not None
        print(f"[MFlux-ComfyUI] Settings: model={model_resolved}, size={width}x{height}, steps={steps}, guidance={guidance}, low_ram={low_ram}, controlnet={'on' if cn_on else 'off'}")
    except Exception:
        pass

    flux = load_or_create_flux(model_resolved, q_val, Local_model if Local_model else None, lora_paths, lora_scales, base_model=base_model)

    # Prepare config kwargs
    cfg_kwargs = dict(
        num_inference_steps=steps,
        height=height,
        width=width,
        guidance=guidance,
        image_path=image_path,
        image_strength=image_strength,
    )

    # ControlNet conditioning (best-effort with 0.10.0 API). If not supported, proceed without it.
    if ControlNet is not None and isinstance(ControlNet, MfluxControlNetPipeline):
        try:
            control_image_path = ControlNet.control_image_path
            control_strength = float(ControlNet.control_strength)
            # Load and preprocess canny with robust fallbacks
            with Image.open(control_image_path) as _img:
                try:
                    img_scaled = ControlnetUtil.scale_image(height, width, _img)
                except Exception:
                    # Fallback to PIL resize if helper missing
                    img_scaled = _img.resize((width, height), Image.BICUBIC)
                try:
                    canny_img = ControlnetUtil.preprocess_canny(img_scaled)
                except Exception:
                    # Fallback simple edge using FIND_EDGES if helper missing
                    from PIL import ImageFilter, ImageOps
                    canny_img = ImageOps.invert(img_scaled.convert("L").filter(ImageFilter.FIND_EDGES))
                canny_np = np.array(canny_img).astype(np.float32) / 255.0
                if canny_np.ndim == 2:
                    # expand to HWC with 1 channel
                    canny_np = np.expand_dims(canny_np, axis=-1)
            # Attach to config if supported
            try:
                h, w = canny_np.shape[0], canny_np.shape[1]
                c = canny_np.shape[2] if canny_np.ndim == 3 else 1
                print(f"[MFlux-ComfyUI] ControlNet enabled: cond shape {h}x{w}x{c}, strength={control_strength}")
                print("[MFlux-ComfyUI] Note: In current mflux backend, ControlNet may run on a slower path than base MLX model, causing large slowdowns.")
            except Exception:
                pass
            cfg_kwargs.update({
                "controlnet_cond": canny_np,
                "controlnet_strength": control_strength,
            })
        except Exception as e:
            print(f"[MFlux-ComfyUI] ControlNet conditioning not applied due to: {e}")

    # Build config
    try:
        cfg = Config(**cfg_kwargs)
    except TypeError as te:
        # Remove unknown keys and retry (backend may not support controlnet or other extras)
        for k in ("controlnet_cond", "controlnet_strength", "low_ram"):
            cfg_kwargs.pop(k, None)
        cfg = Config(**cfg_kwargs)

    try:
        result = flux.generate_image(
            seed=seed_val if seed_val >= 0 else None,
            prompt=prompt,
            config=cfg,
        )
    except ValueError as e:
        msg = str(e).lower()
        # Fallback for MLX dequantize dtype mismatches with local models
        if "dequantize" in msg and Local_model:
            print("[MFlux-ComfyUI] Detected dequantize error with local model. Bypassing local pack and retrying with built-in model weights...")
            # Try using built-in alias weights (ignoring local path). Preserve LoRAs when possible.
            tried = []
            def _try_buildin(q_choice):
                tried.append(q_choice)
                # Reduce chances of 'Too many open files' from huggingface_hub token reads
                try:
                    import os as _os
                    _os.environ.setdefault("HF_HUB_DISABLE_HF_TRANSFER", "1")
                    _os.environ.setdefault("HF_TOKEN", "")
                    _os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", "")
                    _os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
                except Exception:
                    pass
                if ModelConfig is not None:
                    mc = ModelConfig.from_name(model_name=model_resolved, base_model=None)  # type: ignore[attr-defined]
                    return Flux1(model_config=mc, quantize=q_choice, lora_paths=lora_paths, lora_scales=lora_scales)
                else:
                    return Flux1.from_name(model_name=model_resolved, quantize=q_choice)

            try:
                flux_fallback = _try_buildin(None)
                result = flux_fallback.generate_image(seed=seed_val if seed_val >= 0 else None, prompt=prompt, config=cfg)
            except Exception as e2:
                print(f"[MFlux-ComfyUI] Built-in fallback (quantize=None) failed: {e2}")
                try:
                    flux_fallback = _try_buildin(8)
                    result = flux_fallback.generate_image(seed=seed_val if seed_val >= 0 else None, prompt=prompt, config=cfg)
                except Exception as e3:
                    print(f"[MFlux-ComfyUI] Built-in fallback (quantize=8) failed: {e3}")
                    # As a last resort, retry recreating from local with runtime precision
                    print("[MFlux-ComfyUI] Retrying with local model and no quant override...")
                    flux_fallback = load_or_create_flux(model_resolved, None, Local_model, lora_paths, lora_scales, base_model=base_model, respect_ui_quant=False)
                    result = flux_fallback.generate_image(seed=seed_val if seed_val >= 0 else None, prompt=prompt, config=cfg)
        else:
            raise

    def _to_numpy(img_like):
        # PIL Image
        if isinstance(img_like, Image.Image):
            arr = np.array(img_like)
            return arr.astype(np.float32)
        # numpy array
        if isinstance(img_like, np.ndarray):
            return img_like.astype(np.float32)
        # torch tensor
        if isinstance(img_like, torch.Tensor):
            try:
                return img_like.detach().cpu().numpy().astype(np.float32)
            except Exception:
                pass
        # mflux GeneratedImage or similar wrapper classes: try common attributes/methods
        if img_like is not None:
            # Attribute-based extraction
            for attr in ("image", "pil_image", "np_image", "array", "img"):
                try:
                    if hasattr(img_like, attr):
                        val = getattr(img_like, attr)
                        # If list/tuple, pick first
                        if isinstance(val, (list, tuple)) and len(val) > 0:
                            val = val[0]
                        conv = _to_numpy(val)
                        if conv is not None:
                            return conv
                except Exception:
                    pass
            # Method-based extraction
            for meth in ("to_numpy", "numpy", "as_numpy", "to_pil", "pil"):
                fn = getattr(img_like, meth, None)
                if callable(fn):
                    try:
                        val = fn()
                        # Some methods return (image, extra); take first if tuple/list
                        if isinstance(val, (list, tuple)) and len(val) > 0:
                            val = val[0]
                        conv = _to_numpy(val)
                        if conv is not None:
                            return conv
                    except Exception:
                        pass
            # As a last resort, pick the first image-like field from __dict__
            try:
                if hasattr(img_like, "__dict__"):
                    for key, val in img_like.__dict__.items():
                        conv = _to_numpy(val)
                        if conv is not None:
                            return conv
            except Exception:
                pass
        # MLX array
        if mx is not None and img_like is not None:
            mod = getattr(img_like.__class__, "__module__", "")
            if mod.startswith("mlx") or mod.startswith("mlx_core") or "mlx" in mod:
                # Try common conversion patterns
                for attr in ("to_numpy", "numpy", "asnumpy"):
                    fn = getattr(img_like, attr, None)
                    if callable(fn):
                        try:
                            return fn().astype(np.float32)
                        except Exception:
                            pass
                try:
                    return np.array(img_like).astype(np.float32)
                except Exception:
                    pass
        # File path string
        if isinstance(img_like, str):
            try:
                if os.path.exists(img_like) or img_like.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    with Image.open(img_like) as im:
                        return np.array(im).astype(np.float32)
            except Exception:
                pass
        # Bytes (PNG/JPEG)
        if isinstance(img_like, (bytes, bytearray)):
            try:
                import io
                with Image.open(io.BytesIO(img_like)) as im:
                    return np.array(im).astype(np.float32)
            except Exception:
                pass
        return None

    # Normalize various return shapes from mflux
    numpy_image = None
    if isinstance(result, dict):
        # Common keys used by various backends
        candidate = None
        for key in ("image", "images", "np_image", "pil_image", "array", "img"):
            if key in result:
                candidate = result[key]
                break
        if candidate is None:
            # Fallback: try first value
            try:
                candidate = next(iter(result.values()))
            except Exception:
                candidate = None
        # If candidate is a list/tuple, pick first
        if isinstance(candidate, (list, tuple)) and len(candidate) > 0:
            candidate = candidate[0]
        numpy_image = _to_numpy(candidate)
    elif isinstance(result, (list, tuple)):
        # Take the first element sensibly
        first = None
        for item in result:
            first = item
            if isinstance(item, (Image.Image, np.ndarray)):
                break
        numpy_image = _to_numpy(first)
    else:
        numpy_image = _to_numpy(result)

    if numpy_image is None:
        print(f"[MFlux-ComfyUI] Debug: flux.generate_image returned type: {type(result)}")
        if isinstance(result, dict):
            print(f"[MFlux-ComfyUI] Debug: dict keys = {list(result.keys())}")
        raise RuntimeError("Unexpected result type from flux.generate_image")

    if numpy_image is None:
        raise RuntimeError("Failed to convert generated image to numpy array")

    if numpy_image.ndim == 3 and numpy_image.shape[0] in (1, 3, 4) and numpy_image.shape[-1] not in (3, 4):
        numpy_image = np.moveaxis(numpy_image, 0, -1)

    # Scale to [0,1] if needed
    if numpy_image.dtype != np.float32:
        numpy_image = numpy_image.astype(np.float32)
    max_val = float(numpy_image.max()) if numpy_image.size else 1.0
    if max_val > 1.0:
        numpy_image = numpy_image / 255.0

    tensor_image = torch.from_numpy(numpy_image)
    if tensor_image.dim() == 3:
        tensor_image = tensor_image.unsqueeze(0)

    return (tensor_image,)

def save_images_with_metadata(images, prompt, model, quantize, Local_model, seed, height, width, steps, guidance, lora_paths, lora_scales, image_path, image_strength, filename_prefix="Mflux", full_prompt=None, extra_pnginfo=None, base_model=None, low_ram=False, control_image_path=None, control_strength=None, control_model=None, quantize_effective=None):
    
    output_dir = folder_paths.get_output_directory()
    full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
        filename_prefix, output_dir, images[0].shape[1], images[0].shape[0])
    mflux_output_folder = os.path.join(full_output_folder, "MFlux")
    os.makedirs(mflux_output_folder, exist_ok=True)
    existing_files = os.listdir(mflux_output_folder)
    existing_counters = [
        int(f.split("_")[-1].split(".")[0])
        for f in existing_files
        if f.startswith(filename_prefix) and f.endswith(".png")
    ]
    counter = max(existing_counters, default=0) + 1

    results = list()
    for image in images:
        i = 255. * image.cpu().numpy().squeeze()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        metadata = None
        if full_prompt is not None or extra_pnginfo is not None:
            metadata = PngInfo()
            if full_prompt is not None:
                metadata.add_text("full_prompt", json.dumps(full_prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata.add_text(x, json.dumps(extra_pnginfo[x]))
        image_file = f"{filename_prefix}_{counter:05}.png"
        img.save(os.path.join(mflux_output_folder, image_file), pnginfo=metadata, compress_level=4)
        results.append({
            "filename": image_file,
            "subfolder": subfolder,
            "type": "output"
        })

        metadata_jsonfile = os.path.join(mflux_output_folder, f"{filename_prefix}_{counter:05}.json")
        json_dict = {
            "prompt": prompt,
            "model": model,
            "quantize": quantize,
            "quantize_effective": quantize_effective if quantize_effective is not None else ("local_model_precision" if Local_model else quantize),
            "seed": seed,
            "height": height,
            "width": width,
            "steps": steps,
            "guidance": guidance if model == "dev" else None,
            "Local_model": Local_model,
            # Store both legacy and new field names for compatibility
            "init_image_path": image_path,
            "init_image_strength": image_strength,
            "image_path": image_path,
            "image_strength": image_strength,
            "lora_paths": lora_paths,
            "lora_scales": lora_scales,
            "base_model": base_model,
            "low_ram": low_ram,
            "mflux_version": _get_mflux_version(),
            "control_image_path": control_image_path,
            "control_strength": control_strength,
            "control_model": control_model,
        }
        with open(metadata_jsonfile, 'w') as metadata_file:
            json.dump(json_dict, metadata_file, indent=4)
        counter += 1
    return {"ui": {"images": results}, "counter": counter}
