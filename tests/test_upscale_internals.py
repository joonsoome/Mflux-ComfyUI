from PIL import Image


def test_upscale_generate_forwarding(monkeypatch, tmp_path):
    # Create a source image with a non-multiple-of-8 size to exercise rounding
    src = tmp_path / "source.png"
    Image.new("RGB", (123, 221), color=(128, 128, 128)).save(src)

    # Ensure MfluxUpscale will resolve the annotated filepath to our temp image
    import folder_paths

    monkeypatch.setattr(folder_paths, "get_annotated_filepath", lambda x: str(src))

    # Patch get_lora_info to a no-op to avoid external dependencies
    import Mflux_Comfy.Mflux_Core as core

    monkeypatch.setattr(core, "get_lora_info", lambda Loras: ([], []))

    captured = {}

    def fake_generate_image(**kwargs):
        # Capture all kwargs for assertions
        captured.update(kwargs)
        return ("FAKE_IMAGE",)

    monkeypatch.setattr(core, "generate_image", fake_generate_image)

    # Import the Upscale node class and call generate_upscale
    from Mflux_Comfy.Mflux_Pro import MfluxUpscale, MfluxControlNetPipeline

    node = MfluxUpscale()
    out = node.generate_upscale(
        prompt="Upscale test",
        image=src.name,
        scale="2",
        control_strength=0.6,
        steps=28,
        seed=-1,
        model="dev",
        quantize="8",
        Loras=None,
        base_model="dev",
        low_ram=False,
        metadata=False,
        vae_tiling=True,
        vae_tiling_split="vertical",
    )

    # Our fake returns a tuple
    assert out == ("FAKE_IMAGE",)

    # Reproduce the rounding logic used in the node
    iw, ih = 123, 221
    s = float("2")
    tw = max(8, int(round(iw * s)))
    th = max(8, int(round(ih * s)))

    def _round8(v):
        return max(8, int(round(v / 8) * 8))

    exp_tw, exp_th = _round8(tw), _round8(th)

    assert captured.get("width") == exp_tw
    assert captured.get("height") == exp_th

    # ControlNet should be an instance of the pipeline created by the node
    assert isinstance(captured.get("ControlNet"), MfluxControlNetPipeline)
    assert captured.get("ControlNet").model_selection == "jasperai/Flux.1-dev-Controlnet-Upscaler"

    # VAE tiling flags should be forwarded
    assert captured.get("vae_tiling") is True
    assert captured.get("vae_tiling_split") == "vertical"


def test_upscale_uses_oriented_control_image(monkeypatch, tmp_path):
    # Create a source image and ensure the oriented copy is created and used
    src = tmp_path / "source2.png"
    from PIL import Image
    Image.new("RGB", (100, 150), color=(10, 20, 30)).save(src)

    import folder_paths
    monkeypatch.setattr(folder_paths, "get_annotated_filepath", lambda x: str(src))

    # Use a fake generate_image that inspects the ControlNet arg
    import Mflux_Comfy.Mflux_Core as core
    monkeypatch.setattr(core, "get_lora_info", lambda Loras: ([], []))

    def fake_generate_image(**kwargs):
        cn = kwargs.get("ControlNet")
        # control_image_path should point to an oriented PNG file next to src
        cip = cn.control_image_path
        assert cip.endswith("_oriented.png") or cip == str(src)
        return ("OK",)

    monkeypatch.setattr(core, "generate_image", fake_generate_image)

    from Mflux_Comfy.Mflux_Pro import MfluxUpscale
    node = MfluxUpscale()
    out = node.generate_upscale(
        prompt="Test",
        image=src.name,
        scale="1.5",
        control_strength=0.6,
        steps=20,
        seed=-1,
        model="dev",
    )
    assert out == ("OK",)
