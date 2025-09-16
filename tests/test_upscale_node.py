import os

from Mflux_Comfy.Mflux_Pro import MfluxUpscale


def test_upscale_input_validation():
    # Prepare a fake input directory file name; the node's INPUT_TYPES reads available files from folder_paths
    # We only test validation signature here; ComfyUI wiring is not exercised in unit tests.
    iv = MfluxUpscale.VALIDATE_INPUTS  if hasattr(MfluxUpscale, 'VALIDATE_INPUTS') else None
    # Basic smoke: ensure the class exposes the generate function
    assert hasattr(MfluxUpscale, 'generate_upscale')
