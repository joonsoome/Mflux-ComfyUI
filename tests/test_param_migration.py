import pytest

from Mflux_Comfy.Mflux_Core import migrate_legacy_parameters


def test_migrate_legacy_parameters_prefers_new_fields():
    res = migrate_legacy_parameters(init_image_path=None, init_image_strength=None, image_path="/a/b.png", image_strength=0.25)
    assert res["image_path"] == "/a/b.png"
    assert res["image_strength"] == 0.25
    assert res["base_model"] == "dev"
    assert res["low_ram"] is False


def test_migrate_legacy_parameters_maps_legacy():
    res = migrate_legacy_parameters(init_image_path="/legacy.png", init_image_strength=0.4)
    assert res["image_path"] == "/legacy.png"
    assert res["image_strength"] == 0.4
    assert res["base_model"] == "dev"
    assert res["low_ram"] is False
