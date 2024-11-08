from typing import List, Optional, Tuple, Union
import io
import copy

from model_mtd.model_mtd import MTDModel

import pytest

def _test_single_torch_model(model: "torch.nn.Module", model_name: Optional[str] = None):
    torch = pytest.importorskip("torch")

    if model_name is None:
        model_name = model.__class__.__name__

    def _test_mtd(mtd_before: MTDModel):
        hash_before = mtd_before.model_hash()

        mtd_after = copy.deepcopy(mtd_before)
        hash_after_initial = mtd_after.model_hash()

        assert mtd_before == mtd_after, f"Model {model_name} not copied correctly."
        assert hash_before == hash_after_initial, f"Model {model_name} hash does not match after copying."

        mtd_after.obfuscate_model(seed=1)
        hash_after_obfuscated = mtd_after.model_hash()

        assert mtd_before != mtd_after, f"Model {model_name} not obfuscated correctly."
        assert hash_before != hash_after_obfuscated, f"Model {model_name} hash matches after obfuscation."

        mtd_after.deobfuscate_model()
        hash_after_deobfuscated = mtd_after.model_hash()

        assert mtd_before == mtd_after, f"Model {model_name} not deobfuscated correctly."
        assert hash_before == hash_after_deobfuscated, f"Model {model_name} hash does not match after deobfuscation."

    def _test_save_load(mtd: MTDModel):
        model_vfile = io.BytesIO()
        model_weights_shuffle_idxs_vfile = io.BytesIO()
        model_block_shuffle_map_vfile = io.BytesIO()

        hash_before = mtd.model_hash()

        mtd.save_mtd(
            model_file_or_path=model_vfile,
            model_weights_shuffle_idxs_file_or_path=model_weights_shuffle_idxs_vfile,
            model_block_shuffle_map_file_or_path=model_block_shuffle_map_vfile,
            close_files=False,
        )

        model_vfile.seek(0)
        model_weights_shuffle_idxs_vfile.seek(0)
        model_block_shuffle_map_vfile.seek(0)

        mtd_loaded = MTDModel.load_mtd(
            model_file_or_path=model_vfile,
            model_weights_shuffle_idxs_file_or_path=model_weights_shuffle_idxs_vfile,
            model_block_shuffle_map_file_or_path=model_block_shuffle_map_vfile,
            close_files=False,
        )

        hash_after = mtd_loaded.model_hash()

        assert mtd == mtd_loaded, f"Model {model_name} not loaded correctly."
        assert hash_before == hash_after, f"Model {model_name} hash does not match after loading."


    model_mtd = MTDModel(model=model)

    assert model_mtd == model_mtd, f"Model {model_name} validation failed."

    _test_mtd(model_mtd)
    _test_save_load(model_mtd)

def test_torch_mtd_simple():
    torch = pytest.importorskip("torch")
    from torch_classes import SimpleModel

    model = SimpleModel()
    _test_single_torch_model(model, model_name="Custom SimpleModel")

def _test_torch_mtd_vision(module, torchvision_module, model_name_prefix:str=""):
    for model_name in torchvision_module.models.list_models(module):
        model = torchvision_module.models.get_model(model_name, weights="DEFAULT")

        _test_single_torch_model(model, model_name=f'{model_name_prefix}{model_name}')

@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_torch_mtd_vision_classification():
    torchvision = pytest.importorskip("torchvision")
    _test_torch_mtd_vision(torchvision.models, torchvision, model_name_prefix="classification: ")

@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_torch_mtd_vision_quantization():
    torchvision = pytest.importorskip("torchvision")
    _test_torch_mtd_vision(torchvision.models.quantization, torchvision, model_name_prefix="classification (qunatized): ")

@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_torch_mtd_vision_segmentation():
    torchvision = pytest.importorskip("torchvision")
    _test_torch_mtd_vision(torchvision.models.segmentation, torchvision, model_name_prefix="segmentation: ")

@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_torch_mtd_vision_detection():
    torchvision = pytest.importorskip("torchvision")
    _test_torch_mtd_vision(torchvision.models.detection, torchvision, model_name_prefix="detection: ")

@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_torch_mtd_vision_video():
    torchvision = pytest.importorskip("torchvision")
    _test_torch_mtd_vision(torchvision.models.video, torchvision, model_name_prefix="video: ")