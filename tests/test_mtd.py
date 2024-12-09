import gc
import random
from typing import List, Optional, Tuple, Union
import io
import copy

from model_mtd.model_mtd import MTDModel
from model_mtd.model_utils import extract_weights_torch, load_weights_from_flattened_vector_torch

import pytest

def _change_one_random_weight(weights, inplace: bool = False):
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng()

    if not inplace:
        weights = weights.copy()

    idx = random.randint(0, len(weights) - 1)
    weights[idx] += rng.random(dtype=weights.dtype)

    return weights

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

        del mtd_after
        gc.collect()

    def _test_save_load(mtd: MTDModel):
        model_vfile = io.BytesIO()
        mtd_inner_state_file_or_path = io.BytesIO()

        """
            Check valid save/load
        """
        hash_before = mtd.model_hash()

        mtd.save_mtd(
            model_file_or_path=model_vfile,
            mtd_inner_state_file_or_path=mtd_inner_state_file_or_path,
            close_files=False,
        )

        model_vfile.seek(0)
        mtd_inner_state_file_or_path.seek(0)

        mtd_loaded = MTDModel.load_mtd(
            model_file_or_path=model_vfile,
            mtd_inner_state_file_or_path=mtd_inner_state_file_or_path,
            close_files=False,

            validate_hash=True,
        )

        assert mtd_loaded is not None, f"Model {model_name} not loaded correctly."

        hash_after = mtd_loaded.model_hash()

        assert mtd == mtd_loaded, f"Model {model_name} not loaded correctly."
        assert hash_before == hash_after, f"Model {model_name} hash does not match after loading."

        model_vfile.close()
        # mtd_inner_state_file_or_path.close()

        """
            Check load validation
        """
        del mtd_loaded
        gc.collect()

        w = extract_weights_torch(mtd.model)
        w_changed = _change_one_random_weight(w, inplace=True)
        load_weights_from_flattened_vector_torch(mtd.model, w_changed, inplace=True)

        del w, w_changed
        gc.collect()

        model_vfile = io.BytesIO()

        model_vfile.seek(0)
        mtd_inner_state_file_or_path.seek(0)

        MTDModel._save_model_pickle(mtd.model, model_vfile, close_file=False)
        model_vfile.seek(0)

        mtd_loaded_new = MTDModel.load_mtd(
            model_file_or_path=model_vfile,
            mtd_inner_state_file_or_path=mtd_inner_state_file_or_path,
            close_files=False,

            validate_hash=True,
        )
        assert mtd_loaded_new is None, f"Attacked Model {model_name} should not be loaded."


        model_vfile.close()
        mtd_inner_state_file_or_path.close()


    model_mtd = MTDModel(model=model)

    assert model_mtd == model_mtd, f"Model {model_name} validation failed."

    _test_mtd(model_mtd)
    _test_save_load(model_mtd)

    del model_mtd
    gc.collect()

def test_torch_mtd_simple():
    torch = pytest.importorskip("torch")
    from torch_classes import SimpleModel

    model = SimpleModel()
    _test_single_torch_model(model, model_name="Custom SimpleModel")

def _test_torch_mtd_vision(module, torchvision_module, model_name_prefix:str="", exclude_models: Optional[List[str]] = None):
    for model_name in torchvision_module.models.list_models(module, exclude=exclude_models):
        model = torchvision_module.models.get_model(model_name, weights="DEFAULT")

        _test_single_torch_model(model, model_name=f'{model_name_prefix}{model_name}')

        del model
        gc.collect()

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