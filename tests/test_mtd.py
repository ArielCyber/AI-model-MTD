from typing import List, Tuple, Union
import io
import copy

from model_mtd.model_mtd import MTDModel

import pytest

def _test_single_torch_model(model: "torch.nn.Module"):
    torch = pytest.importorskip("torch")

    def _test_mtd(mtd_before: MTDModel):
        mtd_after = copy.deepcopy(mtd_before)
        assert mtd_before == mtd_after, "Model not copied correctly."

        mtd_after.obfuscate_model(seed=1)
        assert mtd_before != mtd_after, "Model not obfuscated correctly."

        mtd_after.deobfuscate_model()
        assert mtd_before == mtd_after, "Model not deobfuscated correctly."

    def _test_save_load(mtd: MTDModel):
        model_vfile = io.BytesIO()
        model_weights_shuffle_idxs_vfile = io.BytesIO()
        model_block_shuffle_map_vfile = io.BytesIO()

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

        assert mtd == mtd_loaded, "Model not loaded correctly."


    model_mtd = MTDModel(model=model)

    assert model_mtd == model_mtd, "Model validation failed."

    _test_mtd(model_mtd)
    _test_save_load(model_mtd)

def test_torch_mtd():
    torch = pytest.importorskip("torch")
    torchvision = pytest.importorskip("torchvision")
    from torch_classes import SimpleModel

    model = torchvision.models.resnet18()
    _test_single_torch_model(model)

    model = SimpleModel()
    _test_single_torch_model(model)
    
    