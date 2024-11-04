import random
from typing import IO, Optional, Union
import torch
import torchvision.models as models
from model_mtd.arr_shuffle import *
import pickle

import logging
logger = logging.getLogger(__name__)

def switch_blocks(model, target_layer, block_idx1, block_idx2):
    # Ensure the target_layer is valid
    if not hasattr(model, target_layer):
        #raise ValueError(f"Invalid target layer: {target_layer}")
        return
    # Get the target layer from the model
    target_layer_module = getattr(model, target_layer)

    # Ensure block_idx1 and block_idx2 are valid indices
    try:
        num_blocks = len(target_layer_module)
    except:
        return
    if not (0 <= block_idx1 < num_blocks) or not (0 <= block_idx2 < num_blocks):
        #raise ValueError("Invalid block indices")
        return

    # Swap the blocks by modifying the model's attributes
    target_layer_module[block_idx1], target_layer_module[block_idx2] = target_layer_module[block_idx2], target_layer_module[block_idx1]

def model_obfuscation(model):
    layer_names = [name for name, _ in model.named_children()]
    switch_dict = {}
    for i in layer_names:
        try:
            target_layer_module = getattr(model, i)
            num_blocks = len(target_layer_module)
            choose_from = list(range(num_blocks))
            array_switch = []
            while len(choose_from) > 0:
                rand_i = choose_from.pop(random.randrange(len(choose_from)))
                rand_j = -1
                if(len(choose_from)> 0):
                    rand_j = choose_from.pop(random.randrange(len(choose_from)))
                    array_switch.append((rand_j,rand_i))
                else:
                    rand_j = rand_i
                    array_switch.append((rand_j,rand_i))
                # Switch the positions of two blocks in 'layer1' (e.g., blocks 0 and 3)
                switch_blocks(model, target_layer=i, block_idx1=rand_j, block_idx2=rand_i)
            switch_dict[i] = array_switch
        except:
            pass

    return switch_dict

def model_deobfuscation(model, obf_dict):
    logger.debug(f'Deobfuscating model with obf_dict: {obf_dict}')
    for key, data in obf_dict.items():
        for switch in data:
            # Switch the positions of two blocks in 'layer1' (e.g., blocks 0 and 3)
            switch_blocks(model, target_layer=key, block_idx1=switch[0], block_idx2=switch[1])

def _pickle_load(file_obj_or_path: Union[str, IO], close_file=True):
    if isinstance(file_obj_or_path, str):
        with open(file_obj_or_path, "rb") as f:
            obj = pickle.load(f)
    else:
        obj = pickle.load(file_obj_or_path)
        if close_file:
            file_obj_or_path.close()
    return obj

def _pickle_save(obj, file_obj_or_path: Union[str, IO], close_file=True):
    if isinstance(file_obj_or_path, str):
        with open(file_obj_or_path, "wb") as f:
            pickle.dump(obj, f)
    else:
        pickle.dump(obj, file_obj_or_path)
        if close_file:
            file_obj_or_path.close()

class MTDModel:
    def __init__(self, model=None, model_weights_shuffle_idxs=None, model_block_shuffle_map=None):
        if model is None:
            logger.warning("Model is None. Please provide a model.")
            return
        
        if not isinstance(model, torch.nn.Module):
            logger.warning("Model is not an instance of torch.nn.Module. MTDModel currently only supports PyTorch models.")
            raise NotImplementedError("Model is not an instance of torch.nn.Module. MTDModel currently only supports PyTorch models.")

        self.model = model
        self.model_weights_shuffle_idxs = model_weights_shuffle_idxs
        self.model_block_shuffle_map = model_block_shuffle_map

    @staticmethod
    def _load_model_pickle(file_obj_or_path: Union[str, IO], close_file=True):
        model = _pickle_load(file_obj_or_path, close_file=close_file)
        return model
            
    @staticmethod
    def _save_model_pickle(model, file_obj_or_path: Union[str, IO], close_file=True):
        _pickle_save(model, file_obj_or_path, close_file=close_file)

    @staticmethod
    def _load_model_weights_shuffle_idxs_pickle(file_obj_or_path: Union[str, IO], close_file=True):
        model_weights_shuffle_idxs = _pickle_load(file_obj_or_path, close_file=close_file)
        return model_weights_shuffle_idxs
    
    @staticmethod
    def _save_model_weights_shuffle_idxs_pickle(model_weights_shuffle_idxs, file_obj_or_path: Union[str, IO], close_file=True):
        _pickle_save(model_weights_shuffle_idxs, file_obj_or_path, close_file=close_file)

    @staticmethod
    def _load_model_block_shuffle_map_pickle(file_obj_or_path: Union[str, IO], close_file=True):
        model_block_shuffle_map = _pickle_load(file_obj_or_path, close_file=close_file)
        return model_block_shuffle_map
    
    @staticmethod
    def _save_model_block_shuffle_map_pickle(model_block_shuffle_map, file_obj_or_path: Union[str, IO], close_file=True):
        _pickle_save(model_block_shuffle_map, file_obj_or_path, close_file=close_file)

    
    def is_objuscated(self) -> bool:
        return self.model_weights_shuffle_idxs and self.model_block_shuffle_map
    
    def _change_weights(self,seed:Optional[int] = None):
        model_weights_shuffle_idxs = []

        for i, tensor in enumerate(self.model.parameters()):
            shuffled, indices = shuffle(tensor.data.cpu().detach().numpy(), seed=seed)

            model_weights_shuffle_idxs.append(indices)

            shuffled_tensor = torch.tensor(shuffled)

            with torch.no_grad():
                tensor.data = shuffled_tensor

        self.model_weights_shuffle_idxs = model_weights_shuffle_idxs
        return model_weights_shuffle_idxs

    def _retrive_weights(self):
        if not self.model_weights_shuffle_idxs:
            logger.warning("Model weights have not been shuffled.")
            return

        for i, tensor in enumerate(self.model.parameters()):
            orig = recover(tensor.data.cpu().detach().numpy(), self.model_weights_shuffle_idxs[i])
            orig_tensor = torch.tensor(orig)

            with torch.no_grad():
                tensor.data = orig_tensor

        self.model_weights_shuffle_idxs = None

    def obfuscate_model(self, seed:Optional[int] = None, override:bool = False):
        if not override and self.is_objuscated():
            logger.warning("Model has already been obfuscated. To re-obfuscate, set override=True.")
            return

        model_weights_shuffle_idxs = self._change_weights(seed=seed)
        model_block_shuffle_map = model_obfuscation(self.model)

        self.model_weights_shuffle_idxs = model_weights_shuffle_idxs
        self.model_block_shuffle_map = model_block_shuffle_map

        return
    
    def deobfuscate_model(self):
        if not self.is_objuscated():
            logger.warning("Model has not been obfuscated.")
            return

        model_deobfuscation(self.model, self.model_block_shuffle_map)
        self._retrive_weights()

        self.model_weights_shuffle_idxs = None
        self.model_block_shuffle_map = None

    def save_mtd(self, 
                model_file_or_path: Union[str, IO],
                model_weights_shuffle_idxs_file_or_path: Union[str, IO],
                model_block_shuffle_map_file_or_path: Union[str, IO],
                close_files=True):
        
        self._save_model_pickle(self.model, model_file_or_path, close_file=close_files)
        self._save_model_weights_shuffle_idxs_pickle(self.model_weights_shuffle_idxs, model_weights_shuffle_idxs_file_or_path, close_file=close_files)
        self._save_model_block_shuffle_map_pickle(self.model_block_shuffle_map, model_block_shuffle_map_file_or_path, close_file=close_files)

    @classmethod
    def load_mtd(cls,
                model_file_or_path: Union[str, IO],
                model_weights_shuffle_idxs_file_or_path: Union[str, IO],
                model_block_shuffle_map_file_or_path: Union[str, IO],
                close_files=True):
        model = cls._load_model_pickle(model_file_or_path, close_file=close_files)
        model_weights_shuffle_idxs = cls._load_model_weights_shuffle_idxs_pickle(model_weights_shuffle_idxs_file_or_path, close_file=close_files)
        model_block_shuffle_map = cls._load_model_block_shuffle_map_pickle(model_block_shuffle_map_file_or_path, close_file=close_files)

        mtd_model = cls(model=model, model_weights_shuffle_idxs=model_weights_shuffle_idxs, model_block_shuffle_map=model_block_shuffle_map)
        return mtd_model

    def validate_model(self, other_model):
        for p1, p2 in zip(self.model.parameters(), other_model.model.parameters()):
            if not torch.allclose(p1.data, p2.data, atol=1e-4):
                return False
        return True