import random
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

    def load_model_pickle(self,file_path):
        with open(file_path,"rb") as f:
            self.model = pickle.load(f)
            
    def save_model_pickle(self,file_path):
        with open(file_path,"wb") as f:
            pickle.dump(self.model,f)

    def is_objuscated(self) -> bool:
        return self.model_weights_shuffle_idxs and self.model_block_shuffle_map
    
    def change_weights(self,seed):
        model_weights_shuffle_idxs = []

        for i, tensor in enumerate(self.model.parameters()):
            shuffled, indices = shuffle_all_axes(tensor.data.cpu().detach().numpy())

            model_weights_shuffle_idxs.append(indices)

            shuffled_tensor = torch.tensor(shuffled)

            with torch.no_grad():
                tensor.data = shuffled_tensor

        self.model_weights_shuffle_idxs = model_weights_shuffle_idxs
        return model_weights_shuffle_idxs

    def retrive_weights(self):
        for i, tensor in enumerate(self.model.parameters()):
            orig = recover_original(tensor.data.cpu().detach().numpy(), self.model_weights_shuffle_idxs[i])
            orig_tensor = torch.tensor(orig)

            with torch.no_grad():
                tensor.data = orig_tensor

        self.model_weights_shuffle_idxs = None

    def obfuscate_model(self, seed:int = -1, override:bool = False):
        if not override and self.is_objuscated():
            logger.warning("Model has already been obfuscated. To re-obfuscate, set override=True.")
            return

        model_weights_shuffle_idxs = self.change_weights(seed=seed)
        model_block_shuffle_map = model_obfuscation(self.model)

        self.model_weights_shuffle_idxs = model_weights_shuffle_idxs
        self.model_block_shuffle_map = model_block_shuffle_map

        return
    
    def deobfuscate_model(self):
        if not self.is_objuscated():
            logger.warning("Model has not been obfuscated.")
            return

        model_deobfuscation(self.model, self.model_block_shuffle_map)
        self.retrive_weights()

        self.model_weights_shuffle_idxs = None
        self.model_block_shuffle_map = None

    def save_mtd(self, file_path, map_path,model_map_path):
        with open(file_path,"wb") as f:
            pickle.dump(self.model,f)
        with open(map_path,"wb") as f:
            pickle.dump(self.model_weights_shuffle_idxs,f)
        with open(model_map_path,"wb") as f:
            pickle.dump(self.model_block_shuffle_map,f)

    @classmethod
    def load_mtd(cls, file_path,map_path,model_map_path):
        with open(model_map_path,"rb") as f:
            model_block_shuffle_map = pickle.load(f)
        with open(file_path,"rb") as f:
            model = pickle.load(f)
        with open(map_path,"rb") as f:
            model_weights_shuffle_idxs = pickle.load(f)

        mtd_model = cls(model=model, model_weights_shuffle_idxs=model_weights_shuffle_idxs, model_block_shuffle_map=model_block_shuffle_map)
        return mtd_model

    def validate_model(self, other_model):
        for p1, p2 in zip(self.model.parameters(), other_model.model.parameters()):
            if not torch.allclose(p1.data, p2.data, atol=1e-4):
                return False
        return True