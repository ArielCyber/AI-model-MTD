from abc import ABC, abstractmethod
import gc
import random
from typing import IO, List, Literal, Optional, Union
import io
import torch
import torchvision.models as models
from model_mtd.arr_shuffle import *
import pickle
import hashlib


from model_mtd.model_utils import extract_weights_torch, load_weights_from_flattened_vector_torch

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

def _torch_load(file_obj_or_path: Union[str, IO], close_file=True):
    model = torch.load(file_obj_or_path, weights_only=False)
    if isinstance(file_obj_or_path, io.IOBase) and close_file:
        file_obj_or_path.close()
    return model

def _torch_save(model, file_obj_or_path: Union[str, IO], close_file=True):
    torch.save(model, file_obj_or_path)
    if isinstance(file_obj_or_path, io.IOBase) and close_file:
        file_obj_or_path.close()

def torch_model_hash(model):
    w = extract_weights_torch(model)

    hash = hashlib.sha256(w.tobytes()).hexdigest()

    del w
    gc.collect()

    return hash

class MTDObfuscationAction(ABC):
    @abstractmethod
    def obfuscate(self, model, **obf_kwargs) -> "model":
        pass

    @abstractmethod
    def deobfuscate(self, model) -> "model":
        pass

class MTDObfuscationActionShuffleWeightsFlat(MTDObfuscationAction):
    def obfuscate(self, model, **obf_kwargs):
        seed = obf_kwargs.get('seed', None)

        w = extract_weights_torch(model)
        w_shuffled, indices = shuffle(w, seed=seed)

        model = load_weights_from_flattened_vector_torch(model, w_shuffled, inplace=True)
        return model, indices

    def deobfuscate(self, model, indices):
        w = extract_weights_torch(model)
        w_orig = recover(w, indices)

        model = load_weights_from_flattened_vector_torch(model, w_orig, inplace=True)
        return model
    
class MTDObfuscationActionShuffleWeightsPerLayer(MTDObfuscationAction):
    def obfuscate(self, model, **obf_kwargs):
        seed = obf_kwargs.get('seed', None)

        model_weights_shuffle_idxs = []

        for i, tensor in enumerate(model.parameters()):
            shuffled, indices = shuffle(tensor.data.cpu().detach().numpy(),  seed=seed)

            model_weights_shuffle_idxs.append(indices)

            shuffled_tensor = torch.tensor(shuffled)

            with torch.no_grad():
                tensor.data = shuffled_tensor

        return model, model_weights_shuffle_idxs

    def deobfuscate(self, model, model_weights_shuffle_idxs):
        for i, tensor in enumerate(model.parameters()):
            orig = recover(tensor.data.cpu().detach().numpy(), model_weights_shuffle_idxs[i])
            orig_tensor = torch.tensor(orig)

            with torch.no_grad():
                tensor.data = orig_tensor

        return model

class MTDObfuscationActionShuffleWeightsTorch(MTDObfuscationAction):
    def obfuscate(self, model, **obf_kwargs):
        model_weights_shuffle_idxs = []
        seed = obf_kwargs.get('seed', None)

        for i, tensor in enumerate(model.parameters()):
            # pass
            shuffled, indices = shuffle_torch(tensor.data, seed=seed)
            model_weights_shuffle_idxs.append(indices)

            with torch.no_grad():
                tensor.data = shuffled

        return model, model_weights_shuffle_idxs

    def deobfuscate(self, model, indices):
        for i, tensor in enumerate(model.parameters()):
            # pass
            orig = recover_torch(tensor.data, indices[i])
            with torch.no_grad():
                tensor.data = orig

        return model
            

class MTDModelInnerState:
    def __init__(self,
                 obfuscation_actions: List[MTDObfuscationAction] = [],
                 ):
        
        self.obfuscation_actions = obfuscation_actions
        obfuscation_actions_artifacts = [None] * len(obfuscation_actions)
        self.obfuscation_actions_artifacts = obfuscation_actions_artifacts

        self.state:Literal['init', 'obfuscated', 'deobfuscated'] = 'init'

    @classmethod
    def ret_shuffle_weights_mtd(cls, shuffle_mode: Literal['flat', 'per_layer'] = 'flat'):
        shuffle_obfuscation_action = MTDObfuscationActionShuffleWeightsFlat() if shuffle_mode == 'flat' else MTDObfuscationActionShuffleWeightsPerLayer()
        return cls(obfuscation_actions=[shuffle_obfuscation_action])
    
    def is_obfuscated(self) -> bool:
        return self.state == 'obfuscated'

    def reset_state(self):
        self.obfuscation_actions_artifacts = [None] * len(self.obfuscation_actions)

    def obfuscate_model(self, model, seed:Optional[int] = None):
        if self.is_obfuscated():
            logger.warning("Model has already been obfuscated.")
            return model
        
        for i, action in enumerate(self.obfuscation_actions):
            model, artifact = action.obfuscate(model, seed=seed)
            self.obfuscation_actions_artifacts[i] = artifact

        self.state = 'obfuscated'

        return model
    
    def deobfuscate_model(self, model):
        if not self.is_obfuscated():
            logger.warning("Model has not been obfuscated.")
            return model

        for i, action in enumerate(reversed(self.obfuscation_actions)):
            model = action.deobfuscate(model, self.obfuscation_actions_artifacts[i])

        self.reset_state()
        self.state = 'deobfuscated'

        return model
    
    def save(self, file_obj_or_path: Union[str, IO], close_file=True):
        _pickle_save(self, file_obj_or_path, close_file=close_file)

    @classmethod
    def load(cls, file_obj_or_path: Union[str, IO], close_file=True):
        return _pickle_load(file_obj_or_path, close_file=close_file)

class MTDModel:
    def __init__(self, model=None,
                mtd_inner_state: Optional[MTDModelInnerState] = None,
                mtd_mode: Optional[Literal['shuffle_per_layer', 'shuffle_flat', 'shuffle_per_layer_torch']] = 'shuffle_per_layer_torch',
                ):
        if model is None:
            logger.warning("Model is None. Please provide a model.")
            return
        
        if not isinstance(model, torch.nn.Module):
            logger.warning("Model is not an instance of torch.nn.Module. MTDModel currently only supports PyTorch models.")
            raise NotImplementedError("Model is not an instance of torch.nn.Module. MTDModel currently only supports PyTorch models.")

        self.model = model

        if mtd_inner_state is not None:
            self.mtd_inner_state = mtd_inner_state
        else:
            if mtd_mode == 'shuffle_flat':
                self.mtd_inner_state = MTDModelInnerState.ret_shuffle_weights_mtd(shuffle_mode='flat')
            elif mtd_mode == 'shuffle_per_layer':
                self.mtd_inner_state = MTDModelInnerState.ret_shuffle_weights_mtd(shuffle_mode='per_layer')
            elif mtd_mode == 'shuffle_per_layer_torch':
                self.mtd_inner_state = MTDModelInnerState(obfuscation_actions=[MTDObfuscationActionShuffleWeightsTorch()])
            else:
                raise ValueError(f"Invalid mode: {mtd_mode}. Please choose from ['shuffle_flat', 'shuffle_per_layer']")

        logger.debug(f"MTDModel initialized with mode: {mtd_mode}")

    @staticmethod
    def _load_model_pickle(file_obj_or_path: Union[str, IO], close_file=True):
        # model = _pickle_load(file_obj_or_path, close_file=close_file)
        model = _torch_load(file_obj_or_path, close_file=close_file)
        return model
            
    @staticmethod
    def _save_model_pickle(model, file_obj_or_path: Union[str, IO], close_file=True):
        # _pickle_save(model, file_obj_or_path, close_file=close_file)
        _torch_save(model, file_obj_or_path, close_file=close_file)

    def obfuscate_model(self, seed:Optional[int] = None, override:bool = False, keep_hashes:bool = False):
        if override and self.mtd_inner_state.is_obfuscated():
            self.model = self.mtd_inner_state.deobfuscate_model(self.model)

        if keep_hashes:
            hash_before_obfuscation = self.model_hash()
            self.mtd_inner_state.hash_before_obfuscation = hash_before_obfuscation
            self.mtd_inner_state.hash_after_obfuscation = None

        self.model = self.mtd_inner_state.obfuscate_model(self.model, seed=seed)

        if keep_hashes:
            hash_after_obfuscation = self.model_hash()
            self.mtd_inner_state.hash_after_obfuscation = hash_after_obfuscation
            if hash_before_obfuscation == hash_after_obfuscation:
                logger.warning("Model hash before and after obfuscation are the same. Model may not have been obfuscated.")
                return None
    
    def deobfuscate_model(self, validate_hash:bool = False):
        self.model = self.mtd_inner_state.deobfuscate_model(self.model)
        if validate_hash:
            hash_after_deobfuscation = self.model_hash()
            hash_before_obfuscation = self.mtd_inner_state.hash_before_obfuscation
            if hash_after_deobfuscation != hash_before_obfuscation:
                logger.warning("Model hash mismatch after deobfuscation. Model may have been tampered with.")
                return None

    def save_mtd(self, 
                model_file_or_path: Union[str, IO],
                mtd_inner_state_file_or_path: Union[str, IO],
                close_files=True):
        
        hash_before_serialization = self.model_hash()
        self.mtd_inner_state.hash_before_serialization = hash_before_serialization
        
        self._save_model_pickle(self.model, model_file_or_path, close_file=close_files)
        self.mtd_inner_state.save(mtd_inner_state_file_or_path, close_file=close_files)

    @classmethod
    def load_mtd(cls,
                model_file_or_path: Union[str, IO],
                mtd_inner_state_file_or_path: Union[str, IO],
                close_files=True,
                validate_hash:bool = False
                ):
        model = cls._load_model_pickle(model_file_or_path, close_file=close_files)
        mtd_inner_state = MTDModelInnerState.load(mtd_inner_state_file_or_path, close_file=close_files)

        if validate_hash:
            hash_before_serialization = mtd_inner_state.hash_before_serialization
            hash_after_serialization = torch_model_hash(model)
            if hash_before_serialization != hash_after_serialization:
                logger.warning("Model hash mismatch after loading. Model may have been tampered with.")
                return None

        mtd_model = cls(model=model, mtd_inner_state=mtd_inner_state)
        return mtd_model

    def model_hash(self):
        return torch_model_hash(self.model)

    def __eq__(self, value):
        for p1, p2 in zip(self.model.parameters(), value.model.parameters()):
            if not torch.allclose(p1.data, p2.data, atol=1e-4):
                return False
        return True

    # def validate_model(self, other_model):
        


class CRYPTOModel(MTDModel):
    def __init__(self, model=None):
        from cryptography.fernet import Fernet

        if model is None:
            logger.warning("Model is None. Please provide a model.")
            return
        
        if not isinstance(model, torch.nn.Module):
            logger.warning("Model is not an instance of torch.nn.Module. MTDModel currently only supports PyTorch models.")
            raise NotImplementedError("Model is not an instance of torch.nn.Module. MTDModel currently only supports PyTorch models.")

        self.model = model

        key = Fernet.generate_key()
        fernet = Fernet(key)
        self.key = key
        self.fernet = fernet

    def encrypt_model(self):
        hash_before_serialization = self.model_hash()

        vfile = io.BytesIO()
        self._save_model_pickle(self.model, vfile, close_file=False)
        vfile.seek(0)
        encrypted_model = self.fernet.encrypt(vfile.read())
        vfile.close()

        self.encrypted_model = encrypted_model
        self.hash_before_serialization = hash_before_serialization
        return encrypted_model

    def decrypt_model(self):
        # Implement decryption logic here
        key = self.key
        fernet = self.fernet

        decrypted_model = fernet.decrypt(self.encrypted_model)
        vfile = io.BytesIO(decrypted_model)
        vfile.seek(0)
        decrypted_model = MTDModel._load_model_pickle(vfile, close_file=True)

        hash_after_serialization = torch_model_hash(decrypted_model)
        if self.hash_before_serialization != hash_after_serialization:
            logger.warning("Model hash mismatch after decryption. Model may have been tampered with.")
            return None
        self.model = decrypted_model
        return decrypted_model

__all__ = ["MTDModel", "MTDObfuscationActionShuffleWeightsFlat", "MTDObfuscationActionShuffleWeightsPerLayer", "MTDModelInnerState"]