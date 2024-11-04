import torch
import torchvision.models as models
from model_mtd.ArrayRandomization import *
import pickle

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
    print(obf_dict)
    for key, data in obf_dict.items():
        for switch in data:
            # Switch the positions of two blocks in 'layer1' (e.g., blocks 0 and 3)
            switch_blocks(model, target_layer=key, block_idx1=switch[0], block_idx2=switch[1])

class MTDModel:
    map_model = []
    def __init__(self, model_architecture=None):
        if model_architecture:
            self.model = model_architecture

    def load_model_pickle(self,file_path):
        with open(file_path,"rb") as f:
            self.model = pickle.load(f)
            
    def save_model_pickle(self,file_path):
        with open(file_path,"wb") as f:
            pickle.dump(self.model,f)

    def change_weights(self,seed):
        i = 0
        for tensor in self.model.parameters():
            i+=1
            weight_list = tensor.data.tolist()


            array_size = tensor.size()
            array, indices = randomize(weight_list,seed)
            self.map_model.append(indices)
            new_weights_list = array
            # Create an array of size 5x3

            # Convert the list to a tensor
            new_weights_tensor = torch.tensor(new_weights_list)
            # Update the tensor with new values
            with torch.no_grad():
                tensor.data = new_weights_tensor
    
    def retrive_weights(self):
        i = 0
        for tensor in self.model.parameters():
            i+=1
            weight_list = tensor.data.tolist()


            array_size = tensor.size()
            array = retrieve(weight_list,self.map_model[i-1])
            new_weights_list = array
            # Create an array of size 5x3

            # Convert the list to a tensor
            new_weights_tensor = torch.tensor(new_weights_list)
            # Update the tensor with new values
            with torch.no_grad():
                tensor.data = new_weights_tensor

    def save_mtd(self, file_path, map_path,model_map_path, seed = -1):
        self.change_weights(seed)
        obf_dict = model_obfuscation(self.model)
        with open(file_path,"wb") as f:
            pickle.dump(self.model,f)
        with open(map_path,"wb") as f:
            pickle.dump(self.map_model,f)
        with open(model_map_path,"wb") as f:
            pickle.dump(obf_dict,f)

    def load_mtd(self, file_path,map_path,model_map_path):
        obf_dict = None
        with open(model_map_path,"rb") as f:
            obf_dict = pickle.load(f)
        with open(file_path,"rb") as f:
            self.model = pickle.load(f)
        with open(map_path,"rb") as f:
            self.map_model = pickle.load(f)
            
        model_deobfuscation(self.model, obf_dict)    
        self.retrive_weights()

    def validate_model(self, other_model):
        for p1, p2 in zip(self.model.parameters(), other_model.model.parameters()):
            if not torch.allclose(p1.data, p2.data, atol=1e-4):
                return False
        return True