import numpy as np

def load_weights_from_flattened_vector_torch(model: "torch.nn.Module", model_weights: np.ndarray, inplace: bool = False) -> "torch.nn.Module":
    import torch
    if inplace:
        model_curr = model
    else:
        model_curr = torch.clone(model)

    if not isinstance(model_weights, torch.Tensor):
        model_weights = torch.from_numpy(model_weights)

    torch.nn.utils.vector_to_parameters(model_weights, model_curr.parameters())

    return model_curr

def extract_weights_torch(model: "torch.nn.Module") -> np.ndarray:
    ws = [w.cpu().detach().numpy().flatten() for w in model.parameters()]
    w = np.concatenate(ws)

    return w

def get_num_weights_torch(model: "torch.nn.Module") -> int:
    s = 0
    for i, tensor in enumerate(model.parameters()):
        s += tensor.numel()

    return s

def get_weights_dtype_torch(model: "torch.nn.Module") -> np.dtype:
    return next(model.parameters()).dtype