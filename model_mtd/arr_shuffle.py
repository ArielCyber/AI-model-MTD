import numpy as np
from itertools import product

def shuffle_all_axes(arr):
    """
    Shuffle a numpy array independently along all axes and all sub-arrays.
    
    Parameters:
    arr (np.ndarray): Input array of any shape
    
    Returns:
    tuple: (shuffled_array, shuffle_indices)
        - shuffled_array: Array shuffled along all axes
        - shuffle_indices: Dictionary containing indices for recovery
    """
    shape = arr.shape
    n_dims = len(shape)
    result = arr.copy()
    indices_dict = {}
    
    # Generate all possible slice combinations for each dimension depth
    for depth in range(n_dims):
        # Generate all possible combinations of indices for fixed dimensions
        fixed_dims = range(depth)
        slice_indices = list(product(*[range(shape[d]) for d in fixed_dims]))
        
        # For each combination of fixed dimensions
        for fixed_idx in slice_indices:
            # Create the indexing tuple for this sub-array
            if fixed_idx:
                idx_tuple = fixed_idx + (slice(None),) + (slice(None),) * (n_dims - depth - 1)
                key = fixed_idx + (depth,)
            else:
                idx_tuple = (slice(None),) * n_dims
                key = (depth,)
            
            # Get the sub-array
            sub_arr = result[idx_tuple]
            
            # Generate and apply random permutation for this dimension
            perm = np.random.permutation(shape[depth])
            
            # Create the appropriate number of ellipsis and slices
            idx_list = list(idx_tuple)
            idx_list[depth] = perm
            result[tuple(idx_list)] = sub_arr
            
            # Store the permutation
            indices_dict[key] = perm
    
    return result, indices_dict

def recover_original(shuffled_arr, shuffle_indices):
    """
    Recover the original array using the shuffle indices.
    
    Parameters:
    shuffled_arr (np.ndarray): Shuffled array
    shuffle_indices (dict): Dictionary of indices returned by shuffle_all_axes
    
    Returns:
    np.ndarray: Recovered original array
    """
    result = shuffled_arr.copy()
    shape = result.shape
    n_dims = len(shape)
    
    # Process dimensions in reverse order
    for depth in range(n_dims - 1, -1, -1):
        # Get all possible fixed dimension combinations
        fixed_dims = range(depth)
        slice_indices = list(product(*[range(shape[d]) for d in fixed_dims]))
        
        # For each combination of fixed dimensions
        for fixed_idx in slice_indices:
            # Reconstruct the key
            if fixed_idx:
                key = fixed_idx + (depth,)
            else:
                key = (depth,)
            
            # Get the permutation
            perm = shuffle_indices[key]
            
            # Create the indexing tuple
            if fixed_idx:
                idx_tuple = fixed_idx + (slice(None),) + (slice(None),) * (n_dims - depth - 1)
            else:
                idx_tuple = (slice(None),) * n_dims
            
            # Get the sub-array
            sub_arr = result[idx_tuple]
            
            # Apply inverse permutation
            inv_perm = np.argsort(perm)
            idx_list = list(idx_tuple)
            idx_list[depth] = inv_perm
            result[tuple(idx_list)] = sub_arr
    
    return result

# Utility function to verify shuffling effectiveness
def verify_shuffling(original, shuffled, dimension):
    """
    Verify that shuffling occurred along the specified dimension.
    Returns the percentage of elements that changed position.
    """
    total_elements = np.prod(original.shape)
    changed = np.sum(original != shuffled)
    return (changed / total_elements) * 100


def shuffle(arr, seed=None):
    shape = arr.shape

    ret = arr.copy().ravel()

    rng = np.random.default_rng(seed)
    idxs_perm = rng.permutation(arr.size)

    ret = ret[idxs_perm]

    return ret.reshape(shape), idxs_perm

def recover(arr, idxs_perm):
    shape = arr.shape

    ret = arr.copy().ravel()

    ret = ret[np.argsort(idxs_perm)]

    return ret.reshape(shape)
    