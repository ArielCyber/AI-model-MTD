import copy
import gc
import os
from typing import Optional
from model_mtd.model_utils import get_num_weights_torch, get_weights_dtype_torch
import torch
import torchvision
import torchvision.models as models

import tqdm
import pandas as pd

import logging
import sys

root = logging.getLogger()
root.setLevel(logging.WARNING)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

from model_mtd.model_mtd import MTDModel, CRYPTOModel

import time
import io

from functools import partial



def timeit(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
        
    elapsed_time = end_time - start_time
    return elapsed_time, result

def mtd_time(model):
    model_cp = copy.deepcopy(model.to('cpu'))

    """
        Regular Save
    """
    model_vfile = io.BytesIO()

    time_save_reg, _ = timeit(MTDModel._save_model_pickle, model_cp, model_vfile, close_file=False)

    """
        Regular Load
    """

    model_vfile.seek(0)

    time_load_reg, _ = timeit(MTDModel._load_model_pickle, model_vfile, close_file=False)
    model_vfile.close()

    ret_dict = {
        'time_save_reg': time_save_reg,
        'time_load_reg': time_load_reg
    }

    """
        MTD construction
    """
    model_cp = copy.deepcopy(model.to('cpu'))
    time_construct, mtd_model = timeit(MTDModel, model_cp)

    """
        MTD obfuscation
    """

    time_obfuscate, _ = timeit(MTDModel.obfuscate_model, mtd_model)

    """
        MTD deobfuscation
    """

    time_deobfuscate, _ = timeit(MTDModel.deobfuscate_model, mtd_model)

    """
        MTD save
    """

    model_vfile = io.BytesIO()
    mtd_inner_state_vfile = io.BytesIO()

    time_save_mtd, _ = timeit(MTDModel.save_mtd, mtd_model, model_vfile, mtd_inner_state_vfile, close_files=False)

    """
        MTD load
    """

    model_vfile.seek(0)
    mtd_inner_state_vfile.seek(0)

    time_load_mtd, _ = timeit(MTDModel.load_mtd, model_vfile, mtd_inner_state_vfile)

    model_vfile.close()
    mtd_inner_state_vfile.close()

    ret_dict.update({
        'time_construct_mtd': time_construct,
        'time_obfuscate_mtd': time_obfuscate,
        'time_deobfuscate_mtd': time_deobfuscate,
        'time_save_mtd': time_save_mtd,
        'time_load_mtd': time_load_mtd
    })

    """
        CRYPTOModel
    """

    model_cp = copy.deepcopy(model.to('cpu'))
    crypt_model = CRYPTOModel(model_cp)

    """
        CRYPT ENCRYPTION
    """

    time_encrypt_encrypt, _ = timeit(CRYPTOModel.encrypt_model, crypt_model)

    """
        CRYPT DECRYPTION
    """
    time_decrypt_encrypt, _ = timeit(CRYPTOModel.decrypt_model, crypt_model)

    ret_dict.update({
        'time_encrypt_crypto': time_encrypt_encrypt,
        'time_decrypt_crypto': time_decrypt_encrypt
    })

    return ret_dict

if __name__ == '__main__':
    repeat_amnt = 10

    dfs = []

    for model_name in tqdm.tqdm(torchvision.models.list_models()):
    # for model_name in tqdm.tqdm(['vgg11',]):
        model = torchvision.models.get_model(model_name, weights="DEFAULT")

        model_n_weights = get_num_weights_torch(model)
        model_dtype_str = str(get_weights_dtype_torch(model))

        time_results = []
        for i in range(repeat_amnt):
            gc.collect()

            ret = mtd_time(model)
            ret['run_name'] = i

            time_results.append(ret)

        df = pd.DataFrame(time_results)
        df['model_name'] = model_name
        df['model_n_weights'] = model_n_weights
        df['model_dtype_str'] = model_dtype_str

        dfs.append(df)

    df = pd.concat(dfs)
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)

    df.to_csv(os.path.join(results_dir, 'mtd_model_time_results_saveload_all.csv'), index=False)