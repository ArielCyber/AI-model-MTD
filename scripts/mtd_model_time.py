import gc
import os
from typing import Optional
from model_mtd.model_utils import get_num_weights_torch, get_weights_dtype_torch
import torch
import torchvision
import torchvision.models as models

import tqdm
import pandas as pd

from model_mtd.model_mtd import MTDModel

import time
import io

from functools import partial

def mtd_time(model):
    """
        MTD construction
    """
    time_start_construct = time.time()

    mtd_model = MTDModel(model)

    time_end_construct = time.time()
    time_construct = time_end_construct - time_start_construct

    """
        MTD obfuscation
    """

    time_start_obfuscate = time.time()

    mtd_model.obfuscate_model()

    time_end_obfuscate = time.time()
    time_obfuscate = time_end_obfuscate - time_start_obfuscate

    """
        MTD deobfuscation
    """

    time_start_deobfuscate = time.time()

    mtd_model.deobfuscate_model()

    time_end_deobfuscate = time.time()
    time_deobfuscate = time_end_deobfuscate - time_start_deobfuscate

    """
        MTD save
    """

    model_vfile = io.BytesIO()
    mtd_inner_state_vfile = io.BytesIO()

    time_start_save = time.time()

    mtd_model.save_mtd(model_vfile, mtd_inner_state_vfile, close_files=False)

    time_end_save = time.time()
    time_save = time_end_save - time_start_save

    """
        MTD load
    """

    model_vfile.seek(0)
    mtd_inner_state_vfile.seek(0)

    time_start_load = time.time()

    MTDModel.load_mtd(model_vfile, mtd_inner_state_vfile)

    time_end_load = time.time()
    time_load = time_end_load - time_start_load

    model_vfile.close()
    mtd_inner_state_vfile.close()

    ret_dict = {
        'time_construct': time_construct,
        'time_obfuscate': time_obfuscate,
        'time_deobfuscate': time_deobfuscate,
        'time_save': time_save,
        'time_load': time_load
    }

    return ret_dict

if __name__ == '__main__':
    repeat_amnt = 10

    dfs = []

    for model_name in tqdm.tqdm(torchvision.models.list_models()):
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

    df.to_csv(os.path.join(results_dir, 'mtd_model_time_results.csv'), index=False)