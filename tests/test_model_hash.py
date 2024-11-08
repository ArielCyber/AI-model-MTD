import random
import multiprocess

from model_mtd.model_mtd import MTDModel

import pytest

import subprocess

def _mp_run(func, timeout=10):
    ctx = multiprocess.get_context('spawn')
    def wrapper(*args, **kwargs):
        q = ctx.Queue()
        p = ctx.Process(target=func, args=args, kwargs={**kwargs, 'q':q})
        p.start()
        p.join(timeout)

        if p.is_alive():
            p.terminate()
            p.join()

            raise TimeoutError("Process did not finish in time!")

        assert p.exitcode == 0, "Process did not exit cleanly!"

        ret = q.get(block=False)
        assert ret is not None, "Queue is empty!"

        return ret
    return wrapper


def _get_hashes(options):
    return [x.model_hash() for x in options]

@_mp_run
def _get_hashes_from_options(*, options=[], q=None):
    ret = _get_hashes(options)
    q.put(ret)

def repeatfunc(func, times=None, **kwargs):
    "Repeat calls to func with specified arguments."
    rets = [None] * times
    for i in range(times):
        rets[i] = func(**kwargs)
    return rets


def test_mtd_torch_model_hash(chunk_size=250, x=3):
    torch = pytest.importorskip("torch")
    torchvision = pytest.importorskip("torchvision")

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    models = [
        torchvision.models.resnet18(),
    ]

    options = [
        MTDModel(model=model) for model in models
    ]

    for curr_chunk in chunks(options, chunk_size):
        rets = repeatfunc(_get_hashes_from_options, times=x, options=curr_chunk)
        assert all([rets[0] == ret_curr for ret_curr in rets]), "Not all runs are the same!"