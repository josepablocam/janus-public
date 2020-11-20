import multiprocessing as mp
from multiprocessing.context import TimeoutError
import sys


def run(seconds, fun, *args, **kwargs):
    if seconds >= 0:
        pool = mp.get_context("spawn").Pool(processes=1)
        try:
            proc = pool.apply_async(fun, args, kwargs)
            result = proc.get(seconds)
            return result
        except mp.TimeoutError:
            pool.terminate()
            pool.close()
            raise mp.TimeoutError()
        finally:
            pool.terminate()
            pool.close()
    else:
        # if no timeout, then no point
        # in incurring cost of running as separate process
        # so call locally
        return fun(*args, **kwargs)
