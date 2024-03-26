import time
from functools import partial
from pathos.pools import ProcessPool, ThreadPool
from tqdm import tqdm

def parallel(func, *args, show=False, thread=False, **kwargs):
    """
    并行计算
    :param func: 函数，必选参数
    :param args: list/tuple/iterable,1个或多个函数的动态参数，必选参数
    :param show:bool,默认False,是否显示计算进度
    :param thread:bool,默认False,是否为多线程
    :param kwargs:1个或多个函数的静态参数，key-word形式
    :return:list,与函数动态参数等长
    """
    # 冻结静态参数
    p_func = partial(func, **kwargs)
    # 打开进程/线程池
    pool = ThreadPool() if thread else ProcessPool()
    try:
        if show:
            start = time.time()
            # imap方法
            with tqdm(total=len(args[0]), desc="计算进度") as t:  # 进度条设置
                r = []
                for i in pool.imap(p_func, *args):
                    r.append(i)
                    t.set_postfix({'并行函数': func.__name__, "计算花销": "%ds" % (time.time() - start)})
                    t.update()
        else:
            # map方法
            r = pool.map(p_func, *args)
        return r
    except Exception as e:
        print(e)
    finally:
        # 关闭池
        pool.close()  # close the pool to any new jobs
        pool.join()  # cleanup the closed worker processes
        pool.clear()  # Remove server with matching state