from functools import wraps
import time

# 测试函数耗时
__USED_TIME  = 0
__CALL_TIMES  = 0

def clear():
    global __USED_TIME
    global __CALL_TIMES
    __USED_TIME  = 0
    __CALL_TIMES  = 0

def avg_used_time():
    if __CALL_TIMES == 0:
        return 0
    else:
        return int((__USED_TIME / __CALL_TIMES) * 1000)

def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        global __USED_TIME
        global __CALL_TIMES

        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        __USED_TIME += (t2-t1)
        __CALL_TIMES += 1
        return result
    return measure_time
