import functools
import time


class Timer(object):

    def time(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print('Starting {}...'.format(func.__name__))
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print('Done. Elapsed time: {} seconds.'.format(end - start))
            return result

        return wrapper

    time = staticmethod(time)
