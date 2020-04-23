from time import time
from libs.utils.logging import beep
from datetime import timedelta, datetime, date
from functools import wraps

class Timer(object):
    def __init__(self, message=None, audio=False, disable=False):
        self._start = None
        self._stop = None
        if message is None:
            message = "Done in {}."
        self.message = str(message)
        if "{}" not in self.message:
            self.message += " ({})"
        self.audio = audio
        self.disable = disable
        self.start()

    def start(self):
        self.clear()
        self._start = time()

    @property
    def duration(self):
        if self._stop is None:
            stop = time()
        else:
            stop = self._stop
        return timedelta(seconds=stop - self._start)

    def warn(self):
        if self.disable:
            return
        print(self.message.format(self.duration))
        if self.audio:
            beep()

    def stop(self):
        self._stop = time()

    def clear(self):
        self._start = None
        self._stop = None

    def __enter__(self):
        self.start()

    def __exit__(self, type, value, traceback):
        self.stop()
        self.warn()


def now():
    return datetime.now().time()


def today():
    return date.today()

def timeit(func):
    """
    Decorator for timing a function
    """
    @wraps(func)
    def timed_func(*args, **kwargs):
        with Timer():
            return func(*args, **kwargs)
    return timed_func