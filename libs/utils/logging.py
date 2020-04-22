import os
from libs.const import AUDIO_BEEP


try:
    from IPython.display import Audio, display

    assert os.path.exists(AUDIO_BEEP)


    def beep(disable=False):
        if not disable:
            display(Audio(filename=AUDIO_BEEP, autoplay=True))

except (ImportError, AssertionError):
    # IPython not available or audio file not found: fall back to winsound
    from winsound import MessageBeep


    def beep(disable=False):
        if not disable:
            MessageBeep()

import traceback
import sys


# Context manager that copies stdout and any exceptions to a log file
class Tee(object):
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.stdout = sys.stdout
        self._history = ""

    def __enter__(self):
        sys.stdout = self

    def __exit__(self, exc_type, exc_value, tb):
        sys.stdout = self.stdout
        if exc_type is not None:
            exc = traceback.format_exc()
            self.file.write(exc)
            self._history += exc
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self._history += data

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def history(self):
        return self._history
