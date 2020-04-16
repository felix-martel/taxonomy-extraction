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