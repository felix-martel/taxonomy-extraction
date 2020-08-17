import os
import datetime as dt
from libs.utils.params import Params


class ExtractionParams(Params):
    def __init__(self, *args, **kwargs):
        super(ExtractionParams, self).__init__(*args, **kwargs)
        self.on_load()

    def on_load(self):
        self.record.taxname = self.record.name_pattern.format(timestamp=dt.datetime.now(), **self)
        self.record.directory = os.path.join(self.record.dirname, self.record.taxname)


