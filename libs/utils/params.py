import json


class Params(dict):
    def __init__(self, *args, **kwargs):
        super(Params, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def to_dict(self):
        return {k: v.to_dict() if isinstance(v, Params) else v for k, v in self.items()}

    @classmethod
    def to_params(cls, d):
        return Params(**{k: cls.to_params(v) if isinstance(v, dict) else v for k, v in d.items()})

    def save(self, filename):
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, filename):
        with open(filename, "r") as f:
            d = json.load(f)
        return cls.to_params(d)
