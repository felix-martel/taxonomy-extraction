FILES = {
        "rel": "relation2id.txt",
        "ent": "entity2id.txt",
        "triples": "triples2id.txt",
        "train": "train2id.txt",
        "test": "test2id.txt",
        "val": "valid2id.txt"
    }


def iter_training_files(files):
    for file in files:
        with open(file, "r", encoding="utf8") as f:
            next(f)
            for line in f:
                h, t, r = map(int, line.rstrip().split(" "))
                yield h, r, t, file


def get_item_number(files):
    n_items = 0
    for file in files:
        with open(file, "r") as f:
            n_items += int(next(f))
    return n_items


def split_line(line, sep=" "):
    a = line.rstrip().split(sep)
    idx = a[-1]
    name = sep.join(a[:-1])
    return name, int(idx)