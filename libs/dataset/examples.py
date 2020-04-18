from .dataset import Dataset

# Create a toy dataset with classes A, B, C, D and 10 entities per class
classes = "ABCD"
n_entities = 10

labels = [i for i in range(len(classes)) for _ in range(n_entities)]
indices = list(range(len(labels)))
name2cls = {k: v for v, k in enumerate(classes)}
cls2name = {v: k for k, v in name2cls.items()}
axioms = [("B", "A"), ("C", "A"), ("D", "B")]

toy_data = Dataset(indices, labels, name2cls, cls2name, axioms)

if __name__ == "__main__":
    print(toy_data.summary())
