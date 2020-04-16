def safe_divide(p, q):
    if q == 0: return 0.
    return p / q

def precision(counter, cls):
    return safe_divide(counter[cls], sum(counter.values()))
    #return counter[cls] / sum(counter.values())

def recall(counter, cls, class_counts):
    return safe_divide(counter[cls], class_counts[cls])
    #return counter[cls] / class_counts[cls]

def precision_recall(counter, cls, class_counts):
    return precision(counter, cls), recall(counter, cls, class_counts)

def f_score(counter, cls, class_counts):
    counter = counter.composition
    p, r = precision_recall(counter, cls, class_counts)
    if not (p or r):
        return 0.
    return 2 * (p * r) / (p+r)