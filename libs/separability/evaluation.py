import sklearn.metrics as metrics
from scipy.spatial.distance import euclidean, cosine
from sklearn.svm import LinearSVC
from collections import defaultdict, Counter
import numpy as np

from sklearn.model_selection import train_test_split

def evaluate(X, y, svc_params={}, split_params={}):
    X_train, X_test, y_train, y_true = train_test_split(X, y, **split_params)
    clf = LinearSVC(**svc_params).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return dict(
        prec=metrics.precision_score(y_true, y_pred),
        rec=metrics.recall_score(y_true, y_pred),
        f1=metrics.f1_score(y_true, y_pred),
        acc=metrics.accuracy_score(y_true, y_pred)
    )

def taxonomic_distance(A, B, tax):
    """Return the taxonomic distance between class A and class B
    
    A, B: str, name of the classes to compare
    tax: libs.taxonomy.Taxonomy
    """
    TA = tax[A]
    TB = tax[B]
    common_ancestors = set(TA.all) & set(TB.all)
    min_depth = max([tax[x].depth for x in common_ancestors])

    w =1
    a_cost = sum(w/(k+1) for k in range(min_depth, TA.depth))
    b_cost = sum(w/(k+1) for k in range(min_depth, TB.depth))
    cost = a_cost + b_cost
    return cost

def geometric_distance(A, B, centroids):
    return euclidean(centroids[A], centroids[B])

def class_distance(a, b, tax, centroids):
    taxo = taxonomic_distance(a, b, tax)
    geom = geometric_distance(a, b, centroids)
    mixed = (taxo + geom) / 2
    return dict(
        taxo=taxo,
        geom=geom,
        mixed=mixed
    )




