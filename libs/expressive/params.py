"""
Here, we provide an example of parameters for the expressive extraction task.
"""
from libs.utils.params import Params


BASE_PARAMS = Params(
    # Sampling size
    size=1000,
    # Random state
    seed=None,
    # Thresholds
    threshold=Params(
        adaptative=True,
        opt=0.9,
        initial=0.9,
        min=0.6,
        step=0.05,
        expressive=0.5,
        current=0.9,
    ),
    # Maximum search depth during the labeling step
    max_depth=4,
    max_depth_step=0,
    # Types of axioms to extract (passed to libs.axiom_extraction.Inducer)
    patterns=Params(
        individuals=True,
        existential=True,
    ),
    # Name or path to the embedding model (passed to libs.embeddings.load)
    embeddings="toy",
    # Clustering parameters (passed to libs.cluster.clusterize)
    clustering=Params(
        affinity="euclidean",
        linkage="ward",
    ),
    metric="harmonic",
    # Maximum length of extracted axioms
    max_axioms=2,
    min_gain=0.08,
    allow_child=False,
    # Dequeueing order (FIFO or most frequent first)
    sort_axioms=False,
    # Whether alternative best candidates should be kept (not implemented yet)
    others=Params(
        keep=True,
        n=8, # Max number of candidates to keep
        threshold=0.9, # % of the optimal score
    ),
    # Halting parameters
    halting=Params(
        min_size=30,
        max_rec_steps=40,
        max_clustering_steps=100,
        max_extracted_depth=15,
        memory_limit=110*1024**2 # gigabytes
    ),
    extra=Params(
        active=True,
        n=100,
        reset_classes=True,
        depth=20,
        threshold=0.15
    ),
    # Checkpointing
    record=Params(
        save_taxonomy=True,
        checkpoints=True,
        checkpoint_every=100,
        dirname="results/taxonomy/auto",
        name_pattern="taxonomy_{halting.max_clustering_steps}s_{timestamp:%m%d_%Hh%M}"
    ),
    display=True
)