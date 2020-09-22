# taxonomy-extraction
Code for automatically extracting expressive and non-expressive taxonomies from knowledge graphs (research project at the [Lama-West lab](http://labowest.ca/?clang=en), Polytechnique Montréal, Canada).

![Overview of the taxonomy extraction method](https://github.com/felix-martel/taxonomy-extraction/raw/master/data/img/summary.png)

**Non-expressive :**
Starting from a knowledge graph KG and a set of entity-type pairs, (1) entities are embedded into a *d*-dimensional vector space (2) then they’re hierarchically clustered; (3) each type in the original dataset is then mapped to one of the cluster, (4) the taxonomy is extracted by removing non-selected clusters.

Two methods for mapping types to clusters: *Hard Mapping* and *Soft Mapping*. Hard Mapping computes an optimal, one-for-one injective mapping between types and clusters by solving a linear sum assignment problem (right now I'm using the [Kuhn-Munkres algorithm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html) but I plan to implement a more efficient heuristic, such as [Asymmetric Greedy Search](https://link.springer.com/article/10.1007/s10878-015-9979-2)). Soft Mapping defines a mapping score between types and clusters, recursively computes a probability for each subsumption axiom, and performs a transitive reduction on the resulting DAG.

*Preliminary Results*

|             |          | direct |      |      | transitive |      |      |
|-------------|----------|--------|------|------|------------|------|------|
| Method      | Model    | cos    | euc  | eucw | cos        | euc  | eucw |
| HardMapping | ComplEx  | 0,52   | 0,49 | 0,47 | 0,74       | 0,67 | 0,65 |
|             | DistMult | 0,5    | 0,39 | 0,39 | 0,64       | 0,59 | 0,63 |
|             | RDF2Vec  | 0,50   | 0,31 | 0,56 | 0,63       | 0,46 | 0,74 |
|             | TransE   | 0,81   | 0,63 | 0,7  | **0,93**       | 0,76 | 0,8  |
| SoftMapping | ComplEx  | 0,5    | 0,48 | 0,47 | 0,79       | 0,74 | 0,7  |
|             | DistMult | 0,47   | 0,45 | 0,45 | 0,74       | 0,74 | 0,74 |
|             | RDF2Vec  | 0,82   |      |      |            |      |      |
|             | TransE   | **0,82**   | 0,69 | **0,77** | **0,93**       | **0,84** | **0,92** |
| TIEmb       | ComplEx  | 0,38   | 0,4  |      | 0,37       | 0,48 |      |
|             | DistMult | 0,26   | 0,39 |      | 0,27       | 0,47 |      |
|             | RDF2Vec* | 0,81   | 0,73 |      | 0,57       | 0,42 |      |
|             | TransE   | 0,74   | **0,76** |      | 0,86       | 0,79 |      |


---

**Expressive :** for expressive taxonomy extraction, the algorithm starts from an axiom *A*, sample *n* entities verifying this axiom, and run a hierarchical clustering over them. The clusters are then labelled by expressive axioms using statistics on linked data, and a taxonomic tree *T(A)* is extracted. Then, *T(A)* is iteratively expanded by sampling new entities from the axioms in *T(A)* and adding the extracted subtrees to *T(A)*. 


The core code is contained in `libs`:
- `libs.graph`: load and query knowledge graphs.
- `libs.dataset`: create datasets for the taxonomy extraction task, or load existing datasets from file.
- `libs.axiom`: provide useful classes for axiom handling and boolean logic.
- `libs.models`: train and use knowledge graph embedding models. For now, we mostly use [OpenKE](https://github.com/thunlp/OpenKE), so `libs.models` only contains an implementation of RDF2Vec.
- `libs.extraction`: extract taxonomies from a `Dataset` and an embedding matrix.
- `libs.axiom_extraction`: extract expressive taxonomies from an embedding matrix and a knowledge graph

See also:
- `libs.clustering`: create and handle clustering trees.
- `libs.taxonomy`: build, display and save taxonomic trees.
- `libs.tree`: basic operations over trees (creation and modifications, BFS, DFS, IO operations, visualizations).
- `libs.utils`: various utility functions, used all over the project

The Jupyter notebook `Getting_Started` provides examples and use-cases for the code described above. 
You can also refer to the associated thesis (in `papers/memoire.pdf`, in French) for in-depth description of datasets,
methods and algorithms.

This project mostly relies on `numpy`, `scipy`, `scikit-learn`, `OpenKE`. `networkx` and `plotly` are required for plotting interactive trees.

