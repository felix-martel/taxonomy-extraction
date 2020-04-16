# taxonomy-extraction
Code for automatically extracting expressive and non-expressive taxonomies from knowledge graphs (research project at the [Lama-West lab](http://labowest.ca/?clang=en), Polytechnique Montréal, Canada).

The core code is contained in `libs`:
- `libs.graph`: load and query knowledge graphs.
- `libs.dataset`: create datasets for the taxonomy extraction task, or load existing datasets from file.
- `libs.axiom`: provide useful classes for axiom handling and boolean logic.
- `libs.models`: train and use knowledge graph embedding models. For now, we mostly use [OpenKE](https://github.com/thunlp/OpenKE), so `libs.models` only contains an implementation of RDF2Vec.
- `libs.extraction`: extract taxonomies from a `Dataset` and an embedding matrix.
- `libs.axiom_induction`: extract expressive taxonomies from a `Dataset`, an embedding matrix and a knowledge graph

See also:
- `libs.clustering`: create and handle clustering trees.
- `libs.taxonomy`: build, display and save taxonomic trees.
- `libs.tree`: basic operations over trees (creation and modifications, BFS, DFS, IO operations, visualizations).
- `libs.utils`: various utility functions, used all over the project

To provide real-world examples and use-cases for the code described above, Jupyter notebooks will be uploaded shortly.

This project mostly relies on `numpy`, `scipy`, `scikit-learn`, `OpenKE`. `networkx` and `plotly` are required for plotting interactive trees.

