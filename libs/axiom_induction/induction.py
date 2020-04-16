from tqdm import tqdm
import numpy as np
import libs.axiom_induction.patterns as patterns

class InducedAxiomMatrix:
    def __init__(self, clustering, data, graph):
        self.clu = clustering
        self.data = data
        self.graph = graph
        
        # Map entity id in the knowledge graph to their corresponding index in the entity-axiom matrix
        self.eid_to_index = {eid: i for i, eid in zip(self.data.ids, self.data.indices)}
        
        # self.cl2id = {cl: i for i, cl in enumerate(self.clu.root.items())} DEPRECATED
        self._M, self.to_ax, self.to_id = self._compute_matrix()
        
    def _compute_matrix(self, init_axioms=[], allow_new_axioms=True):
        id2ax = init_axioms
        ax2id = {cl: i for i, cl in enumerate(init_axioms)}
        axioms = []
        
        n_entities = len(self.data)
        for i, ent in tqdm(zip(self.data.ids, self.data.indices), total=n_entities):
            axs = patterns.extract_from_entity(ent, self.graph)
            for ax in axs:
                if ax in ax2id:
                    j = ax2id[ax]
                elif allow_new_axioms:
                    j = len(id2ax)
                    id2ax.append(ax)
                    ax2id[ax] = j
                else:
                    continue
                axioms.append((i, j))
        
        n_axioms = len(ax2id)
        
        A = np.zeros((n_entities, n_axioms), dtype=bool)
        for i, j in axioms:
            A[i, j] = True

        return A, id2ax, ax2id
        
    def __getitem__(self, x):
        """
        Access a Cluster Axiom from cluster id and relation and tail strings (r, t) from triple (h, r, t)
        
        Let `c` be a cluster of size c_size with id cid, `a` be an axiom, e.g. a = ("birthDate", "xsd:Date")
        Let n_axioms be the number of axioms. Then:
        self[cid, :] has dimension (c_size, n_axioms): cluster-axiom matrix for cluster `c`
        self[:, a] has dimension (n_items,): boolean axiom vector for axiom `a`
        self[cid, a] has dimension (c_size,) cluster-specific axiom vector for axiom `a`
        """
        cid, aid = x # TODO: handle invalid arguments
        aid = self.to_id[aid]
        return self._M[self.cmask(cid), aid]
    
    def cmask(self, cid):
        """
        Return the cluster mask for cluster `cid`
        
        The cluster mask `cmask(c)` of cluster c is a boolean vector of dimension (n_items,) such that
        cmask(c)_i = 1 iff entity i belongs to cluster c. It can be used to select a submatrix from _M :
        _M[cmask(c)] is the entity-axiom matrix corresponding to cluster c, of dimension (c.size, n_axioms)
        """
        if cid == slice(None, None, None): return np.ones(self.n_items, dtype=bool)
        
        cid = list(self.clu[cid].items())
        mask = np.zeros(self.shape[0], dtype=bool)
        mask[cid] = True
        return mask
    
    @property
    def n_axioms(self):
        """Number of candidate axioms (patterns) in the dataset"""
        return self.shape[1]
    
    @property
    def n_items(self):
        """Number of entities in the dataset"""
        return self.shape[0]
    
    @property
    def shape(self):
        """Shape of cluster-axiom matrix"""
        return self._M.shape