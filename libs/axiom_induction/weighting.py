


def compute_or_tfidf(clid, x, y):
    or_axiom = F[clid, x] | F[clid, y]
    tf = np.mean(or_axiom)
    aname = f"{ax_to_str(x)} ∨ {ax_to_str(y)}"
    
    ref_depth = F.clu[clid].depth
    max_tf = np.max([np.mean(F[clid2,x] | F[clid2,y]) for clid2 in cl2id if F.clu[clid2].depth <= ref_depth and clid2 != clid])
    idf = np.log(1 + 1 / max_tf)
    return aname, tf, idf, tf*idf


def compute_idf(clid, ax, F):
    tf = np.mean