def mapping_to_axioms(cls_to_clu, clu):
    selected_clusters = {clu:cls for cls, clu in cls_to_clu.items()}   
    axioms = []
    for child_name, cluster in cls_to_clu.items():
        node = clu[cluster] #ster # self.clu.tree[cluster]
        if node.is_root:
            continue
        parent = node.parent
        while not parent.is_root:
            parent = parent.parent
            if parent.id in selected_clusters:
                parent_name = selected_clusters[parent.id]
                axioms.append((child_name, parent_name))
                break
    return set(axioms)

to_axioms = mapping_to_axioms