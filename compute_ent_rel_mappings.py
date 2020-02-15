# %%
import json
import string
import itertools
import numpy as np
import pandas as pd

whitespace_trans = str.maketrans(string.punctuation, " " * len(string.punctuation))

# %%
if __name__ == "__main__":
    entity2wiki = pd.read_json(
        "benchmarks/FB15K237/entity2wikidata.json", orient="index"
    )
    entity2wiki = entity2wiki[~entity2wiki["wikipedia"].isnull()]
    labels = list(
        itertools.chain(
            *entity2wiki.label.apply(
                lambda x: x.lower().translate(whitespace_trans).split()
            )
        )
    )
    labels_mapping = {k: v for v, k in enumerate(set(labels))}
    with open("benchmarks/FB15K237/entity_mapping.json", "w+") as f:
        json.dump(labels_mapping, f)
    # %%
    relation2id = np.loadtxt(
        "benchmarks/FB15K237/relation2id.txt", delimiter="\t", skiprows=1, dtype=str
    )
    relation2id = pd.DataFrame(data=relation2id[:, 0], columns=["relation"])
    relations = list(
        itertools.chain(
            *relation2id.relation.apply(
                lambda x: x.lower().translate(whitespace_trans).split()
            )
        )
    )
    relations_mapping = {k: v for v, k in enumerate(set(relations))}
    with open("benchmarks/FB15K237/relation_mapping.json", "w+") as f:
        json.dump(relations_mapping, f)