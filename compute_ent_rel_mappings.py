# %%
import json
import string
import itertools
import numpy as np
import pandas as pd

whitespace_trans = str.maketrans(string.punctuation, " " * len(string.punctuation))

# %%
if __name__ == "__main__":
    mid2name = pd.read_csv(
        "benchmarks/FB15K237/mid2name.tsv", delimiter="\t", header=None
    )
    mid2name.columns = ["relation", "label"]
    mid2name = mid2name.drop_duplicates(subset=["relation"])
    entity2wiki = pd.read_json(
        "benchmarks/FB15K237/entity2wikidata.json", orient="index"
    )
    entity2wiki = entity2wiki[~entity2wiki["wikipedia"].isnull()]
    new_entities = []
    for relation, label in mid2name.itertuples(index=False, name=None):
        if relation not in entity2wiki.index:
            new_entities.append([str(relation), [], None, str(label), None, None])
    new_entities = np.array(new_entities)
    new_entities = pd.DataFrame(
        data=new_entities[:, 1:], columns=entity2wiki.columns, index=new_entities[:, 0]
    )
    entity2wiki = pd.concat([entity2wiki, new_entities])
    entity2wiki.to_json("benchmarks/FB15K237/entity2wikidata.json")
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
