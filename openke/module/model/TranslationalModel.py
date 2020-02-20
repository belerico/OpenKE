import os
import json
import torch
import numpy
import gensim
import string
import itertools
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model
from collections import Counter
from compact_bilinear_pooling import CountSketch, CompactBilinearPooling
from wikipedia2vec import Wikipedia2Vec


class TranslationalModel(Model):
    def __init__(
        self,
        ent_tot,
        rel_tot,
        laod_mappings=True,
        entity_mapping="benchmarks/FB15K237/entity_mapping.json",
        entity2wiki_path="benchmarks/FB15K237/entity2wikidata.json",
        entity2id_path="benchmarks/FB15K237/entity2id.txt",
        relation_mapping="benchmarks/FB15K237/relation_mapping.json",
        relation2id_path="benchmarks/FB15K237/relation2id.txt",
        word_embeddings_path="embeddings/enwiki_20180420_100d.pkl",
    ):
        super(TranslationalModel, self).__init__()
        self.ent_tot = ent_tot
        self.rel_tot = rel_tot

        if laod_mappings:
            self.entity_mapping = json.load(open(entity_mapping, "rb"))
            self.relation_mapping = json.load(open(relation_mapping, "rb"))
            if entity2wiki_path:
                self.entity2wiki = pd.read_json(entity2wiki_path, orient="index")
                self.entity2wiki = self.entity2wiki[
                    ~self.entity2wiki["wikipedia"].isnull()
                ]
            else:
                self.entity2wiki = None

            self.entity2id = pd.DataFrame(
                data=numpy.loadtxt(
                    entity2id_path, delimiter="\t", skiprows=1, dtype=str
                ),
                columns=["entity", "id"],
            )
            self.relation2id = pd.DataFrame(
                data=numpy.loadtxt(
                    relation2id_path, delimiter="\t", skiprows=1, dtype=str
                ),
                columns=["relation", "id"],
            )
            if word_embeddings_path is None:
                raise Exception("The path for the word embeddings must be set")

            if word_embeddings_path.split("/")[-1].split(".")[-1] == "txt":
                self.word_embeddings = gensim.models.KeyedVectors.load_word2vec_format(
                    word_embeddings_path
                )
            else:
                self.word_embeddings = Wikipedia2Vec.load(
                    open(word_embeddings_path, "rb")
                )

    def _initialize_embeddings(self, embeddings_name, terms, merge, idx):
        if len(terms) >= 1:
            embeddings = getattr(self, embeddings_name)
            embeddings.weight.data[int(idx)] = torch.Tensor(
                self.word_embeddings.get_word_vector(terms[0])
            ).data
            if merge == "cbp":
                for k in range(1, len(terms)):
                    embeddings.weight.data[int(idx)] = self.CBP(
                        torch.Tensor(
                            self.word_embeddings.get_word_vector(terms[k])
                        ).data,
                        embeddings.weight.data[int(idx)],
                    )
            elif merge == "sum" or merge == "mean":
                for k in range(1, len(terms)):
                    embeddings.weight.data[int(idx)] += torch.Tensor(
                        self.word_embeddings.get_word_vector(terms[k])
                    ).data
            elif merge == "hadamard":
                for k in range(1, len(terms)):
                    embeddings.weight.data[int(idx)] = torch.mul(
                        torch.Tensor(
                            self.word_embeddings.get_word_vector(terms[k])
                        ).data,
                        embeddings.weight.data[int(idx)],
                    )
            if merge == "mean":
                embeddings.weight.data[int(idx)] /= len(terms)

    def _get_existing_terms(self, terms):
        existing_terms = []
        for term in terms:
            try:
                self.word_embeddings.get_word_vector(term)
                existing_terms.append(term)
            except:
                try:
                    self.word_embeddings.get_word_vector(string.capwords(term))
                    existing_terms.append(string.capwords(term))
                except:
                    continue
        return existing_terms

    def _extract_terms(self, s):
        return list(set(s.lower().translate(self.whitespace_trans).split()))

    def initialize_embeddings(self, entity_vector=True, merge="sum"):
        self.whitespace_trans = str.maketrans(
            string.punctuation, " " * len(string.punctuation)
        )
        self.underscore_trans = str.maketrans("-_", " " * 2)
        if merge == "cbp":
            self.CBP = CompactBilinearPooling(self.dim, self.dim, self.dim)
        if entity_vector:
            for entity, idx in self.entity2id.itertuples(index=False, name=None):
                try:
                    if self.entity2wiki is not None:
                        entity_url = (
                            self.entity2wiki[["wikipedia"]].loc[entity].values[0]
                        )
                        entity_name = os.path.basename(entity_url).replace("_", " ")
                    elif self.entity_mapping is not None:
                        entity_name = self.entity_mapping[entity]["label"].translate(
                            self.underscore_trans
                        )
                        entity_name = string.capwords(entity_name)
                    else:
                        raise Exception(
                            "A mapping is needed from entities to text is needed"
                        )
                except KeyError:
                    continue
                try:
                    self.ent_embeddings.weight.data[int(idx)] = torch.Tensor(
                        self.word_embeddings.get_entity_vector(entity_name)
                    ).data
                except KeyError:
                    continue
        else:
            for entity, idx in self.entity2id.itertuples(index=False, name=None):
                try:
                    if self.entity2wiki is not None:
                        entity_url = (
                            self.entity2wiki[["wikipedia"]].loc[entity].values[0]
                        )
                        entity_name = os.path.basename(entity_url)
                    elif self.entity_mapping is not None:
                        entity_name = self.entity_mapping[entity]["label"]
                    else:
                        raise Exception(
                            "A mapping is needed from entities to text is needed"
                        )
                    terms = self._extract_terms(entity_name)
                    existing_terms = self._get_existing_terms(terms)
                    self._initialize_embeddings(
                        "ent_embeddings", existing_terms, merge, idx
                    )
                except KeyError:
                    continue

        for relation, idx in self.relation2id.itertuples(index=False, name=None):
            terms = self._extract_terms(relation)
            existing_terms = self._get_existing_terms(terms)
            self._initialize_embeddings("rel_embeddings", existing_terms, merge, idx)

        del self.word_embeddings

    def forward(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError
