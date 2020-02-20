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


class TransHW(Model):
    def __init__(
        self,
        ent_tot,
        rel_tot,
        dim=100,
        p_norm=1,
        norm_flag=True,
        margin=None,
        epsilon=None,
        laod_mappings=True,
        entity_mapping="benchmarks/FB15K237/entity_mapping.json",
        entity2wiki_path="benchmarks/FB15K237/entity2wikidata.json",
        entity2id_path="benchmarks/FB15K237/entity2id.txt",
        relation_mapping="benchmarks/FB15K237/relation_mapping.json",
        relation2id_path="benchmarks/FB15K237/relation2id.txt",
        word_embeddings_path="embeddings/enwiki_20180420_100d.pkl",
    ):
        super(TransHW, self).__init__(ent_tot, rel_tot)

        self.dim = dim
        self.margin = margin
        self.epsilon = epsilon
        self.norm_flag = norm_flag
        self.p_norm = p_norm

        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)
        self.norm_vector = nn.Embedding(self.rel_tot, self.dim)

        self.whitespace_trans = str.maketrans(
            string.punctuation, " " * len(string.punctuation)
        )
        if laod_mappings:
            self.entity_mapping = json.load(open(entity_mapping, "rb"))
            self.relation_mapping = json.load(open(relation_mapping, "rb"))
            self.entity2wiki = pd.read_json(entity2wiki_path, orient="index")
            self.entity2wiki = self.entity2wiki[~self.entity2wiki["wikipedia"].isnull()]
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
            if word_embeddings_path.split("/")[-1] == ".txt":
                self.word_embeddings = gensim.models.KeyedVectors.load_word2vec_format(
                    word_embeddings_path
                )
            else:
                self.word_embeddings = Wikipedia2Vec.load(
                    open(word_embeddings_path, "rb")
                )

        self.CBP = CompactBilinearPooling(self.dim, self.dim, self.dim)

        if margin == None or epsilon == None:
            nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
            nn.init.xavier_uniform_(self.norm_vector.weight.data)
        else:
            self.embedding_range = nn.Parameter(
                torch.Tensor([(self.margin + self.epsilon) / self.dim]),
                requires_grad=False,
            )
            nn.init.uniform_(
                tensor=self.ent_embeddings.weight.data,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item(),
            )
            nn.init.uniform_(
                tensor=self.rel_embeddings.weight.data,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item(),
            )
            nn.init.uniform_(
                tensor=self.norm_vector.weight.data,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item(),
            )

        if margin != None:
            self.margin = nn.Parameter(torch.Tensor([margin]))
            self.margin.requires_grad = False
            self.margin_flag = True
        else:
            self.margin_flag = False

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
                            self.whitespace_trans
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
                    else:
                        entity_name = self.entity_mapping[entity]["label"]
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

    def _calc(self, h, t, r, mode):
        if self.norm_flag:
            h = F.normalize(h, 2, -1)
            r = F.normalize(r, 2, -1)
            t = F.normalize(t, 2, -1)
        if mode != "normal":
            h = h.view(-1, r.shape[0], h.shape[-1])
            t = t.view(-1, r.shape[0], t.shape[-1])
            r = r.view(-1, r.shape[0], r.shape[-1])
        if mode == "head_batch":
            score = h + (r - t)
        else:
            score = (h + r) - t
        score = torch.norm(score, self.p_norm, -1).flatten()
        return score

    def _transfer(self, e, norm):
        norm = F.normalize(norm, p=2, dim=-1)
        if e.shape[0] != norm.shape[0]:
            e = e.view(-1, norm.shape[0], e.shape[-1])
            norm = norm.view(-1, norm.shape[0], norm.shape[-1])
            e = e - torch.sum(e * norm, -1, True) * norm
            return e.view(-1, e.shape[-1])
        else:
            return e - torch.sum(e * norm, -1, True) * norm

    def forward(self, data):
        batch_h = data["batch_h"]
        batch_t = data["batch_t"]
        batch_r = data["batch_r"]
        mode = data["mode"]
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        r_norm = self.norm_vector(batch_r)
        h = self._transfer(h, r_norm)
        t = self._transfer(t, r_norm)
        score = self._calc(h, t, r, mode)
        if self.margin_flag:
            return self.margin - score
        else:
            return score

    def regularization(self, data):
        batch_h = data["batch_h"]
        batch_t = data["batch_t"]
        batch_r = data["batch_r"]
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        r_norm = self.norm_vector(batch_r)
        regul = (
            torch.mean(h ** 2)
            + torch.mean(t ** 2)
            + torch.mean(r ** 2)
            + torch.mean(r_norm ** 2)
        ) / 4
        return regul

    def predict(self, data):
        score = self.forward(data)
        if self.margin_flag:
            score = self.margin - score
            return score.cpu().data.numpy()
        else:
            return score.cpu().data.numpy()
