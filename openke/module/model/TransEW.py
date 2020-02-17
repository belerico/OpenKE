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


class TransEW(Model):
    def __init__(
            self,
            ent_tot,
            rel_tot,
            dim=100,
            p_norm=1,
            norm_flag=True,
            margin=None,
            epsilon=None,
            entity_mapping="benchmarks/FB15K237/entity_mapping.json",
            entity2wiki_path="benchmarks/FB15K237/entity2wikidata.json",
            entity2id_path="benchmarks/FB15K237/entity2id.txt",
            relation_mapping="benchmarks/FB15K237/relation_mapping.json",
            relation2id_path="benchmarks/FB15K237/relation2id.txt",
            word_embeddings_path="embeddings/enwiki_20180420_100d.pkl",
    ):
        super(TransEW, self).__init__(ent_tot, rel_tot)

        self.dim = dim
        self.margin = margin
        self.epsilon = epsilon
        self.norm_flag = norm_flag
        self.p_norm = p_norm

        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)

        self.whitespace_trans = str.maketrans(
            string.punctuation, " " * len(string.punctuation)
        )

        self.entity_mapping = json.load(open(entity_mapping, "rb"))
        self.relation_mapping = json.load(open(relation_mapping, "rb"))
        if entity2wiki_path:
            self.entity2wiki = pd.read_json(entity2wiki_path, orient="index")
            self.entity2wiki = self.entity2wiki[~self.entity2wiki["wikipedia"].isnull()]
        else:
            self.entity2wiki = None

        self.entity2id = pd.DataFrame(
            data=numpy.loadtxt(entity2id_path, delimiter="\t", skiprows=1, dtype=str),
            columns=["entity", "id"],
        )
        self.relation2id = pd.DataFrame(
            data=numpy.loadtxt(relation2id_path, delimiter="\t", skiprows=1, dtype=str),
            columns=["relation", "id"],
        )
        if word_embeddings_path is None:
            raise Exception("The path for the word embeddings must be set")
        if word_embeddings_path.split("/")[-1] == ".txt":
            self.word_embeddings = gensim.models.KeyedVectors.load_word2vec_format(
                word_embeddings_path
            )
        else:
            self.word_embeddings = Wikipedia2Vec.load(open(word_embeddings_path, "rb"))

        self.CBP = CompactBilinearPooling(self.dim, self.dim, self.dim)

        if margin == None or epsilon == None:
            nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
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

        if margin != None:
            self.margin = nn.Parameter(torch.Tensor([margin]))
            self.margin.requires_grad = False
            self.margin_flag = True
        else:
            self.margin_flag = False

    def initialize_embeddings(self, entity_vector=True, merge="sum"):
        if entity_vector:
            for entity, idx in self.entity2id.itertuples(index=False, name=None):
                try:
                    if self.entity2wiki:
                        entity_url = self.entity2wiki[["wikipedia"]].loc[entity].values[0]
                        entity_name = os.path.basename(entity_url)
                        self.ent_embeddings.weight.data[int(idx)] = torch.Tensor(
                            self.word_embeddings.get_entity_vector(
                                entity_name.replace("_", " ")
                            )
                        ).data
                    else:
                        pass
                # TODO: use mapping file for "label" of each entity
                except KeyError:
                    continue
        else:
            for entity, idx in self.entity2id.itertuples(index=False, name=None):
                try:
                    entity_url = self.entity2wiki[["wikipedia"]].loc[entity].values[0]
                    entity_name = os.path.basename(entity_url)
                    terms = list(
                        set(
                            entity_name.lower().translate(self.whitespace_trans).split()
                        )
                    )
                    if merge == "cbp":
                        terms_indices = []
                        for k, term in enumerate(terms):
                            try:
                                self.word_embeddings.get_word_vector(term)
                                terms_indices.append(k)
                            except KeyError:
                                continue
                        if len(terms_indices) >= 1:
                            self.ent_embeddings.weight.data[int(idx)] = torch.Tensor(
                                self.word_embeddings.get_word_vector(
                                    terms[terms_indices[0]]
                                )
                            ).data
                            for k in range(1, len(terms_indices)):
                                self.ent_embeddings.weight.data[int(idx)] = self.CBP(
                                    torch.Tensor(
                                        self.word_embeddings.get_word_vector(
                                            terms[terms_indices[k]]
                                        )
                                    ).data,
                                    self.ent_embeddings.weight.data[int(idx)],
                                )
                    else:
                        term_found = False
                        old_init = self.ent_embeddings.weight.data[int(idx)]
                        if merge == "sum":
                            self.ent_embeddings.weight.data[int(idx)] = torch.zeros(
                                [1, self.dim]
                            ).data
                        elif merge == "hadamard":
                            self.ent_embeddings.weight.data[int(idx)] = torch.ones(
                                [1, self.dim]
                            ).data
                        for term in terms:
                            try:
                                if merge == "sum":
                                    self.ent_embeddings.weight.data[
                                        int(idx)
                                    ] += torch.Tensor(
                                        self.word_embeddings.get_word_vector(term)
                                    ).data
                                elif merge == "hadamard":
                                    self.ent_embeddings.weight.data[int(idx)] = torch.mul(
                                        torch.Tensor(
                                            self.word_embeddings.get_word_vector(term)
                                        ).data,
                                        self.ent_embeddings.weight.data[int(idx)],
                                    )
                                if not term_found:
                                    term_found = True
                            except KeyError:
                                continue
                        if not term_found:
                            self.rel_embeddings.weight.data[int(idx)] = old_init
                except KeyError:
                    continue

        for relation, idx in self.relation2id.itertuples(index=False, name=None):
            try:
                terms = list(
                    set(relation.lower().translate(self.whitespace_trans).split())
                )
                if merge == "cbp":
                    terms_indices = []
                    for k, term in enumerate(terms):
                        try:
                            self.word_embeddings.get_word_vector(term)
                            terms_indices.append(k)
                        except KeyError:
                            continue
                    if len(terms_indices) >= 1:
                        self.rel_embeddings.weight.data[int(idx)] = torch.Tensor(
                            self.word_embeddings.get_word_vector(
                                terms[terms_indices[0]]
                            )
                        ).data
                        for k in range(1, len(terms_indices)):
                            self.rel_embeddings.weight.data[int(idx)] = self.CBP(
                                torch.Tensor(
                                    self.word_embeddings.get_word_vector(
                                        terms[terms_indices[k]]
                                    )
                                ).data,
                                self.rel_embeddings.weight.data[int(idx)],
                            )
                else:
                    term_found = False
                    old_init = self.rel_embeddings.weight.data[int(idx)]
                    if merge == "sum":
                        self.rel_embeddings.weight.data[int(idx)] = torch.zeros(
                            [1, self.dim]
                        ).data
                    elif merge == "hadamard":
                        self.rel_embeddings.weight.data[int(idx)] = torch.ones(
                            [1, self.dim]
                        ).data
                    for term in terms:
                        try:
                            if merge == "sum":
                                self.rel_embeddings.weight.data[
                                    int(idx)
                                ] += torch.Tensor(
                                    self.word_embeddings.get_word_vector(term)
                                ).data
                            elif merge == "hadamard":
                                self.rel_embeddings.weight.data[int(idx)] = torch.mul(
                                    torch.Tensor(
                                        self.word_embeddings.get_word_vector(term)
                                    ).data,
                                    self.rel_embeddings.weight.data[int(idx)],
                                )
                            if not term_found:
                                term_found = True
                        except KeyError:
                            continue
                    if not term_found:
                        self.rel_embeddings.weight.data[int(idx)] = old_init
            except KeyError:
                continue

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

    def forward(self, data):
        batch_h = data["batch_h"]
        batch_t = data["batch_t"]
        batch_r = data["batch_r"]
        mode = data["mode"]
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
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
        regul = (torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)) / 3
        return regul

    def predict(self, data):
        score = self.forward(data)
        if self.margin_flag:
            score = self.margin - score
            return score.cpu().data.numpy()
        else:
            return score.cpu().data.numpy()
