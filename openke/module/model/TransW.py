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


class TransW(Model):
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
        word_embeddings_path="embeddings/enwiki_20180420_win10_100d.txt",
        unique_ent_terms=12025,
        unique_rel_terms=446,
    ):
        super(TransW, self).__init__(ent_tot, rel_tot)

        self.dim = dim
        self.margin = margin
        self.epsilon = epsilon
        self.norm_flag = norm_flag
        self.p_norm = p_norm

        self.whitespace_trans = str.maketrans(
            string.punctuation, " " * len(string.punctuation)
        )

        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)

        self.entity_mapping = json.load(open(entity_mapping, "rb"))
        self.relation_mapping = json.load(open(relation_mapping, "rb"))

        self.entity2wiki = pd.read_json(entity2wiki_path, orient="index")
        self.entity2id = pd.DataFrame(
            data=numpy.loadtxt(
                entity2id_path, delimiter="\t", skiprows=1, dtype=object
            ),
            columns=["entity", "id"],
        )
        self.relation2id = pd.DataFrame(
            data=numpy.loadtxt(
                relation2id_path, delimiter="\t", skiprows=1, dtype=object
            ),
            columns=["relation", "id"],
        )
        self.unique_ent_terms = unique_ent_terms
        self.unique_rel_terms = unique_rel_terms
        self.We = nn.Embedding(self.unique_ent_terms, dim)
        self.Wr = nn.Embedding(self.unique_rel_terms, dim)

        if word_embeddings_path is None:
            raise Exception("The path for the word embeddings must be set")
        model = gensim.models.KeyedVectors.load_word2vec_format(
            word_embeddings_path, limit=10000
        )
        self.word_embeddings = nn.Embedding.from_pretrained(
            torch.FloatTensor(model.vectors), freeze=True,
        )
        self.word2index = dict(
            zip(model.index2word, list(range(len(model.index2word))))
        )

        del model

        if margin == None or epsilon == None:
            nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
            nn.init.xavier_uniform_(self.We.weight.data)
            nn.init.xavier_uniform_(self.Wr.weight.data)
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

    def get_entity_terms(self, entity_id):
        mid_id = self.entity2id[self.entity2id["id"] == str(entity_id)][
            "entity"
        ].values[0]
        entities = self.entity2wiki[self.entity2wiki.index == mid_id]["label"].values[0]
        return entities.lower().translate(self.whitespace_trans).split()

    def get_relation_terms(self, relation_id):
        relations_raw = self.relation2id[self.relation2id["id"] == str(relation_id)][
            "relation"
        ].values[0]
        relations_term = relations_raw.lower().translate(self.whitespace_trans).split()
        return relations_term

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
        batches_h = data["batch_h"]
        batches_t = data["batch_t"]
        batches_r = data["batch_r"]
        n_samples = len(batches_h)

        mode = data["mode"]
        scores = torch.zeros([n_samples])

        for i in range(n_samples):

            batch_h = int(batches_h[i])
            batch_t = int(batches_t[i])
            batch_r = int(batches_r[i])

            # SUM W_i
            h = torch.zeros([1, self.dim])
            t = torch.zeros([1, self.dim])
            r = torch.zeros([1, self.dim])

            for term_hi in self.get_entity_terms(batch_h):
                term_id = torch.LongTensor(
                    [self.word2index[term_hi] if term_hi in self.word2index else 0]
                )
                # TODO: a entity terms mapping is needed
                w_hi = self.We(term_id)
                h_i = self.word_embeddings(term_id)

                h = h + torch.mul(h_i, w_hi)

            for term_ti in self.get_entity_terms(batch_t):
                term_id = torch.LongTensor(
                    [self.word2index[term_ti] if term_ti in self.word2index else 0]
                )
                w_ti = self.We(term_id)
                h_t = self.word_embeddings(term_id)

                t = t + torch.mul(h_t, w_ti)

            for term_ri in self.get_relation_terms(batch_r):
                term_id = torch.LongTensor(
                    [self.word2index[term_ri] if term_ri in self.word2index else 0]
                )
                try:
                    # TODO: a relation terms mapping is needed
                    w_ri = self.Wr(term_id)
                    h_r = self.word_embeddings(term_id)
                    r = r + torch.mul(h_r, w_ri)
                except:
                    print("Not found")

            score = self._calc(h, t, r, mode)
            if self.margin_flag:
                score = self.margin - score

            scores[i] = score

        return scores


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
