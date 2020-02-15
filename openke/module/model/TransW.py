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
from wikipedia2vec import Wikipedia2Vec

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
        word_embeddings_path="embeddings/enwiki_20180420_100d.pkl",
        # unique_ent_terms=12025,
        # unique_rel_terms=446,
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
            data=numpy.loadtxt(entity2id_path, delimiter="\t", skiprows=1, dtype=str),
            columns=["entity", "id"],
        )
        self.relation2id = pd.DataFrame(
            data=numpy.loadtxt(relation2id_path, delimiter="\t", skiprows=1, dtype=str),
            columns=["relation", "id"],
        )
        # self.unique_ent_terms = unique_ent_terms
        # self.unique_rel_terms = unique_rel_terms
        self.We = nn.Embedding(len(self.entity_mapping), dim)
        self.We.weight.requires_grad = True
        self.Wr = nn.Embedding(len(self.relation_mapping), dim)
        self.Wr.weight.requires_grad = True

        if word_embeddings_path is None:
            raise Exception("The path for the word embeddings must be set")
        # self.word_embeddings = gensim.models.KeyedVectors.load_word2vec_format(word_embeddings_path)
        self.word_embeddings = Wikipedia2Vec.load(open(word_embeddings_path, "rb"))

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
            nn.init.uniform_(
                tensor=self.We.weight.data,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item(),
            )
            nn.init.uniform_(
                tensor=self.Wr.weight.data,
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
        # print(entity_id)
        mid_id = self.entity2id[self.entity2id["id"] == str(entity_id)][
            "entity"
        ].values[0]
        try:
            entities = self.entity2wiki[["label"]].loc[mid_id].values[0]
        except KeyError:
            entities = ""
        return list(set(entities.lower().translate(self.whitespace_trans).split()))

    def get_relation_terms(self, relation_id):
        relations_raw = self.relation2id[self.relation2id["id"] == str(relation_id)][
            "relation"
        ].values[0]
        relations_term = relations_raw.lower().translate(self.whitespace_trans).split()
        return list(set(relations_term))

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

        device = batches_h.device

        mode = data["mode"]
        scores = torch.zeros([n_samples]).to(device)

        for i in range(n_samples):

            batch_h = int(batches_h[i])
            batch_t = int(batches_t[i])
            batch_r = int(batches_r[i])

            # SUM W_i
            h = torch.zeros([1, self.dim]).to(device)
            t = torch.zeros([1, self.dim]).to(device)
            r = torch.zeros([1, self.dim]).to(device)

            terms_hi = self.get_entity_terms(batch_h)
            for term_hi in terms_hi:
                try:
                    w_hi = self.We(
                        torch.LongTensor(
                            [
                                self.entity_mapping[term_hi]
                            ]
                        ).to(device)
                    )
                    h_i = torch.FloatTensor(self.word_embeddings.get_word_vector(term_hi)).to(device)
                    h = h + torch.mul(h_i, w_hi)
                except KeyError:
                    continue
            if terms_hi != []:
                self.ent_embeddings.weight[batch_h] = h
            else:
                h = self.ent_embeddings(torch.LongTensor(batch_h))
            
            terms_ti = self.get_entity_terms(batch_t)
            for term_ti in terms_ti:
                try:
                    w_ti = self.We(
                        torch.LongTensor(
                            [
                                self.entity_mapping[term_ti]
                            ]
                        ).to(device)
                    )
                    h_t = torch.FloatTensor(self.word_embeddings.get_word_vector(term_ti)).to(device)
                    t = t + torch.mul(h_t, w_ti)
                except KeyError:
                    continue
            if terms_ti != []:
                self.ent_embeddings.weight[batch_t] = t
            else:
                print("PORCO DIO:", batch_t)
                print("PORCO DIO:", torch.LongTensor(batch_t))
                t = self.ent_embeddings(torch.LongTensor(batch_t))

            terms_ri = self.get_relation_terms(batch_r)
            for term_ri in terms_ri:
                try:
                    w_ri = self.Wr(
                        torch.LongTensor(
                            [
                                self.relation_mapping[term_ri]
                            ]
                        ).to(device)
                    )
                    h_r = torch.FloatTensor(self.word_embeddings.get_word_vector(term_ri)).to(device)
                    r = r + torch.mul(h_r, w_ri)
                except KeyError:
                    continue
            if terms_ri != []:
                self.rel_embeddings.weight[batch_r] = r
            else:
                r = self.rel_embeddings(torch.LongTensor(batch_r))

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
