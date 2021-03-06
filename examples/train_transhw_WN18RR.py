import openke
from openke.config import Trainer, Tester
from openke.module.model import TransHW
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import torch


def main(epochs, lr, vec, merge):
    name = f"transew_{merge}_{'entvec' if vec else 'novec'}_{merge}_{epochs}ep_{lr}lr.ckpt"
    GPU = torch.cuda.is_available()
    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path="./benchmarks/WN18RR/",
        nbatches=100,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0)

    # define the model
    transw = TransHW(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=100,
        entity_mapping="./benchmarks/WN18RR/mapping.json",
        entity2id_path="./benchmarks/WN18RR/entity2id.txt",
        relation_mapping="./benchmarks/WN18RR/relation_mapping.json",
        entity2wiki_path=None,
        relation2id_path="benchmarks/WN18RR/relation2id.txt",
        word_embeddings_path="openke/embeddings/enwiki_20180420_100d.pkl",
        p_norm=1,
        norm_flag=True)

    transw.initialize_embeddings(entity_vector=vec, merge=merge)

    # define the loss function
    model = NegativeSampling(
        model=transw,
        loss=MarginLoss(margin=5.0),
        batch_size=train_dataloader.get_batch_size()
    )

    # train the model
    trainer = Trainer(model=model, data_loader=train_dataloader, train_times=epochs, alpha=lr,
                      use_gpu=GPU)

    trainer.run()
    transw.save_checkpoint(f'./checkpoint/{name}')

    test_dataloader = TestDataLoader("./benchmarks/WN18RR/", "link")
    # test the model
    transw.load_checkpoint(f'./checkpoint/{name}')
    tester = Tester(model=transw, data_loader=test_dataloader, use_gpu=GPU)
    tester.run_link_prediction(type_constrain=False)
