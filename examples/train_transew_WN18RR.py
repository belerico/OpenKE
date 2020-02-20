import openke
from openke.config import Trainer, Tester
from openke.module.model import TransEW
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import torch

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
    neg_rel=0,
)

# define the model
transw = TransEW(
    ent_tot=train_dataloader.get_ent_tot(),
    rel_tot=train_dataloader.get_rel_tot(),
    dim=100,
    p_norm=1,
    norm_flag=True,
    entity_mapping="benchmarks/WN18RR/entity_mapping.json",
    entity2wiki_path=None,
    entity2id_path="benchmarks/WN18RR/entity2id.txt",
    relation_mapping="benchmarks/WN18RR/relation_mapping.json",
    relation2id_path="benchmarks/WN18RR/relation2id.txt",
    word_embeddings_path="embeddings/enwiki_20180420_100d.pkl",
)

transw.initialize_embeddings(entity_vector=False, merge="mean")

# define the loss function
model = NegativeSampling(
    model=transw,
    loss=MarginLoss(margin=5.0),
    batch_size=train_dataloader.get_batch_size(),
)

# train the model
trainer = Trainer(
    model=model,
    data_loader=train_dataloader,
    train_times=10,
    alpha=1.0,
    use_gpu=GPU,
    opt_method="Adam",
)

trainer.run()
transw.save_checkpoint("./checkpoint/transew.ckpt")

test_dataloader = TestDataLoader("./benchmarks/WN18RR/", "link")
# test the model
transw.load_checkpoint("./checkpoint/transew.ckpt")
tester = Tester(model=transw, data_loader=test_dataloader, use_gpu=GPU)
tester.run_link_prediction(type_constrain=False)
