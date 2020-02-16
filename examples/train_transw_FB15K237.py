import openke
from openke.config import Trainer, Tester
from openke.module.model import TransW
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

TRAIN = True
EPOCHS = 1

# dataloader for training
train_dataloader = TrainDataLoader(
    in_path="./benchmarks/FB15K237/",
    nbatches=100,
    threads=8,
    sampling_mode="normal",
    bern_flag=1,
    filter_flag=1,
    batch_size=1,
    neg_ent=25,
    neg_rel=0)

transw = TransW(
    ent_tot=train_dataloader.get_ent_tot(),
    rel_tot=train_dataloader.get_rel_tot(),
    dim=100,
    p_norm=1,
    word_embeddings_path="openke/embeddings/enwiki_20180420_100d.pkl",
    norm_flag=True)


test_dataloader = TestDataLoader("./benchmarks/FB15K237/", "link")

# define the loss function
model = NegativeSampling(
    model=transw,
    loss=MarginLoss(margin=5.0),
    batch_size=train_dataloader.get_batch_size()
)

# train the model

# test the model
test_dataloader = TestDataLoader("./benchmarks/FB15K237/", "link")
transw.load_checkpoint('./checkpoint/transw.ckpt')
tester = Tester(model=transw, data_loader=test_dataloader, use_gpu=False)
tester.run_link_prediction(type_constrain=False)

trainer = Trainer(model=model, data_loader=train_dataloader, train_times=EPOCHS, alpha=1.0,
                  use_gpu=False)

trainer.run()
transw.save_checkpoint('./checkpoint/transw.ckpt')



