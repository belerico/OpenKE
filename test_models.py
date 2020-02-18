import os
import torch
import openke
import argparse
from openke.config import Tester
from openke.module.model import TransE, TransEW, TransH, TransHW
from openke.data import TrainDataLoader, TestDataLoader

args = argparse.ArgumentParser(description="Test models")
args.add_argument(
    "-p", "--models-path", help="Path to models", action="store", type=str,
)
args = args.parse_args()

if __name__ == "__main__":
    GPU = torch.cuda.is_available()

    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path="./benchmarks/FB15K237/",
        nbatches=100,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0,
    )

    # define the model
    transe = TransE(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=100,
        p_norm=1,
        norm_flag=True,
    )

    # define the model
    transew = TransEW(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        laod_mappings=False,
        word_embeddings_path="openke/embeddings/enwiki_20180420_100d.pkl",
        dim=100,
        p_norm=1,
        norm_flag=True,
    )

    # define the model
    transh = TransH(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=100,
        p_norm=1,
        norm_flag=True,
    )

    # define the model
    transhw = TransHW(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        laod_mappings=False,
        word_embeddings_path="openke/embeddings/enwiki_20180420_100d.pkl",
        dim=100,
        p_norm=1,
        norm_flag=True,
    )
    models = {
        "transe": transe,
        "transew": transew,
        "transh": transh,
        "transhw": transhw,
    }
    # test the model
    test_dataloader = TestDataLoader("./benchmarks/FB15K237/", "link")
    ckpts = os.listdir(args.models_path)
    for ckpt in ckpts:
        filename = os.path.basename(ckpt).split(".")[0]
        model = filename.split("_")[0]
        models[model].load_checkpoint(ckpt)
        tester = Tester(model=models[model], data_loader=test_dataloader, use_gpu=GPU)
        print("* TESTING " + model)
        tester.run_link_prediction(type_constrain=False)
