import os, sys
import copy
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import IPython

import numpy as np
import pytorch_lightning as pl
import torchmetrics
import wandb

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Local Imports
from datasets import GeneralDataset, GeneralDatasetPreloaded

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"

print(torch.__version__)
print(torchaudio.__version__)

torch.random.manual_seed(2667)

print("\n\nUPGRADE TO python3.10 ASAP\n\n")


class LMSystem(pl.LightningModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        
        # Model
        bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH
        self.tacotron2 = bundle.get_tacotron2()
        breakpoint()
        #self.tacotron2.requires_grad = False
        self.vocoder = bundle.get_vocoder()
        self.vocoder.requires_grad = False

        print("\n\nUNFREEZE THE NETWORK\n\n")

        # Criterion and metrics
        self.criterion = nn.MSELoss(reduction="mean")
        # raise NotImplementedError("Sort the metrics and loss")
        # self.criterion = nn.NLLLoss()
        # self.acc = torchmetrics.Accuracy()
        # self.f1 = torchmetrics.F1()
        # self.recall = torchmetrics.Recall()
        # self.prec = torchmetrics.Precision()


    def forward(self, inputs, lengths):
        specgram, lengths, _ = self.tacotron2.infer(inputs[0], lengths[0])
        return specgram, lengths

    def configure_optimizers(self):
        lr = 10**(self.args.lr)
        optimizer_name = self.args.optimiser # ["Adam", "RMSprop", "SGD"])
        optimizer = getattr(torch.optim, optimizer_name)(self.parameters(), lr=lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        inputs = train_batch['inputs']
        lengths = train_batch['lengths']
        breakpoint()
        specgram, lengths = self(inputs, lengths)
        #specgram = specgram.cpu()
        #lengths = lengths.cpu()
        #waveforms, lengths = self.vocoder(specgram, lengths)
        #torchaudio.save(os.path.join(".results", f"un_trained-{batch_idx}-tts.wav"), waveforms, self.vocoder.sample_rate)
        return train_loss

    def validation_step(self, valid_batch, batch_idx):
        pass
        #torchaudio.save("output_waveglow.wav", waveforms[0:1].cpu(), sample_rate=22050)
        #IPython.display.display(IPython.display.Audio("output_waveglow.wav"))
        #inputs = valid_batch['inputs']
        #gt = valid_batch['gt']
        #out = self(inputs)
        #valid_loss = self.criterion(inputs, gt)
        #self.log(f"valid_loss", valid_loss, prog_bar=True, on_step=False, on_epoch=True)
        #self.log(f"valid_acc", self.acc(out, gt), prog_bar=False, on_step=False, on_epoch=True)
        #self.log(f"valid_f1", self.f1(out, gt), on_step=False, on_epoch=True)
        #self.log(f"valid_recall", self.recall(out, gt), on_step=False, on_epoch=True)
        #self.log(f"valid_prec", self.prec(out, gt), on_step=False, on_epoch=True)
        #valid_loss = 0
        #return valid_loss

    #def test_step(self, test_batch, batch_idx):
    #    inputs = test_batch['inputs']
    #    gt = test_batch['gt']
    #    out = self(inputs)
    #    test_loss = self.criterion(inputs, gt)
    #    self.log(f"test_loss", test_loss, prog_bar=True, on_step=False, on_epoch=True)
    #    self.log(f"test_acc", self.acc(out, gt), prog_bar=False, on_step=False, on_epoch=True)
    #    self.log(f"test_f1", self.f1(out, gt), on_step=False, on_epoch=True)
    #    self.log(f"test_recall", self.recall(out, gt), on_step=False, on_epoch=True)
    #    self.log(f"test_prec", self.prec(out, gt), on_step=False, on_epoch=True)
    #    return test_loss

    #def test_epoch_end(self, test_step_outputs):
    #    print("Add ending of test code here?")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", default=0, type=int, help="number of pytorch workers for dataloader")
    parser.add_argument("--device", default=0, type=int, help="Cuda device, -1 for CPU")
    parser.add_argument("--bsz", default=32, type=int, help="Training batch size")
    parser.add_argument("--vt_bsz", default=32, type=int, help="Validation and test batch size")
    parser.add_argument("--dset_seed", type=int, default=2667, help="Random seed to make the dataset and fc_intermediate consistent")
    parser.add_argument("--optimiser", default="Adam", type=str, choices=["Adam", "RMSprop", "SGD"], help="Optimiser kind to use")
    parser.add_argument("--lr", default=-5, type=float, help="Exponent of the learning rate")
    parser.add_argument("--epochs", default=1, type=int, help="Epochs to run for")
    parser.add_argument("--shuffle", action="store_true", default=False, help="Shuffle the dataset")
    parser.add_argument("--wandb", action="store_true", help="To enable online logging for wandb")
    parser.add_argument("--preload", action="store_true", help="Preload dataset. Run this when dataset loading is a bottleneck")
    parser.add_argument("--dropout", type=float, default=0.2, help="FC dropout")

    args = parser.parse_args()
    print(args)

    # WANDB
    if not args.wandb:
        os.environ["WANDB_MODE"] = "offline"
    runname = "CHANGEME"
    wandb.init(entity="jumperkables", project="ikit_alexa", name=runname, config=args)
    config = wandb.config
    wandb_logger = pl.loggers.WandbLogger()
    wandb_logger.log_hyperparams(config)#(args)

    # Datasets
    train_dset = None
    valid_dset = None
    #test_dset = None
    # pin memory for speed
    #pin_memory = (args.device >= 0) and (args.num_workers >= 1)
    if args.preload:
        data = GeneralDatasetPreloaded(args)
    else:
        data = GeneralDataset(args)
    valid_data = copy.deepcopy(data)
    valid_data.choose_split("valid")
    #test_data = copy.deepcopy(data)
    #test_data.choose_split("test")
    train_data = data
    train_data.choose_split("train")
    train_loader = DataLoader(train_data, num_workers=args.num_workers, batch_size=args.bsz)
    valid_loader = DataLoader(valid_data, num_workers=args.num_workers, batch_size=args.vt_bsz)
    #test_loader = DataLoader(test_data, num_workers=args.num_workers, batch_size=args.vt_bsz)#args.vt_bsz)

    # GPU
    if args.device == -1:
        gpus = None 
    else: 
        gpus = [args.device]

    ## Checkpoint callbacks
    #checkpoint_callback = pl.callbacks.ModelCheckpoint(
    #    monitor=f"valid_loss",
    #    dirpath=os.path.join(os.path.dirname(os.path.realpath(__file__)), ".results"),
    #    filename=f"{runname}"+'-{epoch:02d}',
    #    save_top_k=1,
    #    mode="min"
    #)
    #callbacks = [checkpoint_callback]

    # Running
    pl_system = LMSystem(args)
    trainer = pl.Trainer(logger=wandb_logger, gpus=gpus, max_epochs=args.epochs)#, callbacks=callbacks)
    trainer.fit(pl_system, train_loader, valid_loader)
    #trainer.test(model=pl_system, test_dataloaders=test_loader)
