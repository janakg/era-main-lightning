from __future__ import print_function

import torch
import pytorch_lightning as pl

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' 

# UTIL AND CONFIG IMPORTS
from utils import *
from config import *

# model imported from a module
from models.custom_resnet import CustomResNet

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' 


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()


def run_lightning_trainer():
    # Initialize the model
    model = CustomResNet()

    # Initialize the PyTorch Lightning trainer and train the model
    trainer = pl.Trainer(
        accelerator='gpu', 
        devices=AVAIL_GPUS,
        max_epochs = 3
    )
    trainer.fit(model)

    return model

def draw_misclassified_images(model):
    draw_misclassified_images_util(model, model.test_dataloader(), model.device)

def draw_misclassified_with_gradcam_images(model):
    draw_misclassified_with_gradcam_images_util(model, model.test_dataloader(), target_layers=[model.layer3[-1]], device=model.device)
