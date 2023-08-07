import torch
from pytorch_lightning import LightningModule
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
import torch.optim as optim

from config import *
from utils import *


# Custom Resnet block only. Refer model.py file for the full model
class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Dropout(dropout_value_min)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Dropout(dropout_value_min)
        )

    def forward(self, x):
        out = self.block1(x)
        out = x + self.block2(out)
        return out


class CustomResNet(LightningModule):
    def __init__(self, data_dir=PATH_DATASETS, learning_rate=1e-3):
        super().__init__() 

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.learning_rate = learning_rate
        self.accuracy = Accuracy()

         # Prep Layer
        self.preplayer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Layer 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout_value_min),
        )
        
        # Residual Block 1
        self.res1 = ResBlock(128)
        
        # Layer 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(dropout_value_min),
        )
        
        # Layer 3
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(dropout_value_min),
        )

        # Residual Block 2
        self.res2 = ResBlock(512)
        
        # Maxpool
        self.maxpool = nn.MaxPool2d(4)

        # FC Layer
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.preplayer(x)
        
        out = self.layer1(out)
        r1 = self.res1(out)
        out = out + r1

        out = self.layer2(out)

        out = self.layer3(out)
        r2 = self.res2(out)
        out = out + r2
        
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = F.log_softmax(out, dim=1) # we need to use log(softmax), as the cross entropy needs the log function
        return out
    
    def training_step(self, batch, batch_idx):
        data, target = batch
        logits = self(data)
        loss = F.nll_loss(logits, target)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        logits = self(data)
        loss = F.nll_loss(logits, target)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, target)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        train_data = AlbumentationsCIFAR10Wrapper(root=self.data_dir, train=True, download=True)
        return DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)