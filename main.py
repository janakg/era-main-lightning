from __future__ import print_function

import torch
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' 

from torch_lr_finder import LRFinder
from torchsummary import summary
from pytorch_lightning import pl


# UTIL AND CONFIG IMPORTS
from utils import *
from config import *

# model imported from a module
from models.custom_resnet import CustomResNet

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' 

def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train(model, device, train_loader, optimizer, criterion, scheduler):
  model.train()
  pbar = tqdm(train_loader)
  lr_values = []

  train_loss = 0
  train_succeeded = 0
  train_processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    # Predict
    pred = model(data)

    # Calculate loss
    loss = criterion(pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()
    scheduler.step()
    lr_values.append(scheduler.get_last_lr()[0])

    train_succeeded += GetCorrectPredCount(pred, target)
    train_processed += len(data)

    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*train_succeeded/train_processed:0.2f} LR={scheduler.get_last_lr()[0]}')
  
  return train_succeeded, train_processed, train_loss, lr_values

def test(model, device, test_loader, criterion):
    model.eval()

    test_loss = 0
    test_succeeded = 0
    batch_count = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            batch_count += 1
            test_succeeded += GetCorrectPredCount(output, target)


    test_loss = test_loss / batch_count 
    
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, test_succeeded, len(test_loader.dataset),
        100. * test_succeeded / len(test_loader.dataset)))
    
    return test_succeeded, test_loss

def get_lr(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    end_lr=1,
    num_iter=100,
    step_mode="exp",
    start_lr=None,
    diverge_th=5,
):
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(
        train_loader,
        end_lr=end_lr,
        num_iter=num_iter,
        step_mode=step_mode,
        start_lr=start_lr,
        diverge_th=diverge_th,
    )
    _, max_lr = lr_finder.plot(log_lr=False, suggest_lr=True)
    print("max_lr", max_lr)

    # Reset the model and optimizer to initial state
    lr_finder.reset()

    return max_lr


def init():
    # CUDA?
    use_cuda = torch.cuda.is_available()
    print("CUDA Available?", use_cuda)

    # For reproducibility. SEED Random functions
    SEED = 1
    torch.manual_seed(SEED)
    if use_cuda:
        torch.cuda.manual_seed(SEED)

    device = torch.device("cuda" if use_cuda else "cpu")

    dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True) if use_cuda else dict(shuffle=True, batch_size=64)

    train_data, test_data = get_train_and_test_data()
    
    # train dataloader
    train_loader = torch.utils.data.DataLoader(train_data, **dataloader_args)

    # test dataloader
    test_loader = torch.utils.data.DataLoader(test_data, **dataloader_args)

    model = CustomResNet().to(device)
    summary(model, input_size=(3, 32, 32))

    # Set the hook
    # model.layer1.register_forward_hook(print_featuremaps_hook)

    return device, train_loader, test_loader, model


def run(device, train_loader, test_loader, model):
    # Data to plot accuracy and loss graphs
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []

    model = CustomResNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-2)  # you can adjust learning rate as needed
    criterion = nn.CrossEntropyLoss() # reduction='none' // it can be sum also

    # LRMAX = get_lr(
    #         model,
    #         train_loader,
    #         optimizer,
    #         criterion,
    #         device)

    LRMAX = 2.54E-04
    print("LRMAX:", LRMAX)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                            max_lr=LRMAX,
                                            steps_per_epoch=len(train_loader),
                                            epochs=num_epochs,
                                            pct_start=max_lr_epoch/num_epochs,
                                            div_factor=1,
                                            final_div_factor=10,
                                            three_phase=False
                                    )


    for epoch in range(1, num_epochs+1):
        print(f'Epoch {epoch}')
        train_succeeded, train_processed, train_loss, lr_values = train(model, device, train_loader, optimizer, criterion, scheduler)
        train_acc.append(100 * train_succeeded/train_processed)
        train_losses.append(train_loss / len(train_loader))

        test_succeeded, test_loss = test(model, device, test_loader, criterion)
        test_acc.append(100. * test_succeeded / len(test_loader.dataset))
        test_losses.append(test_loss)

    draw_training_loss(train_losses, train_acc, test_losses, test_acc)
    
    return train_losses, test_losses, train_acc, test_acc, lr_values


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()


def run_lightning():
    # Initialize the model
    model = CustomResNet()

    # Initialize the PyTorch Lightning trainer and train the model
    trainer = pl.Trainer(
        gpus = AVAIL_GPUS,
        max_epochs = 3,
        progress_bar_refresh_rate=10
    )
    trainer.fit(model)