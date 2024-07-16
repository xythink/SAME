import torch
from torch import log_softmax
import torch.nn as nn
import time
from typing import Tuple
import torch
from torch.nn import Module
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from tqdm import tqdm
from .misc import get_device

import torch.nn.functional as F


def train_epoch(
    model: Module,
    data_loader: DataLoader,
    opt: Optimizer,
    criterion: _Loss,
    disable_pbar: bool = False,
) -> Tuple[float, float]:
    """
    Train for 1 epoch
    """
    device = get_device()
    model = model.to(device)
    model.train()
    running_loss = correct = 0.0
    n_batches = len(data_loader)
    for (x, y) in tqdm(data_loader, ncols=80, disable=disable_pbar, leave=False):
        # if y.shape[0] < 128:
        #    continue

        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        opt.step()
        running_loss += loss.item()
        pred_class = torch.argmax(pred, dim=-1)
        if y.ndim == 2:
            y = torch.argmax(y, dim=-1)
        correct += (pred_class == y).sum().item()

    loss = running_loss / n_batches
    acc = correct / len(data_loader.dataset)
    return loss, acc

def train_epoch_with_generator(
    model: Module,
    generator: Module,
    data_loader: DataLoader,
    opt: Optimizer,
    criterion: _Loss,
    disable_pbar: bool = False,
):
    device = get_device()
    model = model.to(device)
    model.train()
    running_loss = correct = 0.0
    n_batches = len(data_loader)
    for (x, y) in tqdm(data_loader, ncols=80, disable=disable_pbar, leave=False):
        # if y.shape[0] < 128:
        #    continue

        x, y = x.to(device), y.to(device)
        x_new = generator(x)
        if isinstance(x_new, tuple):
            x_new = x_new[0].detach()
        else:
            x_new = x_new.detach()
        opt.zero_grad()
        pred = model(x_new)
        loss = criterion(pred, y)
        loss.backward()
        opt.step()
        running_loss += loss.item()
        pred_class = torch.argmax(pred, dim=-1)
        if y.ndim == 2:
            y = torch.argmax(y, dim=-1)
        correct += (pred_class == y).sum().item()

    loss = running_loss / n_batches
    acc = correct / len(data_loader.dataset)
    return loss, acc

def train_epoch_with_two_sets(
    model: Module,
    data_loader1: DataLoader,
    data_loader2: DataLoader,
    opt: Optimizer,
    criterion1: _Loss,
    criterion2: _Loss,
    disable_pbar: bool=False
):
    """
    Train for 1 epoch
    """
    device = get_device()
    model = model.to(device)
    model.train()
    running_loss = correct = 0.0
    ood_loss = 0.0
    n_batches = len(data_loader1)
    # assert len(data_loader1) <= len(data_loader2), "len(data_loader1) should <= len(data_loader2)"

    iter_dataloader2 = iter(data_loader2)

    for (x, y) in tqdm(data_loader1, ncols=80, disable=disable_pbar, leave=False):
        # if y.shape[0] < 128:
        #    continue
        # train one step for dataloader1
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        pred = model(x)
        loss = criterion1(pred, y)
        loss.backward()
        opt.step()
        running_loss += loss.item()
        pred_class = torch.argmax(pred, dim=-1)
        if y.ndim == 2:
            y = torch.argmax(y, dim=-1)
        correct += (pred_class == y).sum().item()

        # train one step for dataloader2
        try:
            x, y = next(iter_dataloader2)
        except:
            iter_dataloader2 = iter(data_loader2)
            x, y = next(iter_dataloader2)
        x, y = x.to(device), y.to(device) # y.size() = (batch_size,) or (batch_size, 1)
        opt.zero_grad()
        pred = model(x)
        pred_logsoftmax = F.log_softmax(pred, dim=-1)
        target = torch.ones_like(pred)/pred.size(-1)
        loss = criterion2(pred_logsoftmax, target)
        loss.backward()
        opt.step()

        ood_loss += loss.item()


    train_loss = running_loss / n_batches
    ood_loss = ood_loss / n_batches
    acc = correct / len(data_loader1.dataset)
    return train_loss, ood_loss, acc

def train_epoch_with_two_sets_shadow(
    model: Module,
    data_loader1: DataLoader,
    data_loader2: DataLoader,
    opt: Optimizer,
    criterion1: _Loss,
    criterion2: _Loss,
    disable_pbar: bool=False
):
    """
    This function is used to train model (with shadow class/node) on normal and ood dataset.
    """
    device = get_device()
    model = model.to(device)
    model.train()
    running_loss = correct = 0.0
    ood_loss = 0.0
    n_batches = len(data_loader1)
    assert len(data_loader1) <= len(data_loader2), "len(data_loader1) should <= len(data_loader2)"

    iter_dataloader2 = iter(data_loader2)

    for (x, y) in tqdm(data_loader1, ncols=80, disable=disable_pbar, leave=False):
        # if y.shape[0] < 128:
        #    continue
        # train one step for dataloader1
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        pred = model(x)
        loss = criterion1(pred, y)
        loss.backward()
        opt.step()
        running_loss += loss.item()
        pred_class = torch.argmax(pred, dim=-1)
        if y.ndim == 2:
            y = torch.argmax(y, dim=-1)
        correct += (pred_class == y).sum().item()

        # train one step for dataloader2
        x, y = next(iter_dataloader2)
        x, y  = x.to(device), y.to(device)
        opt.zero_grad()
        pred = model(x)
        
        target = torch.ones_like(y) * (pred.size()[-1] - 1)
        loss = criterion2(pred, target)
        loss.backward()
        opt.step()
        ood_loss += loss.item()


    train_loss = running_loss / n_batches
    ood_loss = ood_loss / n_batches
    acc = correct / len(data_loader1.dataset)
    return train_loss, ood_loss, acc




def test(model, data_loader):
    """
    test accuracy
    """
    device = get_device()
    model = model.to(device)
    model.eval()
    correct = 0.0
    with torch.no_grad():
        for (x, y) in data_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            pred_class = torch.argmax(pred, dim=1)
            correct += (pred_class == y).sum().item()
        acc = correct / len(data_loader.dataset)
    return acc

def test_with_generator(model, generator, data_loader):
    """
    test accuracy with generator
    """
    device = get_device()
    model = model.to(device)
    model.eval()
    correct = 0.0
    with torch.no_grad():
        for (x, y) in data_loader:
            x, y = x.to(device), y.to(device)
            x_new = generator(x)
            if isinstance(x_new, tuple):
                x_new = x_new[0]
            pred = model(x_new)
            pred_class = torch.argmax(pred, dim=1)
            correct += (pred_class == y).sum().item()
        acc = correct / len(data_loader.dataset)
    return acc

def train(
    model: Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: _Loss,
    opt: Optimizer,
    epochs: int = 10,
    sch: _LRScheduler = None,
    disable_pbar=False,
):
    """
    Train model
    """
    for epoch in range(1, epochs + 1):
        s = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, opt, criterion, disable_pbar
        )
        test_acc = test(model, test_loader)
        if sch:
            sch.step()
        e = time.time()
        time_epoch = e - s
        print(
            "Epoch: {} train_loss: {:.3f} train_acc: {:.2f}%, test_acc: {:.2f}% time: {:.1f}".format(
                epoch, train_loss, train_acc * 100, test_acc * 100, time_epoch
            )
        )

def train_with_generator(
    model: Module,
    generator: Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: _Loss,
    opt: Optimizer,
    epochs: int = 10,
    sch: _LRScheduler = None,
    disable_pbar=False,
):
    """
    Train model with synthetic images
    """
    for epoch in range(1, epochs + 1):
        s = time.time()
        train_loss, train_acc = train_epoch_with_generator(
            model, generator, train_loader, opt, criterion, disable_pbar
        )
        test_acc = test_with_generator(model, generator, test_loader)
        if sch:
            sch.step()
        e = time.time()
        time_epoch = e - s
        print(
            "Epoch: {} train_loss: {:.3f} train_acc: {:.2f}%, test_acc: {:.2f}% time: {:.1f}".format(
                epoch, train_loss, train_acc * 100, test_acc * 100, time_epoch
            )
        )
    


def train_with_ood(
    model: Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    ood_loader: DataLoader,
    criterion_hard: _Loss,
    criterion_soft: _Loss,
    opt: Optimizer,
    epochs: int = 10,
    sch: _LRScheduler = None,
    disable_pbar=False,
):
    """
    Train model with normal dataset and ood dataset
    """
    for epoch in range(1, epochs + 1):
        s = time.time()
        train_loss, ood_loss, train_acc = train_epoch_with_two_sets(
            model, train_loader, ood_loader, opt, criterion_hard, criterion_soft, disable_pbar
        )
        test_acc = test(model, test_loader)
        if sch:
            sch.step()
        e = time.time()
        time_epoch = e - s
        print(
            "Epoch: {} train_loss: {:.3f} train_acc: {:.2f}%, test_acc: {:.2f}%, ood_loss: {:.3f}, time: {:.1f}".format(
                epoch, train_loss, train_acc * 100, test_acc * 100, ood_loss, time_epoch
            )
        )


def train_with_shadow_ood(
    model: Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    ood_loader: DataLoader,
    criterion_norm: _Loss,
    criterion_ood: _Loss,
    opt: Optimizer,
    epochs: int = 10,
    sch: _LRScheduler = None,
    disable_pbar=False  
):
    """
    Train shadow node model with normal dataset and ood dataset.
    """
    for epoch in range(1, epochs + 1):
        s = time.time()
        train_loss, ood_loss, train_acc = train_epoch_with_two_sets_shadow(
            model, train_loader, ood_loader, opt, criterion_norm, criterion_ood, disable_pbar
        )
        test_acc = test(model, test_loader)
        if sch:
            sch.step()
        e = time.time()
        time_epoch = e - s
        print(
            "Epoch: {} train_loss: {:.3f} train_acc: {:.2f}%, test_acc: {:.2f}%, ood_loss: {:.3f}, time: {:.1f}".format(
                epoch, train_loss, train_acc * 100, test_acc * 100, ood_loss, time_epoch
            )
        )