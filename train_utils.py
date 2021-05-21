import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import *
import os
import time
import gc
import shutil
import h5py

from dfw.dfw import DFW
from dfw.dfw.losses import set_smoothing_enabled
from dfw.dfw.losses import MultiClassHingeLoss
from dfw.experiments.models.densenet import DenseNet3
from dfw.experiments.models.wide_resnet import WideResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(state, is_best, epoch, name):
    filename = 'checkpoints/v1_ckpt_'+name
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename+'_best')

def test_model(device, model_class, ckpt, test_data):
    model = model_class().cuda()
    print(model)
    checkpoint = torch.load('checkpoints/'+ckpt)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loader = DataLoader(test_data, 64, num_workers=2, pin_memory=True)  
    sm = nn.Softmax(dim=1)  
    total_correct = 0
    total_images = 0
    model.eval()        
    for i, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            sm_outputs = sm(outputs)
            outputs = torch.argmax(outputs, dim=1)
            total_correct += (outputs == labels).sum().item()
            total_images += len(labels)
        torch.cuda.empty_cache()
        gc.collect()
    model_accuracy = total_correct / total_images * 100
    return model_accuracy

def accuracy(out, targets, topk=1):
    if topk == 1:
        _, pred = torch.max(out, 1)
        acc = torch.mean(torch.eq(pred, targets).float())
    else:
        _, pred = out.topk(topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        acc = correct[:topk].view(-1).float().sum(0) / out.size(0)
    return 100. * acc

def plot_train(losses, errors, accs):
    plt.plot(range(len(losses)), losses, label = "Train loss", color='blue')
    plt.plot(range(len(errors)), errors, label = "Val loss", color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train vs Val loss')
    plt.legend()
    plt.show()
    plt.plot(range(len(accs)), accs, label = "Val accuracy", color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train vs Accuracy')
    plt.legend()
    plt.show()
    
class Net_Trainer(object):
    def __init__(self, device, model_class, ckpt_name, trainset, valset, n_classes=10, depth=40, loss='ce', load_ckpt=None, epochs=20, batch_size=64, lr=0.001, optm='sgd', mom=0.0, weight_decay=0, smooth=False, lr_decay=None, step=None):
        self.device = device
        self.lr =  lr
        self.momentum = mom
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.start_epoch = 0
        self.epochs = epochs
        self.workers = 2
        self.seed = int(time.time())
        self.print_freq = 1
        self.checkpoint_path = load_ckpt 
        self.best_error = 1e8
        self.best_epoch = 0
        self.ckpt_name = ckpt_name
        self.loss_history = []
        self.error_history = []
        self.accuracies = []
        self.log_dict = {}
        self.trainset = trainset
        self.valset = valset
        self.smooth_svm = smooth
        self.step = step
        torch.cuda.manual_seed(self.seed)
        print(self.device,torch.cuda.get_device_name(0))

        if model_class == 'wrn':
            self.model = WideResNet(depth, n_classes, widen_factor=4).cuda()
        elif model_class == 'dn':
            self.model = DenseNet3(depth, n_classes, growth_rate=40, bottleneck=True).cuda()
        else:
            raise ValueError

        if loss == 'svm':
            self.criterion = MultiClassHingeLoss().cuda()
        elif loss == 'ce':
            self.criterion = nn.CrossEntropyLoss().cuda()
        else:
            raise ValueError

        if optm == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum, nesterov=bool(self.momentum))
        elif optm == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif optm == "adagrad":
            self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif optm == "amsgrad":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, amsgrad=True)
        elif optm == 'dfw':
            self.optimizer = DFW(self.model.parameters(), eta=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        elif optm == 'bpgrad':
            self.optimizer = BPGrad(self.model.parameters(), eta=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        else:
            raise ValueError(optm)
        
        ### Load saved checkpoint
        if self.checkpoint_path:
            checkpoint = torch.load('checkpoints/'+self.checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch']+1
            loss = checkpoint['loss']
            self.best_error = checkpoint['best_error']
            self.best_epoch = checkpoint['best_epoch']
            self.loss_history = checkpoint['loss_hist']
            self.error_history = checkpoint['err_hist']
            self.accuracies = checkpoint['acc_hist']
        
    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            start = time.time()
            loss = self.train_epoch(epoch)
            if epoch%1 != 0:
                print('Epoch:'+str(epoch),'  Loss:'+str(round(loss,3)))
                continue
            error, acc = self.validate(epoch)
            end = time.time() - start
            if error < self.best_error:
                is_best = True
                self.best_epoch = epoch
            else:
                is_best = False
            self.loss_history.append(loss)
            self.error_history.append(error)
            self.accuracies.append(acc)
            self.best_error = min(error, self.best_error)
            print('Epoch:'+str(epoch),'  Loss:'+str(round(loss,3)),'  Val loss:'+str(round(error,3)),'  Accuracy:'+str(round(acc.item(),3)),'  Best val loss:'+str(round(self.best_error,3)),'  Time taken:'+str(round(end,3)))
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
                'best_error': self.best_error,
                'best_epoch': self.best_epoch,
                'loss_hist': self.loss_history,
                'err_hist': self.error_history,
                'acc_hist': self.accuracies
            }, is_best, epoch, self.ckpt_name)
            if self.step and (epoch + 1) in self.step:
                self.decay_learning_rate()
        return self.best_epoch, self.loss_history, self.error_history, self.accuracies

    def train_epoch(self, cur_epoch):
        train_loader = DataLoader(self.trainset, self.batch_size, shuffle=True, num_workers=self.workers, pin_memory=True)        
        self.model.train()        
        losses = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                with set_smoothing_enabled(self.smooth_svm):
                    loss = self.criterion(outputs, labels)
                losses += loss.detach().cpu().item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step(lambda: float(loss))
            torch.cuda.empty_cache()
            gc.collect()
        return losses/len(train_loader)

    def validate(self, cur_epoch):
        val_loader = DataLoader(self.valset, self.batch_size, num_workers=self.workers, pin_memory=True)        
        self.model.eval()        
        errors = 0
        acc = 0
        for i, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                error = self.criterion(outputs, labels)
                errors += error.detach().cpu().item()
                acc += accuracy(outputs, labels)
            torch.cuda.empty_cache()
            gc.collect()
        return errors/len(val_loader), acc/len(val_loader)

    def decay_learning_rate(self):
        if isinstance(self.optimizer, torch.optim.SGD):
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.lr_decay
            self.lr = self.optimizer.param_groups[0]['lr']
        else:
            raise ValueError
        
    def regularization(self):
        reg = 0.5 * self.weight_decay * sum([p.data.norm() ** 2 for p in self.model.parameters()]) if self.weight_decay else 0
        return reg