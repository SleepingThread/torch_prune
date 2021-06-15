'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as transforms

import time
import sys
import os
from collections import OrderedDict

from . import VariationalDropout
from .vgg import VGG
from .vgg_train_config import data_path, tb_logdir, checkpoint_path, use_vd, vd_config, \
    scheduler_T_max


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root=data_path, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root=data_path, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

train_writer = SummaryWriter(log_dir=os.path.join(tb_logdir, "train"))
test_writer = SummaryWriter(log_dir=os.path.join(tb_logdir, "test"))

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = VGG('VGG19')

if checkpoint_path is not None:
    # load state dict
    sd = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    sd = OrderedDict([(_k[len("module."):], _v) for _k, _v in sd.items()])
    net.load_state_dict(sd)
    print("Model loaded")

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if use_vd:
    modules = [(net.module.classifier, None)]
    for _i, _f in enumerate(net.module.features):
        if isinstance(_f, torch.nn.Conv2d):
            modules.append((_f, None))

    vd = VariationalDropout(modules, **vd_config["constructor"])
    print("VD applied")

net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_T_max)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    _start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        if use_vd:
            loss = loss + vd.get_dkl(vd_config["vd_lambda"])

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        sys.stdout.write("\r%d from %d | %.1fs" % (batch_idx, len(trainloader), time.time() - _start_time))
    sys.stdout.write("\r")
    sys.stdout.write('Train (%.1fs): Loss: %.3f | Acc: %.3f%% (%d/%d)\n' %
                     (time.time() - _start_time,
                      (train_loss/(batch_idx+1)), 100.*correct/total, correct, total))

    # tensorboard
    train_loss = (train_loss/(batch_idx+1))
    train_accuracy = 100.*correct/total
    train_writer.add_scalar("loss", train_loss, epoch)
    train_writer.add_scalar("acc", train_accuracy, epoch)

    for _name, _val in net.named_parameters():
        train_writer.add_histogram(_name, _val, epoch)

    train_writer.flush()


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    _start_time = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            sys.stdout.write("\r%d from %d | %.1fs" % (batch_idx, len(testloader), time.time() - _start_time))

    sys.stdout.write("\r")
    sys.stdout.write('Train (%.1fs): Loss: %.3f | Acc: %.3f%% (%d/%d)\n' %
                     (time.time() - _start_time,
                      (test_loss/(batch_idx+1)), 100.*correct/total, correct, total))

    # tensorboard
    test_loss = (test_loss/(batch_idx+1))
    test_accuracy = 100.*correct/total

    test_writer.add_scalar("loss", test_loss, epoch)
    test_writer.add_scalar("acc", test_accuracy, epoch)

    test_writer.flush()
