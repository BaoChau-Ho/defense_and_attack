import torch
import argparse
import sys
from os import system
import time
import copy
import numpy as np
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.models.inception import InceptionAux
from torchvision.models import mobilenet_v2, resnet50, alexnet, resnet101, densenet121, inception_v3, DenseNet121_Weights, ResNet101_Weights, ResNet50_Weights, MobileNet_V2_Weights, AlexNet_Weights, Inception_V3_Weights
from torchvision import transforms
from pytorchtools import EarlyStopping
from utils import Logger
from pynvml import *

import json
import os
import openpyxl as yxl
import numpy as np
from openpyxl.styles import Alignment
from openpyxl.styles.borders import Border, Side

def make_directory(dirs):
    for dir in dirs:
        if(not os.path.exists(dir)):
            os.makedirs(dir)

parser = argparse.ArgumentParser()
parser.add_argument("--index_model",type=int)
parser.add_argument("--check_ckpt",type=int, default=0)
args=parser.parse_args()

def get_dataset_index(model_idx):
    if(model_idx<6): dak=0
    elif(model_idx<12): dak=1
    elif(model_idx<18): dak=2
    elif(model_idx<24): dak=3
    return dak

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

bs={
    "cifar100": 16,
    "cifar10": 16,
    "mnist": 16,
    "imgnet": 16
}

models = ['resnet50_CIFA100', 'resnet101_CIFA100', 'mobilenet_v2_CIFA100', 'alexnet_CIFA100', 'densenet121_CIFA100','inception_v3_CIFA100',
          'resnet50_CIFA10', 'resnet101_CIFA10', 'mobilenet_v2_CIFA10', 'alexnet_CIFA10', 'densenet121_CIFA10', 'inception_v3_CIFA10',
          'resnet50_MNIST', 'resnet101_MNIST', 'mobilenet_v2_MNIST', 'alexnet_MNIST', 'densenet121_MNIST', 'inception_v3_MNIST',
          'resnet50_IMGNET', 'resnet101_IMGNET','mobilenet_v2_IMGNET','alexnet_IMGNET','densenet121_IMGNET','inception_v3_IMGNET']
all_datasets = ['cifar100','cifar10','mnist','imgnet']
training_method = "ATGR"

father_directory = "/media/administrator/Data1/BC/Test_Python/"
model_father_directory = "/media/administrator/Data1/BC/Test_Python/"

#adver_method_name = adver_methods[2]
# 0-5: cifar100
# 6-11: cifar10
# 12-17: mnist
# 18-23: imgnet
# /home/kt/bc/
model_name = models[args.index_model] 
dataset_name = all_datasets[get_dataset_index(args.index_model)]
data_path = father_directory+"datasets_{}/".format(dataset_name)
ori_model_path = model_father_directory+"models/{}/weights_{}_best.h5".format(model_name, model_name)
model_dst_dir = father_directory+"def_models/{}/{}/".format(training_method,model_name)
log_dst_path = model_dst_dir+"log.txt"
make_directory([model_dst_dir])

mean = (0.4914, 0.4822, 0.4465)
std = (0.247, 0.243, 0.261)
    
def transforms_data_cifa100(width=224, height=224):
    transforms_dict = {
        'train':
        transforms.Compose([
            transforms.Resize((width, height)),
            #transforms.RandomAffine(0,scale=(0.8, 1.2)),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(), # ngang
            transforms.ToTensor(),
            transforms.Normalize(mean , std)
        ]),
        'validation':
        transforms.Compose([
            transforms.Resize((width, height)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }
    return transforms_dict
    
def transforms_data_mnist(width=224, height=224):
    transforms_dict = {
        'train':
        transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize((width, height)),
            #transforms.RandomAffine(0,scale=(0.8, 1.2)),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(), # ngang
            transforms.ToTensor(),
            transforms.Normalize(mean , std)
        ]),
        'validation':
        transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize((width, height)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }
    return transforms_dict

def transforms_data_cifa10(width=224, height=224):
    transforms_dict = {
        'train':
        transforms.Compose([
            transforms.Resize((width, height)),
            #transforms.RandomAffine(0,scale=(0.8, 1.2)),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(), # ngang
            transforms.ToTensor(),
            transforms.Normalize(mean , std)
        ]),
        'validation':
        transforms.Compose([
            transforms.Resize((width, height)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }
    return transforms_dict

def transforms_data_imgnet():
    if(model_name=='inception_v3_MNIST'):
        transforms_dict = {
            "train": transforms.Compose([
                transforms.RandomResizedCrop(299),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
                transforms.Normalize(mean, std)]),
            "validation": transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
                transforms.Normalize(mean, std)])
        }        
    else:
        # models except for inception
        transforms_dict = {
            'train':
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'validation':
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }
    return transforms_dict

def initModel(path = None):
    if(model_name == 'mobilenet_v2_CIFA100'):
        model = mobilenet_v2(weights = MobileNet_V2_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features=in_features, out_features=100)
    if(model_name == 'resnet50_CIFA100'):
        model = resnet50(weights = ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features=in_features, out_features = 100)
    if(model_name == 'alexnet_CIFA100'):
        model = alexnet(weights = AlexNet_Weights.DEFAULT)
        in_features = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(in_features = in_features, out_features = 100)
    if(model_name == 'resnet101_CIFA100'):
        model = resnet101(weights = ResNet101_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features = in_features, out_features = 100)
    if(model_name == 'densenet121_CIFA100'):
        model = densenet121(weights = DenseNet121_Weights.DEFAULT)
        in_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_features=in_features, out_features=100)
    if(model_name == 'inception_v3_CIFA100'):
        model = inception_v3(weights = Inception_V3_Weights.DEFAULT)
        in_features = model.fc.in_features
        in_features2 = 768
        model.fc = torch.nn.Linear(in_features = in_features, out_features = 100)
        model.AuxLogits = InceptionAux(in_channels = in_features2, num_classes = 100)
    if(model_name == 'resnet50_CIFA10'):
        model = resnet50(weights = ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features=in_features, out_features = 10)
    if(model_name == 'resnet101_CIFA10'):
        model = resnet101(weights = ResNet101_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features = in_features, out_features = 10)
    if(model_name == 'mobilenet_v2_CIFA10'):
        model = mobilenet_v2(weights = MobileNet_V2_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features=in_features, out_features=10)       
    if(model_name == 'alexnet_CIFA10'):
        model = alexnet(weights = AlexNet_Weights.DEFAULT)
        in_features = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(in_features = in_features, out_features = 10) 
    if(model_name == 'densenet121_CIFA10'):
        model = densenet121(weights = DenseNet121_Weights.DEFAULT)
        in_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_features=in_features, out_features=10)     
    if(model_name == 'inception_v3_CIFA10'):
        model = inception_v3(weights = Inception_V3_Weights.DEFAULT)
        in_features = model.fc.in_features
        in_features2 = 768
        model.fc = torch.nn.Linear(in_features = in_features, out_features = 10)
        model.AuxLogits = InceptionAux(in_channels = in_features2, num_classes = 10)  
    if(model_name == 'resnet50_MNIST'):
        model = resnet50(weights = ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features=in_features, out_features = 10)
    if(model_name == 'resnet101_MNIST'):
        model = resnet101(weights = ResNet101_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features = in_features, out_features = 10)
    if(model_name == 'mobilenet_v2_MNIST'):
        model = mobilenet_v2(weights = MobileNet_V2_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features=in_features, out_features=10)       
    if(model_name == 'alexnet_MNIST'):
        model = alexnet(weights = AlexNet_Weights.DEFAULT)
        in_features = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(in_features = in_features, out_features = 10) 
    if(model_name == 'densenet121_MNIST'):
        model = densenet121(weights = DenseNet121_Weights.DEFAULT)
        in_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_features=in_features, out_features=10)     
    if(model_name == 'inception_v3_MNIST'):
        model = inception_v3(weights = Inception_V3_Weights.DEFAULT)
        in_features = model.fc.in_features
        in_features2 = 768
        model.fc = torch.nn.Linear(in_features = in_features, out_features = 10)
        model.AuxLogits = InceptionAux(in_channels = in_features2, num_classes = 10)         
    if(model_name == 'resnet50_IMGNET'):
        model = resnet50(weights = ResNet50_Weights.DEFAULT)
    if(model_name == 'resnet101_IMGNET'):
        model = resnet101(weights = ResNet101_Weights.DEFAULT)
    if(model_name == 'mobilenet_v2_IMGNET'):
        model = mobilenet_v2(weights = MobileNet_V2_Weights.DEFAULT)    
    if(model_name == 'alexnet_IMGNET'):
        model = alexnet(weights = AlexNet_Weights.DEFAULT)
    if(model_name == 'densenet121_IMGNET'):
        model = densenet121(weights = DenseNet121_Weights.DEFAULT)
    if(model_name == 'inception_v3_IMGNET'):
        model = inception_v3(weights = Inception_V3_Weights.DEFAULT)     
    if(path!=None and dataset_name!='imgnet'): 
        model.load_state_dict(torch.load(path, weights_only = False))
    return model.to(device)

def initData(path):
    if(dataset_name=='cifar100'):
        if(model_name=='inception_v3_CIFA100'): data_transforms = transforms_data_cifa100(width = 299, height = 299)
        else: data_transforms = transforms_data_cifa100()
        train_data = datasets.CIFAR100(
            root = path,
            train = True,
            download = True,
            transform = data_transforms['train']
        )
        test_data = datasets.CIFAR100(
            root = path,
            train = False,
            download = True,
            transform = data_transforms['validation']
        )  
    elif(dataset_name=='cifar10'):
        if(model_name=='inception_v3_CIFA10'): data_transforms = transforms_data_cifa10(299,299)
        else: data_transforms = transforms_data_cifa10()
        train_data = datasets.CIFAR10(
            root = path,
            train = True,
            download = True,
            transform = data_transforms['train']
        )
        test_data = datasets.CIFAR10(
            root = path,
            train = False,
            download = True,
            transform = data_transforms['validation']
        )
    elif(dataset_name=='mnist'):
        print(path)
        if(model_name=='inception_v3_MNIST'): data_transforms = transforms_data_mnist(299,299)
        else: data_transforms = transforms_data_mnist()
        train_data = datasets.MNIST(
            root = path,
            train = True,
            download = True,
            transform = data_transforms['train']
        )
        test_data = datasets.MNIST(
            root = path,
            train = False,
            download = True,
            transform = data_transforms['validation']
        )
    elif(dataset_name=='imgnet'):
        data_transforms = transforms_data_imgnet()
        train_data = datasets.ImageNet(root = path, split = "val", transforms = data_transforms['train'])
        test_data = datasets.ImageNet(root = path, split = "val", transforms = data_transforms['validation'])
    train_loader = DataLoader(train_data, batch_size = bs[dataset_name], shuffle = False)
    test_loader = DataLoader(test_data, batch_size = bs[dataset_name], shuffle = False)
    return train_loader, test_loader


pgd_criterion = torch.nn.CrossEntropyLoss()
pgd_lr = 2/225
if(model_name[:12]=="inception_v3"): pgd_lr=2/299
pgd_epochs = 50
pgd_epsilon = 0.1

train_criterion = torch.nn.CrossEntropyLoss()
train_alpha = 1
train_alpha_2 = 1
train_lr = 0.001
train_percentage = {
    "train": 30/100,
    "validate": 30/100
}

def pgd(model, X, y, criterion, lr, epsilon, epochs):
    delta = torch.zeros_like(X, requires_grad = True, device = device)
    for i in range(epochs):
        nX = X + delta
        loss = criterion(model(nX), y)
        loss.backward()
        delta.data = (delta + lr * delta.grad.data).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    nX = X + delta.detach()
    return nX

def get_l2_regu(model, imgs, labels, criterion):
    adv_imgs = pgd(model, imgs, labels, pgd_criterion, pgd_lr, pgd_epsilon, pgd_epochs)
    adv_imgs.requires_grad = True
    adv_outs = model(adv_imgs)
    adv_loss = criterion(adv_outs, labels)
    adv_loss.backward()
    l2_regu = torch.linalg.norm(adv_imgs.grad)        
    return l2_regu
    
def train_model(model, loader, criterion, alpha, alpha2, lr, checkpoint_path, epochs):
    model.train()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = lr)
    early_stopping = EarlyStopping(patience=7, verbose=True, path=checkpoint_path)    
    best_acc = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    for iter in range(epochs):
        print("Epoch {}/{}".format(iter+1,epochs))
        print("-"*10)
        for phase in ["train","validate"]:
            running_loss = 0
            running_corrects = 0
            total = 0
            number_of_batch = (len(loader[phase].dataset) * train_percentage[phase]) / bs[dataset_name]
            for batch_idx, (imgs, labels) in enumerate(loader[phase]):
                if((batch_idx+1)%50==0 or batch_idx==0): print("epoch: {} - batch_idx: {}".format(iter+1, batch_idx+1))
                if((batch_idx+1)==number_of_batch): break

                total += imgs.shape[0]
                imgs = imgs.to(device)
                labels = labels.to(device)

                model.eval()

                adv_imgs = pgd(model, imgs, labels, pgd_criterion, pgd_lr, pgd_epsilon, pgd_epochs)
                adv_outs = model(adv_imgs)
                adv_loss = criterion(adv_outs, labels)

                clean_outs = model(imgs)
                clean_loss = criterion(clean_outs, labels)
                
                l2_regu = get_l2_regu(model, imgs, labels, criterion)

                loss = clean_loss + alpha * adv_loss + alpha2 * l2_regu

                if(phase=="train"):
                    model.train()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                running_loss += loss.item()
                adv_preds = torch.argmax(adv_outs, dim = 1)
                running_corrects += torch.sum(adv_preds == labels)

            epoch_loss = running_loss/total
            epoch_acc = running_corrects/total

            if(phase=='validate'):
                valid_loss = epoch_loss
                valid_acc = epoch_acc

            print('phase:{} - loss: {:.4f} - acc: {:.4f}'.format(phase,epoch_loss, epoch_acc))
        
        if(valid_acc>best_acc):
            best_acc=valid_acc
            wts = copy.deepcopy(model.state_dict())
            best_model_wts = wts
            torch.save(best_model_wts, checkpoint_path)

        early_stopping(valid_loss, model)
        if(early_stopping.early_stop):
            print("Early Stopped")
            break
    model.load_state_dict(best_model_wts)
    return model, best_acc

if __name__=='__main__':
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    try:
        gpu_busy = True
        while(gpu_busy):
            info = nvmlDeviceGetMemoryInfo(h)
            print(f'used     : {int(info.used/(1.05*1000000))}',
            end="\r", flush=True)
            gpu_busy = int(info.used/(1.05*1000000)) > 6000
        system("cls")

        sys.stdout = Logger(log_dst_path)

        print(model_name)
        print(dataset_name)        
        print(training_method)
        
        train_loader, test_loader = initData(data_path)
        loader = {
            "train": train_loader,
            "validate": test_loader
        }
        best_model_path = model_dst_dir+"weights_{}_best.h5".format(model_name)
        ckpt_model_path = model_dst_dir+"weights_{}_checkpoint.pt".format(model_name)        
        if(args.check_ckpt==0): nnModel = initModel(ori_model_path)
        else: nnModel=initModel(ckpt_model_path)
        best_nnModel,acc = train_model(
            model = nnModel, 
            loader = loader, 
            criterion = train_criterion, 
            alpha = train_alpha, 
            alpha2 = train_alpha_2,
            lr = train_lr, 
            checkpoint_path =ckpt_model_path,
            epochs = 150)
        torch.save(best_nnModel.state_dict(), best_model_path)


    except KeyboardInterrupt:
        print("Press Ctrl-C to terminate while statement")
    
