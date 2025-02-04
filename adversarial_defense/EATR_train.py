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
import torchattacks

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
    "cifar100": 4,
    "cifar10": 8,
    "mnist": 8,
    "imgnet": 8
}

models = ['resnet50_CIFA100', 'resnet101_CIFA100', 'mobilenet_v2_CIFA100', 'alexnet_CIFA100', 'densenet121_CIFA100','inception_v3_CIFA100',
          'resnet50_CIFA10', 'resnet101_CIFA10', 'mobilenet_v2_CIFA10', 'alexnet_CIFA10', 'densenet121_CIFA10', 'inception_v3_CIFA10',
          'resnet50_MNIST', 'resnet101_MNIST', 'mobilenet_v2_MNIST', 'alexnet_MNIST', 'densenet121_MNIST', 'inception_v3_MNIST',
          'resnet50_IMGNET', 'resnet101_IMGNET','mobilenet_v2_IMGNET','alexnet_IMGNET','densenet121_IMGNET','inception_v3_IMGNET']

all_datasets = ['cifar100','cifar10','mnist','imgnet']
training_method = "EATR"

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
data_path = father_directory + "datasets_{}/".format(dataset_name)
ori_model_path = os.path.join(model_father_directory,"models/{}/weights_{}_best.h5".format(model_name, model_name))
model_dst_dir = father_directory + "models/{}/{}/".format(training_method,model_name)
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
        model.load_state_dict(torch.load(path, weights_only = True))
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
    train_loader = DataLoader(train_data, batch_size = bs[dataset_name], shuffle = True)
    test_loader = DataLoader(test_data, batch_size = bs[dataset_name], shuffle = False)
    return train_loader, test_loader
    
def freeze(model):
    model.eval()

def unfreeze(model):
    model.train()
    for p in model.parameters():
        p.requires_grad = True

pgd_criterion = torch.nn.CrossEntropyLoss()
pgd_lr = 2/255
#if(model_name[:9]=="inception"): pgd_lr = 2/299
pgd_epsilon = 0.1
pgd_epochs = 5

fgsm_criterion = torch.nn.CrossEntropyLoss()
fgsm_epsilon = 0.1

bim_epsilon = 0.1
bim_alpha = 1/255
#if(model_name[:9]="inception"): bim_alpha = 2/299

#cw_lr = 5e-4
cw_lr = 0.01
#cw_steps = 100
cw_steps = 4
#cw_c = 0.5
cw_c = 1


train_criterion = torch.nn.CrossEntropyLoss()
train_fgsm_alpha = 1
train_pgd_alpha = 1
train_bim_alpha = 1
train_cw_alpha  = 1
train_lr = 0.001
train_epochs = 120
train_percentage = {
    "train": 20/100,
    "validate": 20/100
}

# pgd in l2 norm
def pgd(model, X, y, criterion, lr, epsilon, epochs):
    # model eval = True
    flag = 0
    if(model.training):
        flag=1
        model.eval()    
    delta = torch.zeros_like(X, requires_grad = True, device = device)
    for i in range(epochs):
        nX = X + delta
        loss = criterion(model(nX), y)
        loss.backward()
        delta.data = (delta + lr * delta.grad.data).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    nX = X + delta.detach()
    if(flag): model.train()
    return nX
    
def fgsm(model, imgs, labels, criterion, epsilon):
    flag = 0
    if(model.training):
        flag=1
        model.eval()        
    delta = torch.zeros_like(imgs, requires_grad = True, device = device)
    outs = model(imgs+delta)
    loss=criterion(outs,labels)
    loss.backward()
    nimgs = imgs + epsilon * delta.grad.detach().sign()
    nimgs = torch.clamp(nimgs, 0, 1)
    if(flag): model.train()
    return nimgs

def cw_attack_l2(model, imgs, labels, lr, step, c):
    atk = torchattacks.CW(model,c=c, steps=step, lr=lr)
    adv_imgs = atk(imgs,labels)
    return adv_imgs


def bim(model, imgs, labels, epsilon, alpha = 1/255):
    atk = torchattacks.BIM(model, eps = epsilon, alpha = alpha, steps = 0)
    adv_imgs = atk(imgs, labels)
    return adv_imgs

def adver_method(model, images, labels, method):
    if(method == 'BIM'): 
        return bim(model, images, labels, bim_epsilon, alpha = bim_alpha)
    if(method == 'FGSM'): 
        return fgsm(model, images, labels, fgsm_criterion, fgsm_epsilon)
    if(method == 'PGD'): 
        return pgd(model, images, labels, pgd_criterion, pgd_lr, pgd_epsilon, pgd_epochs)
    if(method == 'CW_L2'):
        return cw_attack_l2(model, images, labels, cw_lr, cw_bs_step, cw_c)

def get_grad_regu(model, imgs, labels, criterion):
    l2_regu = 0
    for method in ['FGSM','BIM','PGD','CW_L2']:
        adv_imgs = adver_method(model, imgs, labels, method)
        adv_imgs.requires_grad = True
        adv_outs = model(adv_imgs)
        adv_loss = criterion(adv_outs, labels)
        adv_loss.backward()
        tmp = torch.linalg.norm(adv_imgs.grad)        
        l2_regu = l2_regu + tmp**2
    return l2_regu

def get_l2_regu(model, lambda_reg=0.01):
    reg_loss = 0
    for param in model.parameters():
        reg_loss += torch.sum(param ** 2)
    return lambda_reg * torch.sqrt(reg_loss)
    
def get_outputs(model, imgs, phase):
    if(model_name[:12]=="inception_v3" and phase=="train"): outs, aux_outs = model(imgs)
    else: outs = model(imgs)
    return outs

def get_loss(model, imgs, labels, criterion, phase):
    if(model_name[:12]=="inception_v3" and phase=="train"): 
        outs, aux_outs = model(imgs)
        loss = criterion(outs, labels) + 0.4*criterion(aux_outs, labels)
    else: 
        outs = model(imgs)
        loss = criterion(outs, labels)
    return loss


def train_model(model, loader, criterion, fgsm_alpha, pgd_alpha, bim_alpha, cw_alpha, lr, checkpoint_path, epochs=500):
    model.train()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = lr)
    early_stopping = EarlyStopping(patience=10, verbose=True, path=checkpoint_path)    
    best_acc = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    for iter in range(epochs):
        print("Epoch {}/{}".format(iter+1,epochs))
        print("-"*10)
        for phase in ["train","validate"]:
            if(phase=="validate"): model.eval()
            else: model.train()            
            running_loss = 0
            running_corrects = {
                "bim":0,
                "pgd":0,
                "fgsm":0,
                "cw":0
            }
            total = 0
            number_of_batch = (len(loader[phase].dataset) * train_percentage[phase]) / bs[dataset_name]
            for batch_idx, (imgs, labels) in enumerate(loader[phase]):
                if((batch_idx+1)%25==0 or batch_idx==0): print("epoch: {} - batch_idx: {}".format(iter+1, batch_idx+1))
                if((batch_idx+1)==number_of_batch): break

                total += imgs.shape[0]
                imgs = imgs.to(device)
                labels = labels.to(device)

                pgd_imgs = adver_method(model, imgs, labels, "PGD")
                pgd_outs = get_outputs(model, pgd_imgs, phase)
                pgd_loss = get_loss(model, pgd_imgs, labels, criterion, phase)

                fgsm_imgs = adver_method(model, imgs, labels, "FGSM")
                fgsm_outs = get_outputs(model, fgsm_imgs, phase)
                fgsm_loss = get_loss(model, fgsm_imgs, labels, criterion, phase)

                bim_imgs = adver_method(model, imgs, labels, "BIM")
                bim_outs = get_outputs(model, bim_imgs, phase)
                bim_loss = get_loss(model, bim_imgs, labels, criterion, phase)

                cw_imgs = adver_method(model, imgs, labels, "CW_L2")
                cw_outs = get_outputs(model, cw_imgs, phase)
                cw_loss = get_loss(model, cw_imgs, labels, criterion, phase)

                clean_outs = get_outputs(model, imgs, phase)
                clean_loss = get_loss(model, imgs, labels, criterion, phase)

                l2_regu = get_l2_regu(model)

                loss = clean_loss +  fgsm_alpha * fgsm_loss + pgd_alpha * pgd_loss + bim_alpha * bim_loss + cw_alpha * cw_loss + l2_regu

                if(phase=="train"):
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                running_loss += loss.item()
                bim_preds = torch.argmax(bim_outs, dim = 1)
                pgd_preds = torch.argmax(pgd_outs, dim = 1)
                fgsm_preds = torch.argmax(fgsm_outs, dim = 1)
                cw_preds = torch.argmax(cw_outs, dim = 1)
                running_corrects["bim"] += torch.sum(bim_preds == labels)
                running_corrects["pgd"] += torch.sum(pgd_preds == labels)
                running_corrects["cw"] += torch.sum(cw_preds == labels)
                running_corrects["fgsm"] += torch.sum(fgsm_preds == labels)

            epoch_loss = running_loss/total
            epoch_acc = 0
            for dak in running_corrects.values():
                epoch_acc += dak / total
            epoch_acc /= 4

            if(phase=='validate'):
                valid_loss = epoch_loss
                valid_acc = epoch_acc

            print('phase:{} - loss: {:.4f} - acc: {:.4f}'.format(phase,epoch_loss, epoch_acc))
            print('fgsm: {} - pgd: {} - bim: {} - cw: {}'.format(running_corrects["fgsm"]/total, running_corrects["pgd"]/total, running_corrects["bim"]/total, running_corrects["cw"]/total))
        
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
            fgsm_alpha = train_fgsm_alpha,
            pgd_alpha = train_pgd_alpha,
            bim_alpha = train_bim_alpha,
            cw_alpha = train_cw_alpha,
            lr = train_lr, 
            checkpoint_path =ckpt_model_path,
            epochs = train_epochs)
        torch.save(best_nnModel.state_dict(), best_model_path)


    except KeyboardInterrupt:
        print("Press Ctrl-C to terminate while statement")
    
