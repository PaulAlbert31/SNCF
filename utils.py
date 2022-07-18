import torch
import torchvision
import os
import numpy as np
import datasets
import torch.nn.functional as F
import copy
from PIL import Image
from mypath import Path

def multi_class_loss(pred, target):
    pred = F.log_softmax(pred, dim=1)
    loss = - torch.sum(target*pred, dim=1)
    return loss

def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1-lam)
    else:
        lam = 1

    device = x.get_device()
    batch_size = x.size()[0]
    
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam, index


def make_data_loader(args, **kwargs):

    if args.dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        size1 = 32
        size = 32
    elif args.dataset == 'cifar100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        size1 = 32
        size = 32
    elif args.dataset == "miniimagenet_preset":
        mean = [0.4728, 0.4487, 0.4031]
        std = [0.2744, 0.2663 , 0.2806]
        if args.net!="inception":
            size1 = 32
            size = 32
        else:
            size1 = 84
            size = 84
    elif args.dataset == 'webvision':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        if args.sup:#Supervised training
            size1 = 256
            size = 227
        else:
            size1 = 84
            size = 84

        
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size1, interpolation=Image.BICUBIC),
        torchvision.transforms.RandomCrop(size, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ])
        
    transform_unsup = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size1, interpolation=Image.BICUBIC),
        torchvision.transforms.RandomResizedCrop(size, interpolation=Image.BICUBIC),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        torchvision.transforms.RandomGrayscale(p=0.2),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ])
        
    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size1, interpolation=Image.BICUBIC),
        torchvision.transforms.CenterCrop(size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ])


    if args.dataset == "cifar10":
        from datasets.cifar import CIFAR10
        trainset = CIFAR10(Path.db_root_dir("cifar10"), ood_noise=args.ood_noise, id_noise=args.id_noise, train=True, transform=transform_train, cont=args.cont, transform_unsup=transform_unsup, consistency=len(args.id_in) > 0, corruption=args.corruption)
        testset = CIFAR10(Path.db_root_dir("cifar10"), ood_noise=args.ood_noise, id_noise=args.id_noise, classes_id=trainset.classes_id, train=False, transform=transform_test)
    elif args.dataset == "cifar100":
        from datasets.cifar import CIFAR100
        trainset = CIFAR100(Path.db_root_dir("cifar100"), ood_noise=args.ood_noise, id_noise=args.id_noise, train=True, transform=transform_train, cont=args.cont, transform_unsup=transform_unsup, consistency=len(args.id_in) > 0, corruption=args.corruption)
        testset = CIFAR100(Path.db_root_dir("cifar100"), ood_noise=args.ood_noise, id_noise=args.id_noise, classes_id=trainset.classes_id, train=False, transform=transform_test)
    elif args.dataset == "miniimagenet_preset":
        from datasets.miniimagenet_preset import make_dataset, MiniImagenet84
        train_data, train_labels, val_data, val_labels, test_data, test_labels, clean_noisy = make_dataset(noise_ratio=args.noise_ratio)
        trainset = MiniImagenet84(train_data, train_labels, transform=transform_train, cont=args.cont, transform_unsup=transform_unsup, consistency=len(args.id_in) > 0)
        testset = MiniImagenet84(val_data, val_labels, transform=transform_test)
    elif args.dataset == "webvision":
        from datasets.webvision import webvision_dataset
        trainset = webvision_dataset(transform=transform_train, mode="train", num_classes=50, cont=args.cont, transform_unsup=transform_unsup, consistency=len(args.id_in) > 0)
        testset = webvision_dataset(transform=transform_test, mode="test", num_classes=50)
    else:
        raise NotImplementedError("Dataset {} is not implemented".format(args.dataset))

    trackset = copy.deepcopy(trainset)
    trackset.cont = False
    trackset.transform = transform_test
    warmupset = copy.deepcopy(trainset)
    warmupset.cont = False
    warmupset.consistency = False
    
    track_loader = torch.utils.data.DataLoader(trackset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, **kwargs)

    if args.sup:
        if args.ts: #Equal sampler
            if len(args.id_ood) == 0 and len(args.id_in) > 0:
                from ThreeSampler import TwoStreamBatchSampler
                sampler = TwoStreamBatchSampler(args.id_clean, args.id_in, batch_size=args.batch_size)
                train_loader = torch.utils.data.DataLoader(trainset, batch_sampler=sampler, **kwargs)
                sampler = torch.utils.data.SubsetRandomSampler(args.id_clean)
                warmup_loader = torch.utils.data.DataLoader(warmupset, batch_size=args.batch_size, sampler=sampler, **kwargs)
            elif len(args.id_in) > 0:
                from ThreeSampler import ThreeStreamBatchSampler
                sampler = ThreeStreamBatchSampler(args.id_clean, args.id_in, args.id_ood, batch_size=args.batch_size)
                train_loader = torch.utils.data.DataLoader(trainset, batch_sampler=sampler, **kwargs)
                sampler = torch.utils.data.SubsetRandomSampler(args.id_clean)
                warmup_loader = torch.utils.data.DataLoader(warmupset, batch_size=args.batch_size, sampler=sampler, **kwargs)
            else:
                from ThreeSampler import TwoStreamBatchSampler
                sampler = TwoStreamBatchSampler(args.id_clean, args.id_ood, batch_size=args.batch_size)
                train_loader = torch.utils.data.DataLoader(trainset, batch_sampler=sampler, **kwargs)
                sampler = torch.utils.data.SubsetRandomSampler(args.id_clean)
                warmup_loader = torch.utils.data.DataLoader(warmupset, batch_size=args.batch_size, sampler=sampler, **kwargs)
        else:
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
            sampler = torch.utils.data.SubsetRandomSampler(args.id_clean)
            warmup_loader = torch.utils.data.DataLoader(warmupset, batch_size=args.batch_size, sampler=sampler, **kwargs)
        return warmup_loader, train_loader, test_loader, track_loader
    elif args.dataset != 'clothing':
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    else:
        indexs = torch.randperm(len(trainset))[:args.batch_size*1000]
        sampler = torch.utils.data.SubsetRandomSampler(indexs)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, sampler=sampler, **kwargs)
                
    return train_loader, test_loader, track_loader

def create_save_folder(args):
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    if not os.path.isdir(os.path.join(args.save_dir, args.net + '_'  + args.dataset)):
        os.mkdir(os.path.join(args.save_dir, args.net + '_'  + args.dataset))
    if not os.path.isdir(os.path.join(args.save_dir, args.net + '_'  + args.dataset, args.exp_name)):
        os.mkdir(os.path.join(args.save_dir, args.net + '_'  + args.dataset, args.exp_name))
    if not os.path.isdir(os.path.join(args.save_dir, args.net + '_'  + args.dataset, args.exp_name, str(args.seed))):
        os.mkdir(os.path.join(args.save_dir, args.net + '_'  + args.dataset, args.exp_name, str(args.seed)))
    return
