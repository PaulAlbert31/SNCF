import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

from utils import make_data_loader, create_save_folder, multi_class_loss, mixup_data

import os
import random
import copy

from torch.cuda.amp import GradScaler, autocast
    
class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.epoch = 0
        
        if args.net == 'inception':
            from nets.inceptionresnetv2 import InceptionResNetV2
            model = InceptionResNetV2(num_classes=self.args.num_class, proj_size=self.args.proj_size)
        elif args.net == 'preresnet18':
            from nets.preresnet import PreActResNet18
            model = PreActResNet18(num_classes=self.args.num_class, proj_size=self.args.proj_size)
        else:
            raise NotImplementedError("Network {} is not implemented".format(args.net))
        
        print('Number of parameters', sum(p.numel() for p in model.parameters() if p.requires_grad))

        self.model = nn.DataParallel(model).cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
        
        self.kwargs = {'num_workers': 0, 'pin_memory': True}
        ood_file = self.args.noise_file.replace('clean_noisy', 'ood_labels')
        self.approx_clean = torch.load(self.args.noise_file)
        self.ood_labels = torch.load(ood_file)
        print(self.approx_clean)
        
        self.is_ood = torch.from_numpy((self.approx_clean == 2)) #1 if ood
        self.is_idn = torch.from_numpy((self.approx_clean == 1)) #1 if id noise
        self.is_clean = torch.from_numpy((self.approx_clean == 0)) #1 if clean
        self.is_id = torch.from_numpy((self.approx_clean < 2)) #1 if in distribution
        print("Noise ratio (Clean, IDN, OOD)", self.is_clean.sum() / len(self.is_clean), self.is_idn.sum() / len(self.is_idn), self.is_ood.sum() / len(self.is_ood))
        
        self.noise_r = (self.approx_clean == 2).sum() / len(self.approx_clean)#Zeta for CNWL 32x32
        if self.noise_r == 0:
            self.noise_r = .5
        
        r = torch.arange(len(self.approx_clean))
        
        args.id_clean = r[self.is_clean]
        args.id_in = r[self.is_idn]
        args.id_ood = r[self.is_ood]
        args.id_id = r[self.is_id]
        
        self.warmup_loader, self.train_loader, self.val_loader, self.track_loader = make_data_loader(args, **self.kwargs)
           
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.args.steps, gamma=self.args.gamma)
        self.criterion_nored = nn.CrossEntropyLoss(reduction='none')
        self.best = 0
        self.best_epoch = 0
        self.scaler = GradScaler(enabled=not self.args.fp32)
            
        
    def train(self, epoch):
        running_loss = 0.0
        self.model.train()
        
        acc = 0
        if epoch >= self.args.warmup:
            tbar = tqdm(self.train_loader)
        else:
            tbar = tqdm(self.warmup_loader)
            
        self.epoch = epoch
        total_sum = 0
        for i, sample in enumerate(tbar):
            target_cont = copy.deepcopy(sample['target']).float() #For the guided contrastive learning
            if epoch >= self.args.warmup:
                ids = sample['index']
                batch_clean = self.is_clean[ids]
                batch_idn = self.is_idn[ids]
                batch_ood = self.is_ood[ids]
                im_l1, im_l2, t_l = sample['image'][batch_clean], sample['image_'][batch_clean], sample['target'][batch_clean]
                im_i1, im_i2, t_i = sample['image'][batch_idn], sample['image_'][batch_idn], sample['target'][batch_idn]
                ids_l, ids_i = sample['index'][batch_clean], sample['index'][batch_idn]
                if self.args.cuda:
                    im_l1, im_l2, t_l = im_l1.cuda(), im_l2.cuda(), t_l.cuda()
                    im_i1, im_i2, t_i = im_i1.cuda(), im_i2.cuda(), t_i.cuda()

                with torch.no_grad():
                    with autocast(enabled = not self.args.fp32):
                        out1 = F.softmax(self.model(sample['image'].cuda()), dim=1)
                        out2 = F.softmax(self.model(sample['image_'].cuda()), dim=1)
                        if len(im_i1) > 0:
                            #Label guessing for ID noisy samples
                            target_idn = (out1[batch_idn] + out2[batch_idn]) / 2
                            target_idn = target_idn **2 #temp sharp
                            target_idn = target_idn / target_idn.sum(dim=1, keepdim=True) #normalization
                            target_idn = target_idn.detach()
                            t_i = target_idn
                            target_cont[batch_idn] = t_i.cpu() #Update the guide for the contrastive loss

                if not self.args.ts:
                    im = torch.cat([im_l1, im_i1], dim=0)
                    target = torch.cat([t_l, t_i], dim=0)
                    ids = torch.cat((ids_l, ids_i))
                    n_clean = batch_clean.sum()
                else:
                    im = torch.cat([im_l1, im_l2, im_i1], dim=0) #For equal batch sizes with the contrastive forward pass
                    target = torch.cat([t_l, t_l, t_i], dim=0)
                    ids = torch.cat((ids_l, ids_l, ids_i))
                    n_clean = batch_clean.sum()*2                
            else:
                im_l1, im_l2, t_l, ids_l = sample['image'], sample['image_'], sample['target'], sample['index']
                if self.args.cuda:
                    im_l1, im_l2, t_l = im_l1.cuda(), im_l2.cuda(), t_l.cuda()
                    
                if not self.args.ts:
                    im = im_l1
                    target = t_l
                    ids = ids_l
                else:
                    im = torch.cat([im_l1, im_l2], dim=0)
                    target = torch.cat([t_l, t_l], dim=0)
                    ids = torch.cat((ids_l, ids_l))
                    
            if self.args.mixup:
                image, la, lb, lam, o = mixup_data(im, target, alpha=self.args.mixup_alpha)#Mix the ID data
                target = lam*la + (1-lam)*lb
            else:
                image = im
                
            with autocast(enabled = not self.args.fp32):
                outputs = self.model(image)
                #Supervised ce pass
                if self.args.mixup:
                    loss_c = lam * multi_class_loss(outputs, la) + (1-lam) * multi_class_loss(outputs, lb)
                    loss_c = loss_c.mean()
                else:
                    loss_c = multi_class_loss(outputs, target).mean()
                #Track training acc if not mixup
                if not self.args.mixup:
                    preds = torch.argmax(F.log_softmax(outputs, dim=1), dim=1)
                    acc += torch.sum(preds == torch.argmax(target, dim=1))
                    
                total_sum += outputs.shape[0]
                    
                #Contrastive pass
                if (self.args.cont and epoch >= self.args.warmup):
                    ii = sample["index"]
                    
                    if self.args.mixup:
                        image1, _, _, lam, o = mixup_data(sample["image1"].cuda(), -torch.ones(len(sample["image1"])))
                    else:
                        image1 = sample["image1"]

                    #Constrastive feature learning on heavily augmented images
                    _, feats1 = self.model(image1, return_features=True)
                    _, feats2 = self.model(sample["image2"].cuda(), return_features=True)
                    #Corrected target
                    labels = torch.argmax(target_cont, dim=1).cuda()
                    m = labels.max()
                    
                    if self.is_ood[ii].sum() > 0: #Group OOD data using cluster assignments
                        ood_lab = self.ood_labels[ii][self.is_ood[ii]]
                        r = torch.arange(len(ood_lab[ood_lab == -1]))
                        ood_lab[ood_lab == -1] = ood_lab.max() + r + 1 #OOD labs not assigned to OOD clusters
                        ood_lab = ood_lab.cuda()
                        labels[self.is_ood[ii]] = ood_lab + m + 1 # OOD labs part of a cluster
                        
                    labels = F.one_hot(labels).float()
                    labels = torch.matmul(labels, labels.t())

                    #feats are l2 normalized in the network forward pass
                    logits = torch.div(torch.matmul(feats1, feats2.t()), 0.2)
                    if self.args.mixup:
                        labels = lam * labels + (1-lam) * labels[o]

                    loss_u = multi_class_loss(logits, labels) / labels.sum(dim=-1)
                               
                    if self.args.dataset == "miniimagenet_preset": #Only if size 32x32
                        loss_u *= self.noise_r

                    loss_c += loss_u.mean() 
                        
                else:
                    loss_u = torch.cuda.FloatTensor([0])

                loss = loss_c.mean()
                
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
            if i % 10 == 0:
                tbar.set_description('Training loss {0:.2f}, LR {1:.6f}, L_class {2:.2f}, L_cont {3:.2f}'.format(loss.mean(), self.optimizer.param_groups[0]['lr'], loss_c.mean(), loss_u.mean()))
        self.scheduler.step()
        #Checkpoint
        self.save_model(epoch)
        print('[Epoch: {}, numImages: {}, numClasses: {}]'.format(epoch, total_sum, self.args.num_class))
        if not self.args.mixup:
            print('Training Accuracy: {0:.4f}'.format(float(acc)/total_sum))
        return
                    
    def val(self, epoch, dataset='val', save=True):
        self.model.eval()
        acc = 0

        vbar = tqdm(self.val_loader)
        total = 0
        losses, accs = torch.tensor([]), torch.tensor([])
        
        with torch.no_grad():
            for i, sample in enumerate(vbar):
                image, target = sample['image'], sample['target']
                if self.args.cuda:
                    image, target = image.cuda(), target.cuda()
                    
                with autocast(enabled = not self.args.fp32):
                    outputs = self.model(image)
                    loss = self.criterion_nored(outputs, target)

                losses = torch.cat((losses, loss.cpu()))

                preds = torch.argmax(F.log_softmax(outputs, dim=1), dim=1)
                accs = torch.cat((accs, (preds==target.data).float().cpu()))

                acc += torch.sum(preds == target.data)
                total += preds.size(0)

                if i % 10 == 0:
                    if dataset == 'val':
                        vbar.set_description('Validation loss: {0:.2f}'.format(loss.mean()))
                    else:
                        vbar.set_description('Test loss: {0:.2f}'.format(loss.mean()))
        final_acc = float(acc)/total
        if i % 10 == 0:
            print('[Epoch: {}, numImages: {}]'.format(epoch, (len(self.val_loader)-1)*self.args.batch_size + image.shape[0]))
        if final_acc > self.best and save:
            self.best = final_acc
            self.best_epoch = epoch
            self.save_model(epoch, best=True)
            
        print('Validation Accuracy: {0:.4f}, best accuracy {1:.4f} at epoch {2}'.format(final_acc, self.best, self.best_epoch))
        return final_acc, losses.mean(), accs.mean()

    def save_model(self, epoch, t=False, best=False):
        if t:
            checkname = os.path.join(self.args.save_dir, '{}_{}.pth.tar'.format(self.args.checkname, epoch))
        elif best:
            checkname = os.path.join(self.args.save_dir, '{}_best.pth.tar'.format(self.args.checkname, epoch))
            with open(os.path.join(self.args.save_dir, 'bestpred_{}.txt'.format(self.args.checkname)), 'w') as f:
                f.write(str(self.best))
        else:
            checkname = os.path.join(self.args.save_dir, '{}.pth.tar'.format(self.args.checkname, epoch))
            
        torch.save({
            'epoch': epoch+1,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best': self.best,
            'best_epoch':self.best_epoch
        }, checkname)
            
def main():


    parser = argparse.ArgumentParser(description="SNCF")
    parser.add_argument('--net', type=str, default='preresnet18',
                        choices=['preresnet18', 'inception'],
                        help='net name (default: preresnet18)')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['miniimagenet_preset', 'webvision', 'cifar100'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1, help='Multiplicative factor for lr decrease, default .1')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--steps', type=int, default=None, nargs='+', help='Epochs when to reduce lr')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Probably not working')
    parser.add_argument('--save-dir', type=str, default='checkpoints')
    parser.add_argument('--checkname', type=str, default=None)
    parser.add_argument('--exp-name', type=str, default='')
    parser.add_argument('--seed', default=1, type=float)
    parser.add_argument('--mixup', default=False, action='store_true')
    parser.add_argument('--strong-aug', default=False, action='store_true')

    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--pretrained', default=None, type=str)
    parser.add_argument('--cont', default=False, action='store_true')
    parser.add_argument('--fp32', default=False, action='store_true')
    parser.add_argument('--warmup', default=30, type=int)
    parser.add_argument('--proj-size', type=int, default=128)
    parser.add_argument('--ts', default=False, action='store_true')
    parser.add_argument('--mixup-alpha', default=1, type=float)
    parser.add_argument('--noise-file', type=str, default=None)

    #MiniImageNet
    parser.add_argument('--noise-ratio', default="0.2", type=str)

    #CIFAR100
    parser.add_argument('--ood-noise', default=.0, type=float)
    parser.add_argument('--id-noise', default=.0, type=float)
    parser.add_argument('--corruption', default="inet", type=str)   

    args = parser.parse_args()
    args.sup = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    seeds = {'1': round(torch.exp(torch.ones(1)).item()*1e6), '2': round(torch.acos(torch.zeros(1)).item() * 2), '3':round(torch.sqrt(torch.tensor(2.)).item()*1e6)}
    try:
        torch.manual_seed(seeds[str(args.seed)])
        torch.cuda.manual_seed_all(seeds[str(args.seed)])  # GPU seed
        random.seed(seeds[str(args.seed)])  # python seed for image transformation                                                                                                                                                                                              
    except:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # GPU seed
        random.seed(args.seed)

    dict_class = {'webvision':50, 'miniimagenet_preset':100, 'cifar100':100}
    
    args.num_class = dict_class[args.dataset]
        
    if args.steps is None:
        args.steps = [args.epochs]
        
    create_save_folder(args)
    args.checkname = args.net + '_' + args.dataset
    args.save_dir = os.path.join(args.save_dir, args.checkname, args.exp_name, str(args.seed))
    args.cuda = not args.no_cuda

    
    _trainer = Trainer(args)
    #Use one hot targets
    relabel = torch.tensor(_trainer.train_loader.dataset.targets)
    relabel = F.one_hot(relabel, num_classes=args.num_class)
    
    _trainer.train_loader.dataset.targets = relabel
    _trainer.warmup_loader.dataset.targets = relabel
    
    start_ep = 0
    if args.pretrained is not None:#Self sup init, not used by default
        load_dict = torch.load(args.pretrained, map_location='cpu')['state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in load_dict.items():
            if not "linear" in k:
                new_state_dict[k] = v
        load_dict = new_state_dict       
        _trainer.model.module.load_state_dict(load_dict, strict=False)
    
    if args.resume is not None:
        load_dict = torch.load(args.resume, map_location='cpu')
        _trainer.model.module.load_state_dict(load_dict['state_dict'])
        _trainer.optimizer.load_state_dict(load_dict['optimizer'])
        _trainer.scheduler.load_state_dict(load_dict['scheduler'])
        start_ep = load_dict['epoch']
        del load_dict
        v, loss, acc = _trainer.val(start_ep)

    for eps in range(start_ep, args.epochs):
        _trainer.train(eps)
        v, loss, acc = _trainer.val(eps)

if __name__ == "__main__":
   main()
