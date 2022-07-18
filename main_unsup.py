import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import make_data_loader, create_save_folder, multi_class_loss, mixup_data
import os
import random


#Mixed precision
from torch.cuda.amp import GradScaler, autocast

class Trainer(object):
    def __init__(self, args):
        #Compat
        args.strong_aug=False
        args.cont = True
        args.asym = False
        args.id_in = []
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
        
        self.kwargs = {'num_workers': 12, 'pin_memory': True}
        self.train_loader, self.val_loader, self.track_loader = make_data_loader(args, **self.kwargs)
   
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.args.steps, gamma=self.args.gamma)
        
        self.best = 0
        self.best_epoch = 0
        
        self.scaler = GradScaler(enabled=not self.args.fp32)
            
        
    def train(self, epoch):
        running_loss = 0.0
        self.model.train()
        
        acc = 0
        tbar = tqdm(self.train_loader)
        m_dists = torch.tensor([])
        l = torch.tensor([])
        self.epoch = epoch
        total_sum = 0

        for i, sample in enumerate(tbar):
            im, target, ids = sample['image'], sample['target'], sample['index']                                                                           
            if self.args.mixup:
                image1, _, _, lam, o = mixup_data(sample["image1"].cuda(), -torch.ones(self.args.batch_size))
            elif self.args.cutmix:
                image1, _, _, lam, o = cut_mix(sample["image1"].cuda(), -torch.ones(self.args.batch_size))
            else:
                image1 = sample["image1"].cuda()
                
            with autocast(enabled = not self.args.fp32):
                        
                _, feats1 = self.model(image1, return_features=True)
                _, feats2 = self.model(sample["image2"].cuda(), return_features=True)
            
                logits = torch.div(torch.matmul(feats1, feats2.t()), 0.2) #With temperature tau2
                
                labels = torch.arange(len(feats1)).cuda()
                labels = F.one_hot(labels).float()
                labels = torch.matmul(labels, labels.t())
                
                if self.args.mixup or self.args.cutmix:
                    loss = lam * multi_class_loss(logits, labels) + (1 - lam) * multi_class_loss(logits, labels[o])
                else:
                    loss = multi_class_loss(logits, labels)

            self.scaler.scale(loss.mean()).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            #self.optimizer.zero_grad()
            for param in self.model.parameters():
                param.grad = None
                
            if i % 10 == 0:
                tbar.set_description('Training loss {0:.2f}, LR {1:.6f}'.format(loss.mean(), self.optimizer.param_groups[0]['lr']))
                
        self.scheduler.step()
        #Checkpoint
        self.save_model(epoch)
        print('[Epoch: {}, numImages: {}, numClasses: {}]'.format(epoch, total_sum, self.args.num_class))
        return
    
    def save_model(self, epoch):
        checkname = os.path.join(self.args.save_dir, '{}.pth.tar'.format(self.args.checkname, epoch))
            
        torch.save({
            'epoch': epoch+1,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best': self.best,
            'best_epoch':self.best_epoch
        }, checkname)
            

    def track_features(self):
        self.model.eval()
        acc = 0
        total_sum = 0
        with torch.no_grad():
            tbar = tqdm(self.track_loader)
            tbar.set_description('Tracking features')
            
            features = torch.zeros(len(self.train_loader.dataset), self.args.proj_size)
            
            for i, sample in enumerate(tbar):
                image, ids = sample['image'], sample['index']
                if self.args.cuda:
                    image = image.cuda()
                
                with autocast(enabled = not self.args.fp32):
                    _, feats = self.model(image, return_features=True)

                features[ids] = feats.detach().cpu().float()
               
        return features
            
    def kNN(self, trainFeatures, K=200, sigma=0.1):
        # set the model to evaluation mode
        self.model.eval()
        # tracking variables
        total = 0

        trainFeatures = trainFeatures
        trainLabels = torch.from_numpy(self.track_loader.dataset.targets)
        if self.args.cuda:
            trainFeatures = trainFeatures.cuda()
            trainLabels = trainLabels.cuda()

        trainFeatures = trainFeatures.t()
        C = trainLabels.max() + 1
        C = C.item()
        # start to evaluate
        top1 = 0.
        top5 = 0.
        tbar = tqdm(self.val_loader)
        tbar.set_description("kNN eval")
        with torch.no_grad():
            retrieval_one_hot = torch.zeros(K, C)
            if self.args.cuda:
                retrieval_one_hot = retrieval_one_hot.cuda()
            for i, sample in enumerate(tbar):
                images, targets = sample["image"], sample["target"]
                if self.args.cuda:
                    images, targets = images.cuda(), targets.cuda()

                batchSize = images.size(0)
                # forward                                                                                                                                                                                                                                                         
                _, features = self.model(images, return_features=True)
                
                features = F.normalize(features, p=2)

                # cosine similarity                                                                                                                                                                                                                                               
                dist = torch.mm(features, trainFeatures)

                yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
                candidates = trainLabels.view(1,-1).expand(batchSize, -1)
                retrieval = torch.gather(candidates, 1, yi)

                retrieval_one_hot.resize_(batchSize * K, C).zero_()
                retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
                yd_transform = yd.clone().div_(sigma).exp_()
                probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C),
                                            yd_transform.view(batchSize, -1, 1)), 1)
                _, predictions = probs.sort(1, True)

                # Find which predictions match the target
                
                correct = predictions.eq(targets.data.view(-1,1))

                top1 = top1 + correct.narrow(1,0,1).sum().item()
                top5 = top5 + correct.narrow(1,0,5).sum().item()

                total += targets.size(0)

        print("kNN accuracy", top1/total)
        if self.best <= top1/total:
            self.best = top1/total
            self.best_epoch = self.epoch
            
        return top1/total
    
def main():


    parser = argparse.ArgumentParser(description="PyTorch N-pairs")
    parser.add_argument('--net', type=str, default='preresnet18',
                        choices=['preresnet18', 'inception'],
                        help='net name (default: preresnet18)')
    parser.add_argument('--dataset', type=str, default='miniimagenet_preset', choices=['miniimagenet_preset', 'webvision','cifar100', 'cifar10'])
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0.1, help='Multiplicative factor for lr decrease, default .1')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--steps', type=int, default=None, nargs='+', help='Epochs when to reduce lr')
    parser.add_argument('--save-dir', type=str, default='checkpoints')
    parser.add_argument('--checkname', type=str, default=None)
    parser.add_argument('--exp-name', type=str, default='')
    parser.add_argument('--seed', default=1, type=float)
    parser.add_argument('--mixup', default=False, action='store_true')
    parser.add_argument('--no-cuda', default=False, action='store_true') #Probably does not work

    parser.add_argument('--noise-ratio', default="0.2", type=str) #For CWNL
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--fp32', default=False, action='store_true')
    parser.add_argument('--proj-size', type=int, default=128)

    #CIFAR
    parser.add_argument('--ood-noise', default=.0, type=float)
    parser.add_argument('--id-noise', default=.0, type=float) #Not really usefull because unsupervised
    parser.add_argument('--corruption', default="inet", type=str)

    args = parser.parse_args()
    args.sup = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    seeds = {'1': round(torch.exp(torch.ones(1)).item()*1e6), '2': round(torch.acos(torch.zeros(1)).item() * 2), '3':round(torch.sqrt(torch.tensor(2.)).item()*1e6)}
    try:
        torch.manual_seed(seeds[str(args.seed)])
        torch.cuda.manual_seed_all(seeds[str(args.seed)])  # GPU seed
        random.seed(seeds[str(args.seed)])  # python seed for image transformation                                                                                                                                                                                              
    except:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # GPU seed
        random.seed(args.seed)

    dict_class = {'webvision':50, 'miniimagenet_preset':100, 'cifar100':100, 'cifar10': 10}
    
    args.num_class = dict_class[args.dataset]
        
    if args.steps is None:
        args.steps = [args.epochs]
    if args.checkname is None:
        args.checkname = "{}_{}".format(args.net, args.dataset)
        
    create_save_folder(args)
    args.save_dir = os.path.join(args.save_dir, args.net + '_'  + args.dataset, args.exp_name, str(args.seed))
    args.cuda = not args.no_cuda
    
    _trainer = Trainer(args)
    save_dict = {}

    #Refinement of the class labels
    start_ep = 0
        
    if args.resume is not None:
        load_dict = torch.load(args.resume, map_location='cpu')
        _trainer.model.module.load_state_dict(load_dict['state_dict'])
        _trainer.optimizer.load_state_dict(load_dict['optimizer'])
        _trainer.scheduler.load_state_dict(load_dict['scheduler'])
        start_ep = load_dict['epoch']
        del load_dict
        
    for eps in range(start_ep, args.epochs):
        _trainer.train(eps)

        if eps%10 == 0: #Making sure the training converges, not used for early stopping
            features = _trainer.track_features()
            _trainer.kNN(features)

if __name__ == "__main__":
   main()
