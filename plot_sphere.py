import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from nets.inceptionresnetv2 import InceptionResNetV2
from nets.preresnet import PreActResNet18
from PIL import Image
import torch.nn as nn
from mypath import Path
import argparse

parser = argparse.ArgumentParser(description="Clean/noisy cluster retreival")
parser.add_argument('--weights', type=str, default=None, help='unsupervised weights')
parser.add_argument('--id-noise', type=float, default=0.0, help='id noise ratio')
parser.add_argument('--ood-noise', type=float, default=0.0, help='ood noise ratio')
parser.add_argument('--noise-ratio', type=float, default=0.0, help='CWNL noise ratio')
args = parser.parse_args()

weights = args.weights

size1 = 84
size = 84

if "mini" in weights:
    #Miniimagenet
    mean = [0.4728, 0.4487, 0.4031]
    std = [0.2744, 0.2663 , 0.2806]
    dataset = "miniimagenet_preset"
elif "webvis" in weights:
    #Webvision
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    dataset = "webvision"
    size1 = 256
    size = 224
elif "cifar100" in weights:
    #CIFAR-100
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    size1 = 32
    size = 32
    dataset = "cifar100"
elif "cifar10" in weights:
    #CIFAR-10
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    size1 = 32
    size = 32
    dataset = "cifar10"

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size1, interpolation=Image.BICUBIC),
    torchvision.transforms.CenterCrop(size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std)
])

if "mini" in weights:
    from datasets.miniimagenet_preset import make_dataset as make_dataset_mini
    from datasets.miniimagenet_preset import MiniImagenet84

    train_data, train_labels, val_data, val_labels, test_data, test_labels, clean_noisy = make_dataset_mini(noise_ratio=args.noise_ratio)
    trackset = MiniImagenet84(train_data, train_labels, transform=transforms)
    trackset.clean_noisy = clean_noisy
    num_class = 100
    clean_dist = torch.ones(len(trackset), dtype=torch.bool)
    out_dist = torch.ones(len(trackset), dtype=torch.bool)
    in_dist = torch.zeros(len(trackset), dtype=torch.bool)
    clean_dist[~clean_noisy] = 0
    out_dist[clean_noisy] = 0
    
elif "webvis" in weights:
    from datasets.webvision import webvision_dataset
    trackset = webvision_dataset(transform=transforms, mode="train", num_classes=50)
    num_class = 50
    in_dist = {s:0 for s in trackset.data}
    out_dist = {s:0 for s in trackset.data}
    clean_dist = {s:0 for s in trackset.data}
    in_dist = torch.tensor([in_dist[s] for s in trackset.data]).bool()
    out_dist = torch.tensor([out_dist[s] for s in trackset.data]).bool()
    clean_dist = torch.tensor([clean_dist[s] for s in trackset.data]).bool()
elif "cifar100" in weights:
    ood_r, id_r = float(args.ood_noise), float(args.id_noise)
    from datasets.cifar import CIFAR100
    num_class = 100

    if "places" in args.weights:
        trackset = CIFAR100(Path.db_root_dir('cifar100'), ood_noise=ood_r, id_noise=id_r, train=True, transform=transforms, corruption="places")
    else:
        trackset = CIFAR100(Path.db_root_dir('cifar100'), ood_noise=ood_r, id_noise=id_r, train=True, transform=transforms)
        
    out_dist = torch.tensor([1 if i in trackset.ids_ood else 0 for i in range(len(trackset))], dtype=torch.bool)
    in_dist = torch.tensor([1 if i in trackset.ids_id else 0 for i in range(len(trackset))], dtype=torch.bool)
    clean_dist = torch.tensor([1 if (i not in trackset.ids_id and i not in trackset.ids_ood) else 0 for i in range(len(trackset))], dtype=torch.bool)
    ids_anno = np.arange(len(trackset))
elif "cifar10" in weights:
    ood_r, id_r = float(args.ood_noise), float(args.id_noise)
    from datasets.cifar import CIFAR10
    num_class = 10

    if "places" in args.weights:
        trackset = CIFAR10(Path.db_root_dir('cifar10'), ood_noise=ood_r, id_noise=id_r, train=True, transform=transforms, corruption="places")
    else:
        trackset = CIFAR10(Path.db_root_dir('cifar10'), ood_noise=ood_r, id_noise=id_r, train=True, transform=transforms)
        
    out_dist = torch.tensor([1 if i in trackset.ids_ood else 0 for i in range(len(trackset))], dtype=torch.bool)
    in_dist = torch.tensor([1 if i in trackset.ids_id else 0 for i in range(len(trackset))], dtype=torch.bool)
    clean_dist = torch.tensor([1 if (i not in trackset.ids_ood and i not in trackset.ids_id) else 0 for i in range(len(trackset))], dtype=torch.bool)
    ids_anno = np.arange(len(trackset))
else:
    raise NotImplementedError

track_loader = torch.utils.data.DataLoader(trackset, batch_size=100, shuffle=True, num_workers=12)

display_acc = 0

if "cifar10" in weights:
    net = "preresnet18"
elif "webvis" in weights or "mini" in weights:
    net = "inception"
else:
    raise NotImplementedError

proj_size = 2#Change here if using UMAP to do 128->2

if net == "inception":
    model = InceptionResNetV2(num_classes=num_class, proj_size=proj_size)
elif net == "preresnet18":
    model = PreActResNet18(num_classes=num_class, proj_size=proj_size)
    
dic = torch.load(weights)["state_dict"]

model.load_state_dict(dic, strict=True)
model.cuda()
model.eval()

features = torch.zeros(len(trackset), proj_size)

tbar = tqdm(track_loader)
tbar.set_description('Computing features...')

t1 = t2 = 0
for i, sample in enumerate(tbar):
    image, ids = sample['image'], sample['index']
    
    image = image.cuda()
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            _, feats = model(image, return_features=True)
            
            features[ids] = feats.detach().cpu()

#UMAP could be used to go from 128->2 but some approximations are made (fixed neigh size)
#import umap.umap_ as umap
#reductor = umap.UMAP(n_components=2, n_neighbors=50, verbose=True, metric="cosine")
#features = reductor.fit_transform(features)
#features = features / np.sqrt((features*features).sum(axis=-1, keepdims=True))

from sklearn.linear_model import LogisticRegression
labs = torch.zeros(len(features))
labs[out_dist] = 1
reg = LogisticRegression(penalty="none").fit(features, labs)
print("Separability", reg.score(features, labs))
subset = torch.randperm(len(features))[:1000]
feats = features[subset]

plt.xlim(-1.2, 1.2)
plt.ylim(-1.2, 1.2)
# Retrieve the model parameters.
b = reg.intercept_[0]
w1, w2 = reg.coef_.T
# Calculate the intercept and gradient of the decision boundary.
c = -b/w2
m = -w1/w2

# Plot the data and the classification with the decision boundary.
xmin, xmax = -1, 1
ymin, ymax = -1, 1
xd = np.array([xmin, xmax])
yd = m*xd + c

plt.plot(xd, yd, 'k', lw=1, ls='--', label="OOD boundary")
plt.scatter(feats[clean_dist[subset], 0], feats[clean_dist[subset], 1], s=1, alpha=.8, label="Clean", color="cornflowerblue", zorder=0)
plt.scatter(feats[in_dist[subset], 0], feats[in_dist[subset], 1], s=1, alpha=.8, label="ID noise", color="darkorchid", zorder=1)
plt.scatter(feats[out_dist[subset], 0], feats[out_dist[subset], 1], s=1, alpha=.8, label="OOD", color="indianred", zorder=2)

lgnd = plt.legend(prop={'size':10}, loc="lower right")
lgnd.legendHandles[0]._sizes = [20]
lgnd.legendHandles[1]._sizes = [20]
lgnd.legendHandles[2]._sizes = [20]
lgnd.legendHandles[3]._sizes = [20]
plt.figure()

#Per class for cifar10 or first 10 classes
for c in range(10):
    plt.subplot(2,5,c+1)
    subset = torch.tensor([i for i in range(len(trackset)) if trackset.targets[i] == c], dtype=torch.long)
    reg = LogisticRegression(penalty="none").fit(features[subset], labs[subset])
    s = reg.score(features[subset], labs[subset])
    perm = torch.randperm(len(subset))[:500]
    subset = subset[perm]
    feats = features[subset]
    
    plt.title("Class {0}, separability {1:.2f}%".format(c, s*100))
    plt.scatter(feats[clean_dist[subset], 0], feats[clean_dist[subset], 1], s=1, alpha=1, color="cornflowerblue", label="Clean", zorder=0)
    plt.scatter(feats[in_dist[subset], 0], feats[in_dist[subset], 1], s=1, alpha=1, color="darkorchid", label="ID noise", zorder=1)
    plt.scatter(feats[out_dist[subset], 0], feats[out_dist[subset], 1], s=1, alpha=1, color="indianred", label="OOD", zorder=2)

    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    # Retrieve the model parameters.
    b = reg.intercept_[0]
    w1, w2 = reg.coef_.T
    # Calculate the intercept and gradient of the decision boundary.
    c = -b/w2
    m = -w1/w2
    
    # Plot the data and the classification with the decision boundary.
    xmin, xmax = -1, 1
    ymin, ymax = -1, 1
    xd = np.array([xmin, xmax])
    yd = m*xd + c
    plt.plot(xd, yd, 'k', lw=1, ls='--', label="OOD boundary")

plt.show()






