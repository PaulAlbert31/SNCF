import torch
import torchvision
import faiss
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from PIL import Image
import scipy.sparse
from sklearn.cluster import OPTICS
from sklearn.neighbors import NearestNeighbors
from mypath import Path
import argparse
import os

def min_max(x, mi=None, ma=None):
    if mi is None:
        mi = x.min()
    if ma is None:
        ma = x.max()    
    return (x - mi)/(ma - mi)

#Affinity matrix, eq 3
def get_affinity(X, k=100):
    N = X.shape[0]

    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    D, I = index.search(X, k + 1)

    D = D[:,1:] ** 3
    I = I[:,1:]
    row_idx = np.arange(N)
    row_idx_rep = np.tile(row_idx,(k,1)).T
    W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))
    W = W + W.T
    W = W - scipy.sparse.diags(W.diagonal())

    S = W.sum(axis = 1)
    S[S==0] = 1
    D = np.array(1./ np.sqrt(S))
    D = scipy.sparse.diags(D.reshape(-1))
    Wn = D * W * D

    return Wn

#Compute the embedding from the affinity matrix
def get_embedding(graph, dim):
    diag_data = np.asarray(graph.sum(axis=0)).flatten()
    # Symetric Normalized Laplacian
    I = scipy.sparse.identity(graph.shape[0], dtype=np.float64)
    D = scipy.sparse.spdiags(
        1.0 / np.sqrt(diag_data), 0, graph.shape[0], graph.shape[0]
    )
    
    L = I - D * graph * D

    k = dim + 1
    num_lanczos_vectors = max(2 * k + 1, int(np.sqrt(graph.shape[0])))
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
        L,
        k,
        which="SM",
        ncv=num_lanczos_vectors,
        tol=1e-4,
        v0=np.ones(L.shape[0]),
        maxiter=graph.shape[0] * 5,
    )

    order = np.argsort(-eigenvalues)[:dim]
    E = eigenvectors[:, order]

    return E
   

#Get the cluster assignments from the embedding at the class level
def get_clusters(embed, feats):
    if dataset == "cifar100":
        #OPTICS parameters
        neighborhood = 75
        xi = .02
        #Custom cluster discovery
        min_s = 75
        tol = .4
    elif dataset == "webvision":
        neighborhood = 100
        xi= .02
        min_s = 50
        tol = .4
    else:
        raise NotImplementedError(dataset)
        
    #Contrusting the chain at three neighborhood sizes and extracting the clusters, using the original OPTICS cluster detection might work better for low noise cases on CIFAR
    scan = OPTICS(min_samples=neighborhood, metric="cosine", xi=xi, min_cluster_size=min_s)
    labels = scan.fit_predict(embed)
    
    scan2 = OPTICS(min_samples=neighborhood-25, metric="cosine", xi=xi, min_cluster_size=min_s)
    labels2 = scan2.fit_predict(embed)
    
    scan3 = OPTICS(min_samples=neighborhood-50, metric="cosine", xi=xi, min_cluster_size=min_s)
    labels3 = scan3.fit_predict(embed)
    
    final_scan = scan
    #Switch to a better neighborhood size if less outlier (-1) are present. Conditional on having discovered more than 1 cluster.
    if len(labels2[labels2 == -1]) < len(labels[labels == -1]) and len(labels2[labels2 == 1]) > 0 or len(labels[labels == 1]) == 0:
        print('Switching to -25')
        labels = labels2
        final_scan = scan2
    if len(labels3[labels3 == -1]) < len(labels[labels == -1]) and len(labels3[labels3 == 1]) > 0 or len(labels[labels == 1]) == 0:
        print('Switching to -50')
        labels = labels3
        final_scan = scan3

    uni, counts = np.unique(labels, return_counts=True)
 
    dists = []
    u = uni[uni>=0] #No outliers    
    for l in uni[uni>=0]:
        n = min(neighborhood, len(labels[labels==l]))
        nbrs = NearestNeighbors(n_neighbors=n, metric='cosine').fit(feats[labels==l])#Faiss will return -1 as distance if no neighbors are found
        distances, _ = nbrs.kneighbors(feats[labels==l]) #Computing on the unsupervised features not the embedding
        d = np.mean(distances[:, 1:], axis=1)
        dists.append(np.mean(d))
            
    dists = np.array(dists)
    
    if len(dists) == 1: #Only one cluster was found
        o = [0]
    else:
        o = min_max(dists)

    #Deciding which cluster is the OOD
    final_labels = np.zeros(len(labels)) 
    for u in uni[uni>=0]:
        if o[u] == 1.: #Cluster with highest internal distances
            final_labels[labels==u] = 2
    
    final_labels[labels==-1] = 1

    if dataset == "webvision":
        #Additional precautions on webvision, as we observe multiple clean modes in a class
        if dists.min() / dists.max() > .8 or len(final_labels[final_labels == 2]) > len(final_labels[final_labels == 0]):
            final_labels[final_labels == 2] = 0
            final_labels[final_labels == 1] = 0
        
    labels = final_labels    
    return labels

parser = argparse.ArgumentParser(description="Clean/noisy cluster retreival")
parser.add_argument('--weights', type=str, default=None, help='unsupervised weights')
parser.add_argument('--id-noise', type=float, default=0.0, help='id noise ratio')
parser.add_argument('--ood-noise', type=float, default=0.0, help='ood noise ratio')
parser.add_argument('--noise-ratio', type=float, default=0.0, help='CWNL noise ratio')
args = parser.parse_args()

weights = args.weights

if "mini" in weights:
    #Miniimagenet
    mean = [0.4728, 0.4487, 0.4031]
    std = [0.2744, 0.2663 , 0.2806]
    dataset = "miniimagenet_preset"
    size1 = 84
    size = 84
elif "webvis" in weights:
    #Webvision
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    dataset = "webvision"
    size1 = 84
    size = 84
elif "cifar100" in weights:
    #CIFAR-100
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    dataset = "cifar100"
    size1 = 32
    size = 32

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
    clean_dist = {s:1 for s in trackset.data}

    in_dist = torch.tensor([in_dist[s] for s in trackset.data]).bool()
    out_dist = torch.tensor([out_dist[s] for s in trackset.data]).bool()
    clean_dist = torch.tensor([clean_dist[s] for s in trackset.data]).bool()
elif "cifar" in weights:
    ood_r, id_r = float(args.ood_noise), float(args.id_noise)
    from datasets.cifar import CIFAR100
    num_class = 100

    if "places" in args.weights:
        trackset = CIFAR100(Path.db_root_dir('cifar100'), ood_noise=ood_r, id_noise=id_r, train=True, transform=transforms, asym=False, corruption="places")
    else:
        trackset = CIFAR100(Path.db_root_dir('cifar100'), ood_noise=ood_r, id_noise=id_r, train=True, transform=transforms, asym=False)
        
    out_dist = torch.tensor([1 if i in trackset.ids_ood else 0 for i in range(len(trackset))], dtype=torch.bool)
    in_dist = torch.tensor([1 if i in trackset.ids_id else 0 for i in range(len(trackset))], dtype=torch.bool)
    clean_dist = torch.tensor([1 if (i not in trackset.ids_id and i not in trackset.ids_ood) else 0 for i in range(len(trackset))], dtype=torch.bool)
else:
    raise NotImplementedError

track_loader = torch.utils.data.DataLoader(trackset, batch_size=100, shuffle=True, num_workers=12)

#CIFAR-100

display_acc = 0

if "cifar100" in weights:
    net = "preresnet18"
elif "webvis" in weights or "mini" in weights:
    net = "inception"
elif "clothing" in weights:
    net = "resnet50"
else:
    raise NotImplementedError

if net == "inception":
    from nets.inceptionresnetv2 import InceptionResNetV2
    model = InceptionResNetV2(num_classes=num_class, proj_size=128)
elif net == "preresnet18":
    from nets.preresnet import PreActResNet18
    model = PreActResNet18(num_classes=num_class, proj_size=128)

dic = torch.load(weights)["state_dict"]

model.load_state_dict(dic, strict=True)
model.cuda()
model.eval()

features = torch.zeros(len(trackset), 128)

tbar = tqdm(track_loader)
tbar.set_description('Computing features...')

for i, sample in enumerate(tbar):
    image, target, ids = sample['image'], sample['target'], sample['index']
    target, image = target.cuda(), image.cuda()
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            _, feats = model(image, return_features=True)

            features[ids] = feats.detach().cpu() #Features L2 normalized in the forward pass
 
features = features.numpy()

#Get the embedding
graph = get_affinity(features, 50)
red_dim = 20
embedding = get_embedding(graph, red_dim).astype(np.float32)

#Detecting at the dataset level on mini since little ID noise is present
if dataset == "miniimagenet_preset":
    dists = []
    if float(args.noise_ratio) in [0.2, 0.8]:
        cv = "diag"
    else:
        cv = "full"
    from sklearn.mixture import GaussianMixture
    scan = GaussianMixture(n_components=2, n_init=50, covariance_type=cv) #diag can help for imbalanced cluster sizes
    labels = scan.fit_predict(embedding)
    
    for l in [0, 1]:
        index = faiss.IndexFlatIP(features[labels==l].shape[1])
        index.add(features[labels==l])
        D, _ = index.search(features[labels==l], 100)
        dists.append(np.mean(D[:, 1:].mean(axis=1)))
        
    id_ood_clust = torch.zeros(len(labels))
    
    if dists[0] > dists[1]:
        id_ood_clust[labels==0] = 2
    else:
        id_ood_clust[labels==1] = 2
    
    fpr, tpr, thresholds = metrics.roc_curve(out_dist, (id_ood_clust == 2))
    print('Retreival OOD', metrics.auc(fpr, tpr))
    
    print(np.unique(id_ood_clust, return_counts=True))
else:
    id_ood_clust = - np.ones(len(trackset), dtype=np.int)
    for c in tqdm(range(num_class)):
        ids_c = (torch.tensor(trackset.targets) == c)

        embedding_c = embedding[ids_c]
        feats = features[ids_c] #To compute accurate distances

        if dataset == 'miniimagenet_preset':
            ood_labels = id_ood_clust[ids_c]
        else:
            ood_labels = get_clusters(embedding_c, feats)

        id_ood_clust[ids_c] = ood_labels
        
       
#Re cluster the OOD samples (see supplementary material)
graph = get_affinity(features[id_ood_clust==2], 50)
embedding = get_embedding(graph, red_dim).astype(np.float32)

scan = OPTICS(min_samples=100, metric="cosine", xi=0.02)
labels = scan.fit_predict(embedding)

#Visualize the clustering
ood_labels = - torch.ones(len(trackset), dtype=torch.long) * 2 #-2 for the ID data, -1 unassigned OOD, the rest is OOD clusters
ood_labels[id_ood_clust == 2] = torch.from_numpy(labels)

#Saving the weights and ood clusters to use in the supervised phase
path = f"noise_files/{dataset}"
if not os.path.isdir("noise_files"):
    os.mkdir("noise_files")
if not os.path.isdir(path):
    os.mkdir(path)
if dataset == "cifar100":
    if "places" in args.weights:
        torch.save(id_ood_clust, os.path.join(path, "clean_noisy_{}_{}_{}_places.pth.tar".format(dataset, id_r, ood_r)))
        torch.save(ood_labels, os.path.join(path, "ood_labels_{}_{}_{}_places.pth.tar".format(dataset, id_r, ood_r)))
    else:
        torch.save(id_ood_clust, os.path.join(path, "clean_noisy_{}_{}_{}.pth.tar".format(dataset, id_r, ood_r)))
        torch.save(ood_labels, os.path.join(path, "ood_labels_{}_{}_{}.pth.tar".format(dataset, id_r, ood_r)))
elif dataset == "miniimagenet_preset":
    torch.save(id_ood_clust, os.path.join(path, "clean_noisy_{}_{}.pth.tar".format(dataset, args.noise_ratio)))
    torch.save(ood_labels, os.path.join(path, "ood_labels_{}_{}.pth.tar".format(dataset, args.noise_ratio)))
elif dataset == "webvision":
    torch.save(id_ood_clust, os.path.join(path, "clean_noisy_webvis.pth.tar"))
    torch.save(ood_labels, os.path.join(path, "ood_labels_webvis.pth.tar"))

ret = np.zeros(len(ids_c))
ra = np.arange(len(trackset))
interest_c = ra[id_ood_clust==0]
interest_n = ra[id_ood_clust==1]
interest_o = ra[id_ood_clust==2]

#Print the noise distribution accross the detected subsets
print("In clean", clean_dist[interest_c].sum() / len(interest_c), "In id", clean_dist[interest_n].sum() / len(interest_n), "In ood", clean_dist[interest_o].sum() / len(interest_o))
print("In clean", in_dist[interest_c].sum() / len(interest_c), "In id", in_dist[interest_n].sum() / len(interest_n), "In ood", in_dist[interest_o].sum() / len(interest_o))
print("In clean", out_dist[interest_c].sum() / len(interest_c), "In id", out_dist[interest_n].sum() / len(interest_n), "In ood", out_dist[interest_o].sum() / len(interest_o))
