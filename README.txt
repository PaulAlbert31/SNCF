#Pytorch code to run SNCF
Paper (ECCV 2022): [SNCF](https://arxiv.org/pdf/2207.01573)
[couldn't display image](https://github.com/PaulAlbert31/SNCF/blob/master/SNCF.pdf)

#Install requirements using conda (cuda version 10.2)
conda env create -f environment.yml
conda activate sncf

#Edit the mypath.py file
mypath.py lists the location of the datasets
For the Places365 dataset, we use the small images (256x256) test set available at http://places2.csail.mit.edu/download.html (4.4G)
For ImageNet32 see https://patrykchrabaszcz.github.io/Imagenet32/

#Pretrained unsupervised weights and noise clustering
This is a link to pretrained unsupervised weights and noise detection for seed 1 of this repository: [CIFAR-100](https://drive.google.com/drive/folders/1pyCWGwAqU1cesjwqOJNb-bv3faWTACN5?usp=sharing), [miniImageNet](https://drive.google.com/drive/folders/1x74sP4rk7umEqq9E5iOqk8TaMhMV-ujN?usp=sharing)

#Visualizing the linear separation of contrastive features on the 2D hypersphere for CIFAR-10

## Phase 1 - learn unsupervised features using iMix + N-pairs - projection size for the contrastive feature is 2 for a 2D hypersphere
CUDA_VISIBLE_DEVICES=0 python main_unsup.py --dataset cifar10 --ood-noise 0.2 --id-noise 0.0 --epochs 2000 --batch-size 256 --net preresnet18 --lr 0.01 --steps 1000 1500 --seed 1 --exp-name cifar10_test --proj-size 2 --mixup

## Phase 2 - vizualize the separation
CUDA_VISIBLE_DEVICES=0 python plot_sphere.py --weights checkpoints/preresnet18_cifar10/cifar10_test/1.0/preresnet18_cifar10.pth.tar --id-noise 0.2 --ood-noise 0.2

Note that a higher linear separation will be observed if the projection size is higher. We use 128 in the paper.
For higher separations you could visualize using [Umap](umap-learn.readthedocs.io) to map to 2D before reapplying L2 normalization.

#Retreiving noisy samples using OPTICS on unsupervised contrastive features
## Phase 1 - learn the unsupervised features
CUDA_VISIBLE_DEVICES=0 python main_unsup.py --dataset cifar100 --ood-noise 0.2 --id-noise 0.0 --epochs 2000 --batch-size 256 --net preresnet18 --lr 0.01 --steps 1000 1500 --seed 1 --exp-name cifar100_test_unsup --proj-size 128 --mixup

Note that there is no point in adding in-distribution noise here since the algorithm is unsupervised

## Phase 2 - extract the clusters
CUDA_VISIBLE_DEVICES=0 python evaluate_clust.py --weights checkpoints/preresnet18_cifar100/cifar100_test_unsup/1.0/preresnet18_cifar100.pth.tar --ood-noise 0.2 --id-noise 0.2

Both the noisy label assignment file and the ood clusters will be stored in noise_files/{dataset}/ (created automatically if it does not exist)

## Phase 3 - train the noise robust algorithm
CUDA_VISIBLE_DEVICES=0 python main_sup.py --dataset cifar100 --ood-noise 0.2 --id-noise 0.2 --epochs 100 --batch-size 256 --net preresnet18 --lr 0.1 --steps 50 80 --seed 1 --exp-name cifar100_sup_test --warmup 30 --proj-size 128 --mixup --cont --ts --noise-file noise_files/cifar100/clean_noisy_cifar100_0.2_0.2.pth.tar


The file train.sh lists commands to run the algorithm
