##To visualize the linear separation in 2D on cifar 10
CUDA_VISIBLE_DEVICES=0 python main_unsup.py --dataset cifar10 --ood-noise 0.2 --id-noise 0.2 --epochs 2000 --batch-size 256 --net preresnet18 --lr 0.01 --steps 1000 1500 --seed 1 --exp-name cifar10_test --proj-size 2 --mixup

#Visualize python plot_sphere.py weights ood_noise id_noise
CUDA_VISIBLE_DEVICES=0 python plot_sphere.py --weights checkpoints/preresnet18_cifar10/cifar10_test/1.0/preresnet18_cifar10.pth.tar --id-noise 0.2 --ood-noise 0.2

##Example on cifar100 with 20% OOD noise from imagenet32, 20% ID uniform noise
#Learn the unsupervised features, id noise set to 0.2 but 0.0 would be equivalent for unsup learning
CUDA_VISIBLE_DEVICES=0 python main_unsup.py --dataset cifar100 --ood-noise 0.2 --id-noise 0.2 --epochs 2000 --batch-size 256 --net preresnet18 --lr 0.01 --steps 1000 1500 --seed 1 --exp-name cifar100_test --proj-size 128 --mixup

#Compute the embedding and extract the clusters from the unsupervised features. python evaluate_clust.py weights ood_noise id_noise
CUDA_VISIBLE_DEVICES=0 python evaluate_clust.py --weights checkpoints/preresnet18_cifar100/cifar100_test/1.0/preresnet18_cifar100.pth.tar --ood-noise 0.2 --id-noise 0.2

#Supervised stage
CUDA_VISIBLE_DEVICES=0 python main_sup.py --dataset cifar100 --ood-noise 0.2 --id-noise 0.2 --epochs 100 --batch-size 256 --net preresnet18 --lr 0.1 --steps 50 80 --seed 1 --exp-name cifar100_sup_test --warmup 30 --proj-size 128 --mixup --cont --ts --noise-file noise_files/cifar100/clean_noisy_cifar100_0.2_0.2.pth.tar

##Same but with places as perturbation
CUDA_VISIBLE_DEVICES=0 python main_unsup.py --dataset cifar100 --ood-noise 0.2 --id-noise 0.2 --epochs 2000 --batch-size 256 --net preresnet18 --lr 0.01 --steps 1000 1500 --seed 1 --exp-name cifar100_test_places --proj-size 128 --mixup --corruption places

#Include places in weights name for evaluate_clust.py to choose the right corruption
CUDA_VISIBLE_DEVICES=0 python evaluate_clust.py --weights checkpoints/preresnet18_cifar100/cifar100_test_places/1.0/preresnet18_cifar100.pth.tar --ood-noise 0.2 --id-noise 0.2

CUDA_VISIBLE_DEVICES=0 python main_sup.py --dataset cifar100 --ood-noise 0.2 --id-noise 0.2 --epochs 100 --batch-size 256 --net preresnet18 --lr 0.1 --steps 50 80 --seed 1 --exp-name cifar100_sup_test_places --warmup 30 --proj-size 128 --mixup --cont --ts --corruption places --noise-file noise_files/cifar100/clean_noisy_cifar100_0.2_0.2_places.pth.tar
