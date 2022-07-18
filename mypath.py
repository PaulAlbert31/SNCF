class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == "cifar10":
            return "/path/to/cifar10/"
        elif dataset == "cifar100":
            return "/path/to/cifar100/"
        elif dataset == "imagenet32":
            return "/path/to/imagenet32/"
        elif dataset == 'webvision':
            return '/path/to/webvision/'
        elif dataset == 'miniimagenet_preset':
            return '/path/to/miniImagenet/miniimagenet_web/dataset/mini-imagenet/'
        elif dataset == 'places':
            return '/path/to/Places/test_256/'
        else:
            raise NotImplementedError('Dataset {} not available.'.format(dataset))
        
