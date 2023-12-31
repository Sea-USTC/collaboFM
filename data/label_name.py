import os 

def get_label_name(cfg, dataset):
    if dataset == "cifar10":
        return cifar10_label()
    elif dataset == "cifar100":
        pass
    elif dataset == "caltech101":
        return caltech101_label(cfg)
    elif dataset == "food101":
        return food_label(cfg)


def cifar10_label():
    label_name = ["a photo of airplane",
                  "a photo of automobile",
                  "a photo of bird",
                  "a photo of cat",
                  "a photo of deer",
                  "a photo of dog",
                  "a photo of frog",
                  "a photo of horse",
                  "a photo of ship",
                  "a photo of truck"
                ]
    return label_name

def caltech101_label(cfg):
    from torchvision.datasets import Caltech101
    mydataset = Caltech101(root=cfg.data.root, target_type="category")
    label_name = mydataset.categories
    for i, label in enumerate(label_name):
        if "_" in label:
            label = label.replace("_"," ")
        label_name[i] = "A photo of "+label 
    # print(label_name)
    return label_name

def food_label(cfg):
    dataset_dir="/mnt/workspace/colla_group/data/food-101/"
    label_name = []
    with open(os.path.join(dataset_dir,"meta/classes.txt"),"r") as f1:
        f1 = f1.read().split('\n')
        for line in f1:
            label = line
            if "_" in line:
                label = label.replace('_', ' ')
            if label != '':
                label_name.append("A photo of "+label)
    # print(label_name)
    return label_name
    