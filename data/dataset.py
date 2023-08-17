from torch.utils import data
from PIL import Image
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor
from torchvision import transforms
class dataset_base(data.Dataset):#需要继承data.Dataset
    def __init__(self,data_x,data_y):
        self.raw_data_x=data_x
        self.raw_data_y=data_y
        self.new_size=224
        
    def __getitem__(self, index):
        data_x=self.raw_data_x[index]
        data_y=self.raw_data_y[index]
        if isinstance(data_x, str):
            data_x=np.array(Image.open(data_x).convert("RGB").resize((self.new_size, self.new_size)))
        data_x=to_tensor(data_x)
        if data_x.shape[0] == 1:
            data_x=data_x.repeat(3,1,1)
        return (data_x,data_y)


    def __len__(self):
        return len(self.raw_data_x)

normalize4caltech = transforms.Normalize((0.485, 0.456, 0.406),
                                        (0.229, 0.224, 0.225))
normalize4caltech_new = transforms.Normalize((0.5477,0.5305,0.5034),
                                         (0.3151,0.3095,0.3225))
normalize4cifar10 = transforms.Normalize((0.4914,0.4822,0.4465),
                                         (0.2470,0.2435,0.2615))           
normalize4food = transforms.Normalize((0.5450, 0.4435, 0.3436),
                                      (0.2709, 0.2735, 0.2781))
normalize4clip = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                      (0.26862954, 0.26130258, 0.27577711))
# normalize4food = transforms.Normalize()
def get_mean_std(dataset, method="woclip"):
    if method in ["fedclip","cliptqn"]:
        return normalize4clip
    if dataset == "cifar10":
        return normalize4cifar10
    elif dataset == "caltech101":
        return normalize4caltech
    elif dataset == "food101":
        return normalize4food

def get_num_classes(dataset):
    if dataset == "cifar10":
        return 10
    elif dataset == "caltech101":
        return 101
    elif dataset == "food101":
        return 101
    