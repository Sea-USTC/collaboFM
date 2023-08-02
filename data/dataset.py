from torch.utils import data
from PIL import Image
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor
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

