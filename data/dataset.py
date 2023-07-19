from torch.utils import data
from PIL import Image
import numpy as np
import torch
class dataset_base(data.Dataset):#需要继承data.Dataset
    def __init__(self,data_x,data_y):
        self.raw_data_x=data_x
        self.raw_data_y=data_y
        
    def __getitem__(self, index):
        data_x=self.raw_data_x[index]
        data_y=self.raw_data_y[index]
        if isinstance(data_x, str):
            data_x=np.array(Image.open(image_path))
        
        #data_x=torch.tensor(data_x)
        data_x=torch.tensor(data_x, dtype=torch.float32)
        data_y=torch.tensor(data_y, dtype=torch.int64)
        #self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        #print(data_x.shape)
        #print(data_x.shape)
        return (data_x,data_y)


    def __len__(self):

        
        return len(self.raw_data_x)