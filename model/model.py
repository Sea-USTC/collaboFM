from collaboFM import clip
import torch.nn as nn
import torch
from collaboFM.model.resnetcifar import ResNet18_cifar10, ResNet50_cifar10,ResNet18_mnist,ResNet18_cifar10_align
import numpy as np

class orthogonal(nn.Module):
    def __init__(self,feat_dim=128,class_num=10):
        super(orthogonal,self).__init__()
        proxy_dim=np.ceil(class_num / 2).astype(int)
        self.encode=nn.Linear(feat_dim,proxy_dim)
        vec = np.identity(proxy_dim)
        vec = np.vstack((vec, -vec))
        vec=torch.tensor(vec,dtype=torch.float)
        self.proxy=nn.Parameter(vec)
        self.proxy.requires_grad=False
    def forward(self,feature):
        hidden=self.encode(feature)
        output=torch.mm(hidden,self.proxy.T)
        return output

def no_meaning(x):
    return x

def l2norm(x):
    x=nn.functional.normalize(x,p=2,dim=1)
    return x

class l2noetf(nn.Module):
    def __init__(self,feat_dim,class_num):
        super(l2noetf,self).__init__()
        self.model=nn.Linear(feat_dim,class_num)
    def forward(self,x):
        x=nn.functional.normalize(x,p=2,dim=1)
        x=self.model(x)
        return x

class l2CLS(nn.Module):
    def __init__(self,feat_dim,class_num):
        super(l2CLS,self).__init__()
        self.model=nn.Linear(feat_dim,class_num)
    def forward(self,x):
        x=nn.functional.normalize(x,p=2,dim=1)
        x=self.model(x)
        return x

class cell(nn.Module):    
    def __init__(self,module_type="linear",param_dict=None):
        super(cell,self).__init__()
        if module_type=="linear":
            self.proxy=nn.Linear(param_dict["in_dim"],param_dict["out_dim"])
        elif module_type=="orthogonal":
            self.proxy=orthogonal(feat_dim=param_dict["feat_dim"],class_num=param_dict["class_num"])
        elif module_type =="etf":
            self.proxy=nn.BatchNorm1d(param_dict["feat_dim"])
        elif module_type =="identity":
            self.proxy=no_meaning
        elif module_type=="l2":
            self.proxy=l2norm#cls_self undefined
        elif module_type=="norm":
            self.proxy=nn.BatchNorm1d(param_dict["feat_dim"])
    def forward(self,feature):
        return self.proxy(feature)

class SimpleCNN_header(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN_header, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        #self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # x = self.fc3(x)
        return x

class model_cifar(nn.Module):
    def __init__(self,model_name, encoder_list=["identity"],encoder_para_list=None, \
        head_list=["linear"],head_para_list={"in_dim":256,"out_dim":10}):
        super(model_cifar, self).__init__()
        if "resnet18" in model_name:
            basemodel = ResNet18_cifar10()
            self.backbone = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif "resnet50" in model_name:
            basemodel = ResNet50_cifar10()
            self.backbone = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif "cnn" in model_name:
            self.backbone = SimpleCNN_header(input_dim=(16 * 5 * 5), hidden_dims=[120, 84])
            num_ftrs = basemodel.fc.in_features
        #elif "clip" in model_name:
        self.encoder=nn.Sequential()
        for idx,module_name in enumerate(encoder_list):
            name="encoder_"+str(idx)
            self.encoder.add_module(name,cell(module_name,encoder_para_list))
        self.head=nn.Sequential()
        for idx,module_name in enumerate(head_list):
            name="head_"+str(idx)
            self.head.add_module(name,cell(module_name,head_para_list))
    def forward(self,x):
        h = self.backbone(x)
        
        h=h.squeeze()
        #print(h.shape)
        x=h
        x=self.encoder(x)
        y = self.head(x)
        return y
    def forward_with_feature(self, x):
        h = self.backbone(x)
        h=h.squeeze()
        x=h
        x=self.encoder(x)
        y = self.head(x)
        return h, x, y
    def forward_head(self, x):
        y = self.head(x)
        return y




class model_clip(nn.Module):
    def __init__(self,model_name):
        super(model_clip, self).__init__()
        if model_name=="clip_rn50x4":
            path="/mnt/workspace/zqfan/foundation/ckpt/RN50.pt"
        elif model_name=="clip_rn50x4":
            path="/mnt/workspace/zqfan/foundation/ckpt/RN50x4.pt"  
        elif model_name=="clip_rn50x16":
            path="/mnt/workspace/zqfan/foundation/ckpt/RN50x16.pt"
        elif model_name=="clip_rn101":
            path="/mnt/workspace/zqfan/foundation/ckpt/RN101.pt"
        elif model_name=="clip_ViT16":
            path="/mnt/workspace/zqfan/foundation/ckpt/ViT-B-16.pt"
        elif model_name=="clip_ViT32":
            path="/mnt/workspace/zqfan/foundation/ckpt/ViT-B-32.pt"
        model, preprocess = clip.load(path, device="cpu")
        self.backbone=model
    # def encode_image(image):
    #     # from Image(png) to (b,3,224,224)
    #     img_vec = preprocess(image).unsqueeze(0)
    #     return img_vec
    # def encode_text(text):
    #     # from string(list) to (b,77)
    #     txt_vec=clip.tokenize(text)
        
    def forward(self,image,text):
        # image: (b,3,224,224)g
        # text:
        image_features = self.model.encode_image(image)
        text_features = self.model.encode_text(text)
        logits_per_image, logits_per_text = self.model(image, text)
        return logits_per_image,logits_per_text
