from collaboFM import clip
import torch.nn as nn
import torch
from collaboFM.model.resnetcifar import ResNet18_cifar10, ResNet50_cifar10
import numpy as np
import timm
import logging

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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
    def __init__(self,module_type="linear",param_list=None):
        super(cell,self).__init__()
        if module_type=="linear":
            self.proxy=nn.Linear(param_list[1],param_list[3])
        elif module_type=="orthogonal":
            self.proxy=orthogonal(feat_dim=param_list[1],class_num=param_list[3])
        elif module_type =="etf":
            self.proxy=nn.BatchNorm1d(param_list[1])
        elif module_type =="identity":
            self.proxy=no_meaning
        elif module_type=="l2":
            self.proxy=l2norm#cls_self undefined
        elif module_type=="norm":
            self.proxy=nn.BatchNorm1d(param_list[1])
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
        head_list=["linear"],head_para_list=["in_dim",512,"out_dim",10]):
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
        #logger.debug(x)
        h = self.backbone(x)
        #logger.debug(h.shape)
        h=h.squeeze()
        x=h
        x=self.encoder(x)
        y = self.head(x)
        return h, x, y
    def forward_head(self, x):
        y = self.head(x)
        return y


class model_resnet(nn.Module):
    
    def __init__(self,model_name,encoder_list=["identity"],encoder_para_list=None, \
        head_list=["linear"],head_para_list=["in_dim",512,"out_dim",10],pretrained=False,num_classes=100):
        import torchvision.models as models
        '''
        from pytorch
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
        '''
        super(model_resnet, self).__init__()
        basemodel = None
        if "resnet18" in model_name:
            basemodel = models.resnet18(num_classes=num_classes, pretrained=False)
            param_dicts_path = '/mnt/workspace/colla_group/ckpt/resnet18-5c106cde.pth'
        elif "resnet34" in model_name:
            basemodel = models.resnet34(num_classes=num_classes, pretrained=False)
            param_dicts_path = '/mnt/workspace/colla_group/ckpt/resnet34-333f7ec4.pth'
        elif "resnet50" in model_name:
            basemodel = models.resnet50(num_classes=num_classes, pretrained=False)
            param_dicts_path = '/mnt/workspace/colla_group/ckpt/resnet50-19c8e357.pth'
        elif "resnet101" in model_name:
            basemodel = models.resnet101(num_classes=num_classes, pretrained=False)
            param_dicts_path = '/mnt/workspace/colla_group/ckpt/resnet101-5d3b4d8f.pth'
        elif "resnet152" in model_name:
            basemodel = models.resnet152(num_classes=num_classes, pretrained=False)
            param_dicts_path = '/mnt/workspace/colla_group/ckpt/resnet152-b121ed2d.pth'
        if pretrained:
            param_dicts=torch.load(param_dicts_path)
            basemodel.load_state_dict(param_dicts, strict=False)
        self.backbone = nn.Sequential(*list(basemodel.children())[:-1])
        self.encoder=nn.Sequential()
        for idx,module_name in enumerate(encoder_list):
            name="encoder_"+str(idx)
            self.encoder.add_module(name,cell(module_name,encoder_para_list))
        self.head=nn.Sequential()
        for idx,module_name in enumerate(head_list):
            name="head_"+str(idx)
            self.head.add_module(name,cell(module_name,head_para_list))
    def forward(self, x):
        h = self.backbone(x)
        h=h.squeeze()
        #print(h.shape)
        x=h
        x=self.encoder(x)
        y = self.head(x)
        return y
    def forward_with_feature(self, x):
        #logger.debug(x)
        h = self.backbone(x)
        #logger.debug(h.shape)
        h=h.squeeze()
        x=h
        x=self.encoder(x)
        y = self.head(x)
        return h, x, y

class model_vit(nn.Module):
    def __init__(self,model_name,encoder_list=["identity"],encoder_para_list=None, \
        head_list=["linear"],head_para_list=["in_dim",512,"out_dim",10],pretrained=False,num_classes=100):
        super(model_vit, self).__init__()
        #checkpoint_path="/mnt/workspace/colla_group/ckpt/"
        checkpoint_path = ""
        if model_name=="vittiny_16_224":
            basemodel=timm.create_model('vit_tiny_patch16_224', pretrained=pretrained, drop_path_rate=0.1,num_classes=head_para_list[3],checkpoint_path=checkpoint_path)
        if model_name=="vitsmall_16_224":
            basemodel=timm.create_model('vit_small_patch16_224', pretrained=pretrained, drop_path_rate=0.1,num_classes=head_para_list[3],checkpoint_path=checkpoint_path)
        if model_name=="vitbase_16_224":
            basemodel=timm.create_model('vit_base_patch16_224', pretrained=pretrained, drop_path_rate=0.1,num_classes=head_para_list[3],checkpoint_path=checkpoint_path)
        if model_name=="vitbase_32_224":
            basemodel=timm.create_model('vit_base_patch32_224', pretrained=pretrained, drop_path_rate=0.1,num_classes=head_para_list[3],checkpoint_path=checkpoint_path)
        if model_name=="vitlarge_16_224":
            basemodel=timm.create_model('vit_large_patch16_224', pretrained=pretrained, drop_path_rate=0.1,num_classes=head_para_list[3],checkpoint_path=checkpoint_path)
        if model_name=="vitlarge_32_224":
            basemodel=timm.create_model('vit_large_patch32_224', pretrained=pretrained, drop_path_rate=0.1,num_classes=head_para_list[3],checkpoint_path=checkpoint_path)
        self.backbone = basemodel
        self.encoder=nn.Sequential()
        for idx,module_name in enumerate(encoder_list):
            name="encoder_"+str(idx)
            self.encoder.add_module(name,cell(module_name,encoder_para_list))
        self.head=nn.Sequential()
        for idx,module_name in enumerate(head_list):
            name="head_"+str(idx)
            self.head.add_module(name,cell(module_name,head_para_list))

    def forward(self, x):
        h = self.backbone.forward_features(x)
        #print(h.shape)
        x = h
        x = self.backbone.forward_head(x,pre_logits=True)
        x = self.encoder(x)
        y = self.head(x)
        return y
    def forward_with_feature(self, x):
        h = self.backbone.forward_features(x)
        #logger.info(h.shape)
        x=h
        x=self.backbone.forward_head(x,pre_logits=True)
        x=self.encoder(x)
        #logger.info(x.shape)
        return h, x



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
