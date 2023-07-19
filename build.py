#import clip
from collaboFM.model import *
from collaboFM.data.dataset import dataset_base
from torchvision.datasets import CIFAR10,CIFAR100
from collaboFM.model.model import *
from collaboFM.auxiliaries.partition import *
import torch.optim as optim
import torch.nn as nn
# def build_weights(train_y):
#     pass
#     return 0

def build_ds(data_x,data_y):
    return dataset_base(data_x, data_y)

def build_criterion(criterion_dicts):
    if criterion_dicts["type"]=="cross_entropy":
        return nn.CrossEntropyLoss()
    

def build_optimizer(net,optim_dicts):
    if optim_dicts["type"]=="SGD":
        # for num,para in enumerate(paras):
        #     print('number:',num)
        #     print(para)
        #     print('_____________________________')
        return optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=optim_dicts["lr"],\
             momentum=optim_dicts["momentum"], weight_decay=optim_dicts["weight_decay"])
    
def build_training_sequence(n_clients=10,clients_per_round=3,n_rounds=2,mode="random"):
    ### input:
    #n_clients: type:int,number of client
    #clients_per_round: type:int,number of clients training in each round
    #n_roundss: type:int, the number of total communication rounds
    if mode=="random":
        #need to fill
        training_sequence = [np.random.choice(n_clients,clients_per_round,replace=False).tolist() 
                            for _ in range(n_rounds)]
        
    ### output, list=[[0,1,3],[5,7,9]]the index list of training clients in each round
    return training_sequence

def build_server_model(cfg):
    pass

def build_client_model(client_id, cfg):
    if cfg.federate.use_hetero_model:# not empty means the model may be diversied
        cfg_dict=cfg.client_resource #rsrc key: dataset model encoder_list encoder_para_list head_list head_para_list
        model_name=getattr(cfg_dict, "backbone")[str(client_id)].lower()
    else:
        cfg_dict=cfg.model    
        model_name=getattr(cfg_dict, "backbone").lower()
    if "clip" in model_name:
        return model_clip(model_name=model_name)
    encoder_list=getattr(cfg_dict, "encoder_list")
    encoder_para_list=getattr(cfg_dict, "encoder_para_list")
    head_list=getattr(cfg_dict, "head_list")
    head_para_list=getattr(cfg_dict,"head_para_list")
    if cfg.federate.use_hetero_model:
        encoder_list=encoder_list[str(client_id)]
        encoder_para_list=encoder_para_list[str(client_id)]
        head_list=head_list[str(client_id)]
        head_para_list=head_para_list[str(client_id)]
    if "cifar" in model_name:
        return model_cifar(model_name=model_name, encoder_list=encoder_list, encoder_para_list=encoder_para_list, \
        head_list=head_list,head_para_list=head_para_list)
    elif "femnist" in model_name:
        pass



def build_data(cfg,data_name):
    
    # data_name: type:str "cifar10"
    # basic_config 是保存在本地默认极少修改的参数集合，例如basic_config.hidden_dim=256,说明模型中间层是256
    # control config 是训练过程中经常调整的参数集合，例如batchsize,epoch_per_round等
    if data_name=="cifar10":
        cifar10_train_ds = CIFAR10(root=cfg.data.root,train=True)#
        cifar10_test_ds = CIFAR10(root=cfg.data.root,train=False)#
        train_x, train_y = cifar10_train_ds.data, cifar10_train_ds.targets        
        test_x, test_y = cifar10_test_ds.data, cifar10_test_ds.targets
       
    elif data_name=="cifar100":
        # need to fill    
        pass
  
    elif data_name=="tiny-imagenet":
        # need to fill    
        pass
   
    elif data_name=="fashion-mnist":
        # need to fill    
        pass
    elif data_name=="mnist":
        # need to fill    
        pass
    elif data_name=="femnist":
        # need to fill    
        pass
    elif data_name=="svhn":
        # need to fill    
        pass
    elif data_name=="isic":
        # need to fill    
        pass
    elif data_name=="pacs":
        # need to fill    
        pass
    elif data_name=="officehome":
        # need to fill    
        pass
    elif data_name=="domainnet":
        # need to fill    
        pass
    elif data_name=="shakespeare":
        # need to fill    
        pass
    elif data_name=="celeba":
        # need to fill    
        pass
    elif data_name=="sent140":
        # need to fill    
        pass
    elif cfg.data_name=="chestxray":
        # need to fill    
        pass
    elif cfg.data_name=="mimic":
        # need to fill    
        pass
    elif cfg.data_name=="chexpert":
        # need to fill    
        pass

    ### output:
    # train_y,test_y is the list of the data label [0,1,1,1,2,5,7,6]
    # train_x,train_y is the list of image path(data_num), or the image data itself tensor(data_num,3,224,224)
    return train_x,train_y,test_x,test_y


def build_split(y_train,y_test,cfg,n_clients):
    if cfg.data.splitter=="dirichlet":
        return split_dirichlet(y_train,y_test,n_clients=n_clients,beta=cfg.data.splitter_args[0]["beta"])
    elif cfg.data.splitter=="class":
        return split_class(y_train,y_test,n_clients=n_clients,n_class=cfg.n_class)


def build_multi_task(dataset_list=["cifar10","cifar100","mnist"],model_list=["lenet","resnet18","simple-cnn"],n_client=3,split_mode_dict={"type":"dirichlet","beta":0.5}):
    
    pass
