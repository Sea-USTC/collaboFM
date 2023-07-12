from build import build_client_model,build_ds,build_data,build_split
import numpy as np
from torch.utils.data.dataloader import DataLoader
class CLIENT():
    def __init__(self):
        self.id=None
        self.net=None
        self.train_idx=None
        self.test_idx=None
        self.train_ds=None
        self.test_ds=None
        self.train_dl=None
        self.test_dl=None

class Client_Manager_base():
    def __init__(self,control_config,basic_config):
        
        ## commonly used params
        self.basic_config=basic_config
        self.control_config=control_config
        self.n_clients=self.control_config.n_clients
        
        ## init clients
        self.clients={}
        for i in range(self.n_clients):
            self.clients[i]=CLIENT()
            self.clients[i].id=i
        
        ## overall data information of all clients
        self.train_ds_all=None
        self.train_dl_all=None
        self.test_ds_all=None
        self.test_dl_all=None

        self.train_x=None
        self.train_y=None
        self.test_x=None
        self.test_y=None
        self.train_idx_dict=None
        self.test_idx_dict=None

    # build data(train_x,train_y,test_x,test_y), build split, build index of all clients
    def create_raw_data(self):
        self.train_x,self.train_y,self.test_x,self.test_y=build_data(self.basic_config,self.control_config)
        
    def create_fed_split_index(self):    
        self.train_idx_dict,self.test_idx_dict=build_split(self.train_y,self.test_y,self.basic_config,self.control_config)
        for i in range(self.n_clients):
            self.clients[i].train_idx=self.train_idx_dict[i]
            self.clients[i].test_idx=self.test_idx_dict[i]

    def create_multi_task_level_index(self):
        pass
    
    # load datasets of all clients in RAM
    def create_all_datasets(self,train_batchsize=128,test_batchsize=128,num_workers=8):
        
        for client_idx in range(self.n_clients):
            ## build dataset of all clients
            train_idx,test_idx=self.train_idx_dict[client_idx],self.test_idx_dict[client_idx]
            train_x,train_y=np.array(self.train_x)[train_idx],np.array(self.train_y)[train_idx]
            test_x,test_y=np.array(self.test_x)[test_idx],np.array(self.test_y)[test_idx]

            self.clients[client_idx].train_ds=build_ds(train_x,train_y)
            self.clients[client_idx].train_dl=DataLoader(dataset=self.clients[client_idx].train_ds, batch_size=train_batchsize, \
                drop_last=True, shuffle=True,num_workers=num_workers)

            self.clients[client_idx].test_ds=build_ds(test_x,test_y)
            self.clients[client_idx].test_dl=DataLoader(dataset=self.clients[client_idx].test_ds, batch_size=test_batchsize, \
                drop_last=False, shuffle=False,num_workers=num_workers)
            
    
    # load only one recent training dataset in RAM and replace it when training next client
    def create_one_dataset(self,client,train_batchsize=128,test_batchsize=128,num_workers=8):
        train_idx,test_idx=client.train_idx,client.test_idx
        #print(self.train_x)
        #print(self.train_y)
        train_x,train_y=np.array(self.train_x)[train_idx],np.array(self.train_y)[train_idx]
        test_x,test_y=np.array(self.test_x)[test_idx],np.array(self.test_y)[test_idx]

        this_train_ds=build_ds(train_x,train_y)
        this_train_dl=DataLoader(dataset=this_train_ds, batch_size=train_batchsize, \
            drop_last=True, shuffle=True,num_workers=num_workers)
        this_test_ds=build_ds(test_x,test_y)
        this_test_dl=DataLoader(dataset=this_test_ds, batch_size=test_batchsize, \
            drop_last=False, shuffle=False,num_workers=num_workers)
        return this_train_ds,this_train_dl,this_test_ds,this_test_dl
    
    # load the union dataset and dataloader of all clients in RAM
    def create_whole_dataset(self,train_batchsize=128,test_batchsize=128,num_workers=8):
        self.train_ds_all=build_ds(self.train_x,self.train_y,data_idx=None)
        self.train_dl_all=DataLoader(dataset=self.train_ds_all, batch_size=train_batchsize,\
             drop_last=False, shuffle=False,num_workers=num_workers)
        self.test_ds_all=build_ds(self.test_x,self.test_y,data_idx=None)
        self.test_dl_all=DataLoader(dataset=self.test_ds_all, batch_size=test_batchsize,\
             drop_last=False, shuffle=False,num_workers=num_workers)
    
    # load models of all clients in RAM
    def create_all_models(self):
        for i in range(self.n_clients):
            self.clients[i].net=build_client_model(self.basic_config,self.control_config)

    # load a single model used to train now in RAM and replace it when training next client
    def create_one_model(self):
        this_model=build_client_model(self.basic_config,self.control_config)
        return this_model










