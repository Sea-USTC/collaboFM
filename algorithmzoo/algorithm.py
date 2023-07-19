from collaboFM.build import *
from torch.utils.data.dataloader import DataLoader
import logging
from collaboFM.algorithmzoo.fedavg import FedAvg
from collaboFM.algorithmzoo.collabo import collabo
from collaboFM.algorithmzoo.local import local_baseline
from collaboFM.data.label_name import get_label_name

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

gpus = [0, 1, 2, 3]
torch.cuda.set_device('cuda:{}'.format(gpus[0]))

class SERVER():
    def __init__(self):
        self.net=None
        self.train_ds=None
        self.test_ds=None
        self.train_dl=None
        self.test_dl=None
class Algorithm_Manager():
    def __init__(self,control_config,client_manager):
        self.cfg=control_config
        self.server=SERVER()
        if self.cfg.federate.generic_fl_eval:
            self.init_server_dataset()
            self.init_server_model()
        self.n_clients=self.cfg.federate.client_num
        self.training_epochs=self.cfg.train.local_update_steps
        self.clients_per_round=self.cfg.federate.sample_client_num
        self.n_rounds=self.cfg.federate.total_round_num
        self.client_selection=self.cfg.federate.sample_mode
        self.client_manager=client_manager
        self.train_x=None
        self.train_y=None
        self.test_x=None
        self.test_y=None
    def init_server_model(self):
        self.server.net= build_server_model(self.cfg)
    def init_server_dataset(self,):
        train_batchsize=self.cfg.train.batchsize
        test_batchsize=self.cfg.eval.batchsize
        num_workers=8
        #build_data only build one certain dataset, when there are several datasets, what's the behavior fo the server?
        self.train_x,self.train_y,self.test_x,self.test_y=build_data(self.cfg)

        self.server.train_ds=build_ds(self.train_x,self.train_y)
        self.server.train_dl=DataLoader(dataset=self.server.train_ds, batch_size=train_batchsize, \
            drop_last=True, shuffle=True,num_workers=num_workers)
        self.server.test_ds=build_ds(self.test_x,self.test_y)
        self.server.test_dl=DataLoader(dataset=self.server.test_ds, batch_size=test_batchsize, \
            drop_last=False, shuffle=False,num_workers=num_workers)        
    
    def simulation(self):
        method = self.cfg.federate.method.lower()
        if method=="fedavg":
            self.run_fl()
        elif method=="local":
            self.run_local()
        elif method=="collabo":
            self.run_with_clip()
    
    
    def run_fl(self):
        self.algorithm=FedAvg(self.cfg)
        self.algorithm.global_para=self.client_manager.clients[0].net.state_dict()
        training_sequence=build_training_sequence(self.n_clients,self.clients_per_round,self.n_rounds,self.client_selection)
        for round_idx in range(self.n_rounds):
            training_clients=training_sequence[round_idx]
            self.algorithm.broadcast([self.client_manager.clients[idx] for idx in training_clients])
            weights=[len(self.client_manager.clients[i].train_ds) for i in training_clients]

            for client_idx in training_clients:
                if(self.cfg.data.load_all_dataset):
                    this_train_ds=self.client_manager.clients[client_idx].train_ds
                    this_train_dl=self.client_manager.clients[client_idx].train_dl
                    this_test_ds=self.client_manager.clients[client_idx].test_ds
                    this_test_dl=self.client_manager.clients[client_idx].test_dl
                else:
                    this_train_ds,this_train_dl,this_test_ds,this_test_dl=\
                        self.client_manager.create_one_dataset(self.client_manager.clients[client_idx],\
                            train_batchsize=self.cfg.train.batchsize,\
                            test_batchsize=self.cfg.test.batchsize,num_workers=8)
                local_data_points = len(this_train_ds)
                net=self.client_manager.clients[client_idx].net

                criterion=build_criterion(self.cfg.criterion).cuda()
                optimizer=build_optimizer(net,self.cfg.train.optimizer)
                net.cuda()
                for epoch in range(self.training_epochs):
                    
                    net.train()
                    for batch_idx, (batch_x, batch_y) in enumerate(this_train_dl):
                        self.algorithm.update_client_iter(net,client_idx,batch_x,batch_y,criterion,optimizer)
                    net.eval()
                    loss,acc=self.evaluate(net,this_train_dl,criterion)#training loss
                    logger.info(f"--------------client #{client_idx}------------------")
                    logger.info(f"train acc:{acc} train loss:{loss}")
                    loss,acc=self.evaluate(net,this_test_dl,criterion)
                    logger.info(f"test acc:{acc} test loss:{loss}")

                net.to('cpu')
            self.algorithm.para_aggregate([self.client_manager.clients[idx] for idx in training_clients],weights)
            if self.cfg.federate.generic_fl_eval:
                self.algorithm.update_server(self.server)
                loss,acc=self.evaluate(self.server.net,round_idx)

    def run_local(self):
        self.algorithm=local_baseline(self.cfg)
        for dataset_name, client_list in self.client_manager.dataset2idx.items():
            for round_idx in range(self.n_rounds):
                logger.info(f"-------------Round #{round_idx} start---------------")
                for client_idx in client_list:
                    if(self.cfg.data.load_all_dataset):
                        this_train_dl=self.client_manager.clients[client_idx].train_dl
                        this_test_dl=self.client_manager.clients[client_idx].test_dl
                    else:
                        this_train_ds,this_train_dl,this_test_ds,this_test_dl=\
                            self.client_manager.create_one_dataset(self.client_manager.clients[client_idx],\
                                train_batchsize=self.cfg.train.batchsize,\
                                test_batchsize=self.cfg.test.batchsize,num_workers=8)
                    net=self.client_manager.clients[client_idx].net

                    criterion=build_criterion(self.cfg.criterion).cuda()
                    optimizer=build_optimizer(net,self.cfg.train.optimizer)
                    net=nn.DataParallel(net.to(f"cuda:{gpus[0]}"), device_ids=gpus, output_device=gpus[0])
                    for epoch in range(self.training_epochs):                    
                        net.train()
                        for batch_idx, (batch_x, batch_y) in enumerate(this_train_dl):
                            self.algorithm.update_client_iter(net,client_idx,batch_x,batch_y,criterion,optimizer)
                        net.eval()
                        loss,acc=self.evaluate(net,this_train_dl,criterion)#training loss
                        logger.info(f"--------------client #{client_idx}------------------")
                        logger.info(f"train acc:{acc} train loss:{loss}")
                        loss,acc=self.evaluate(net,this_test_dl,criterion)
                        logger.info(f"test acc:{acc} test loss:{loss}")


    def run_with_clip(self):
        self.algorithm=collabo(self.cfg)
        import collaboFM.clip as clip
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.server.net, preprocess = clip.load("/mnt/workspace/colla_group/ViT-B-32.pt", device=device)
        for dataset_name, client_list in self.client_manager.dataset2idx.items():
            label_name = get_label_name(self.cfg, dataset_name)
            label2token=clip.tokenize(label_name).cuda()
            label2repre=self.server.net.encode_text(label2token).to('cpu')
            for round_idx in range(self.cfg.tqn_train.key_train_round):
                logger.info(f"##########Round #{round_idx} ################")
                for client_idx in client_list:
                    logger.info(f"--------------client #{client_idx}------------------\n"
                                f"              train start!")
                    if(self.cfg.data.load_all_dataset):
                        this_train_dl=self.client_manager.clients[client_idx].train_dl
                        this_test_dl=self.client_manager.clients[client_idx].test_dl
                    else:
                        this_train_ds,this_train_dl,this_test_ds,this_test_dl=\
                            self.client_manager.create_one_dataset(self.client_manager.clients[client_idx],\
                                train_batchsize=self.cfg.train.batchsize,\
                                test_batchsize=self.cfg.test.batchsize,num_workers=8)
                    net=self.client_manager.clients[client_idx].net

                    criterion=build_criterion(self.cfg.criterion).cuda()
                    optimizer=build_optimizer(net,self.cfg.train.optimizer)
                    net.cuda()
                    for epoch in range(self.training_epochs):
                        net.train()
                        for batch_idx, (batch_x, batch_y) in enumerate(this_train_dl):
                            self.algorithm.update_client_iter(net,client_idx,batch_x,batch_y,criterion,optimizer,label2repre)
            model_list=[]
            dump_path="/mnt/workspace/lisiyi/result/backbone.pt"
            for client_idx in client_list:
                self.client_manager.clients[client_idx].net.eval()
                model_list[client_idx]=self.client_manager.clients[client_idx].net.state_dict()
            torch.save(model_list,dump_path)
            for round_idx in range(self.n_rounds):
                logger.info(f"#######TQN Round #{round_idx} ################")
                for client_idx in client_list:
                    logger.info(f"--------------client #{client_idx}------------------\n"
                                f"              train start!")
                    if(self.cfg.data.load_all_dataset):
                        this_train_dl=self.client_manager.clients[client_idx].train_dl
                        this_test_dl=self.client_manager.clients[client_idx].test_dl
                    else:
                        this_train_ds,this_train_dl,this_test_ds,this_test_dl=\
                            self.client_manager.create_one_dataset(self.client_manager.clients[client_idx],\
                                train_batchsize=self.cfg.train.batchsize,\
                                test_batchsize=self.cfg.test.batchsize,num_workers=8)
                    net=self.client_manager.clients[client_idx].net
                    net.cuda()  
                    for epoch in range(self.training_epochs):                    
                        self.algorithm.tqn.train()
                        for batch_idx, (batch_x, batch_y) in enumerate(this_train_dl):
                            self.algorithm.train_tqn_model(net,client_idx,batch_x,batch_y,label2repre)
                        self.algorithm.tqn.eval()
                        loss,acc=self.algorithm.evaluate(net,this_train_dl,label2repre)#training loss
                        logger.info(f"--------------client #{client_idx}------------------")
                        logger.info(f"train acc:{acc} train loss:{loss}")
                        loss,acc=self.algorithm.evaluate(net,this_test_dl,label2repre)
                        logger.info(f"test acc:{acc} test loss:{loss}")
    

     

