from collaboFM.build import *
from sklearn.metrics import accuracy_score
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
import logging
from collaboFM.algorithmzoo.fedavg import FedAvg
from collaboFM.algorithmzoo.collabo import collabo

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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

        self.build_algorithm()
        #print(self.algorithm)
        
        self.client_manager=client_manager
        #self.server_manager=server_manager
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
    def run(self):
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


    def run_with_clip(self):
        training_sequence=build_training_sequence(self.n_clients,self.clients_per_round,self.n_rounds,self.client_selection)
        self.algorithm.global_para=self.client_manager.clients[0].net.state_dict()
        self.algorithm.broadcast([self.client_manager.clients[idx] for idx in range(self.n_clients)])
        import clip
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.server.net, preprocess = clip.load("ViT-B/32", device=device)
        
        for round_idx in range(self.clients_per_round):
            training_clients=training_sequence[round_idx]
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
                net=self.client_manager.clients[client_idx].net

                criterion=build_criterion(self.cfg.model.criterion).cuda()
                optimizer=build_optimizer(net,self.cfg.train.optimizer)
                net.cuda()
                for epoch in range(self.training_epochs):
                    
                    net.train()
                    for batch_idx, (batch_x, batch_y) in enumerate(this_train_dl):
                        self.algorithm.update_client_iter(net,client_idx,batch_x,batch_y,criterion,optimizer)
                    net.eval()
                    loss,acc=self.evaluate(net,this_train_dl,criterion)#training loss
                    logger.info(f"--------------client #{client_idx}------------------\n"
                                f"train acc:{acc} train loss:{loss}")
                    loss,acc=self.evaluate(net,this_test_dl,criterion)
                    logger.info(f"test acc:{acc} test loss:{loss}\n"
                                f"------------------------------------------------")
                net.to('cpu')
            

                
    def evaluate(self,net,dataloader,criterion):
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2470, 0.2435, 0.2615))
        
        transform_test = transforms.Compose([
                normalize
            ])
        #net=client.net
        
        criterion.cuda()
        true_labels_list, pred_labels_list = np.array([]), np.array([])
        loss_collector = []

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(dataloader):
                x=transform_test(x)
                x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
                out = net(x)
                loss = criterion(out, target)
                _, pred_label = torch.max(out.data, 1)
                loss_collector.append(loss.item())
                #total += x.data.size()[0]
                #correct += (pred_label == target.data).sum().item()
                pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
            avg_loss = sum(loss_collector) / len(loss_collector)
            acc=accuracy_score(true_labels_list,pred_labels_list)

        return avg_loss,acc

    def build_algorithm(self,):
        #print(self.cfg.method)
        if self.cfg.federate.method=="FedAvg":
            self.algorithm=FedAvg(self.cfg)
        
     

