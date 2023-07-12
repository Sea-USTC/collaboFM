from build import *
from sklearn.metrics import accuracy_score
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms

class SERVER():
    def __init__(self):
        self.net=None
        self.train_ds=None
        self.test_ds=None
        self.train_dl=None
        self.test_dl=None
class Algorithm_Manager():
    def __init__(self,basic_config,control_config,client_manager):
        self.basic_config=basic_config
        self.control_config=control_config
        
        self.server=SERVER()
        self.init_server_dataset()
        self.init_server_model()
        self.n_clients=control_config.n_clients
        self.training_epochs=control_config.training_epochs
        self.clients_per_round=control_config.clients_per_round
        self.n_rounds=control_config.n_rounds
        self.client_selection=control_config.client_selection
        
        self.build_algorithm()
        #print(self.algorithm)
        
        self.client_manager=client_manager
        #self.server_manager=server_manager
        self.train_x=None
        self.train_y=None
        self.test_x=None
        self.test_y=None
    def init_server_model(self):
        self.server.net= build_client_model(self.basic_config,self.control_config)
    def init_server_dataset(self,):
        data_name=self.control_config.dataset
        train_batchsize=self.control_config.train_batchsize
        test_batchsize=self.control_config.test_batchsize
        num_workers=8
        #build_data only build one certain dataset, when there are several datasets, what's the behavior fo the server?
        self.train_x,self.train_y,self.test_x,self.test_y=build_data(self.basic_config,self.control_config)

        self.server.train_ds=build_ds(self.train_x,self.train_y)
        self.server.train_dl=DataLoader(dataset=self.server.train_ds, batch_size=train_batchsize, \
            drop_last=True, shuffle=True,num_workers=num_workers)
        self.server.test_ds=build_ds(self.test_x,self.test_y)
        self.server.test_dl=DataLoader(dataset=self.server.test_ds, batch_size=test_batchsize, \
            drop_last=False, shuffle=False,num_workers=num_workers)
        

    def run(self):
        #print(self.algorithm)
        training_sequence=build_training_sequence(self.n_clients,self.clients_per_round,self.n_rounds,self.client_selection)
        for round_idx in range(self.clients_per_round):
            training_clients=training_sequence[round_idx]
            self.algorithm.broadcast([self.client_manager.clients[idx] for idx in training_clients],self.server)
            weights=[0 for i in range(len(training_clients))]

            for client_idx in training_clients:
                if(self.client_manager.clients[client_idx].train_dl):
                    this_train_ds=self.client_manager.clients[client_idx].train_ds
                    this_train_dl=self.client_manager.clients[client_idx].train_dl
                    this_test_ds=self.client_manager.clients[client_idx].test_ds
                    this_test_dl=self.client_manager.clients[client_idx].test_dl
                else:
                    this_train_ds,this_train_dl,this_test_ds,this_test_dl=\
                        self.client_manager.build_one_dataset(self.client_manager.clients[client_idx],\
                            train_batchsize=self.control_config.train_batchsize,\
                            test_batchsize=self.control_config.test_batchsize,num_workers=8)
                local_data_points = len(this_train_ds)
                net=self.client_manager.clients[client_idx].net

                #train_ds=
                #dl=
                criterion=build_criterion(self.control_config.criterion_dicts).cuda()
                optimizer=build_optimizer(net,self.control_config.optimizer_dicts)
                net.cuda()
                for epoch in range(self.training_epochs):
                    
                    net.train()
                    for batch_idx, (batch_x, batch_y) in enumerate(this_train_dl):
                        self.algorithm.update_client_iter(net,client_idx,batch_x,batch_y,criterion,optimizer)
                    net.eval()
                    loss,acc=self.evaluate(net,this_train_dl,criterion)
                    print(loss,acc)

                net.to('cpu')
            self.algorithm.update_server(self.client_manager.clients[training_clients],self.server,weights)
            loss,acc=self.evaluate(self.server.net,round_idx)
            
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
        #print(self.control_config.method)
        if self.control_config.method=="FedAvg":
            self.algorithm=FedAvg(self.basic_config,self.control_config)
        

class FedAvg():
    def __init__(self,basic_config,control_config):
        self.basic_config=basic_config
        self.control_config=control_config
        self.n_clients=control_config.n_clients
        self.clients_per_round=control_config.clients_per_round
        self.n_rounds=control_config.n_rounds

        self.epochs=self.control_config.training_epochs


    def update_client_iter(self,net,net_id,batch_x,batch_y,criterion,optimizer):
        #print(batch_x.shape)
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2470, 0.2435, 0.2615))
        
        transform_train = transforms.Compose([
                normalize
            ])
        batch_x=transform_train(batch_x)

        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

        optimizer.zero_grad()
        batch_x.requires_grad = False
        batch_y.requires_grad = False
        #target = target.long()

        out = net(batch_x)

        loss = criterion(out, batch_y)
        
        loss.backward()
        #nn.utils.clip_grad_norm_(net.parameters(),max_norm=5,norm_type=2)
        optimizer.step()
        
    
    
    def update_server(self,clients,server,weights):
        
        for idx in range(len(clients)):
            net_para = clients[idx].net.state_dict()
            weight = weights[idx] / sum(weights)
            if idx == 0:
                for key in net_para:
                    global_para[key] = net_para[key] * weight
            else:
                for key in net_para:
                    global_para[key] += net_para[key] * weight
        server.net.load_state_dict(global_para)
        
    def broadcast(self,clients,server):
        global_para = server.net.state_dict()
        for client in clients:
            client.net.load_state_dict(global_para)
        
        

