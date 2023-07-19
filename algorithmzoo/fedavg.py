import torchvision.transforms as transforms
import torch
from sklearn.metrics import accuracy_score
import numpy as np

class FedAvg():
    def __init__(self,control_config):
        self.cfg=control_config
        self.n_clients=self.cfg.federate.client_num        
        self.clients_per_round=self.cfg.federate.sample_client_num
        self.n_rounds=self.cfg.federate.total_round_num
        self.epochs=self.cfg.train.local_update_steps
        self.global_para={}


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
        
    
    
    def para_aggregate(self,clients,weights):
        for idx in range(len(clients)):
            net_para = clients[idx].net.state_dict()
            weight = weights[idx] / sum(weights)
            if idx == 0:
                for key in net_para:
                    self.global_para[key] = net_para[key] * weight
            else:
                for key in net_para:
                    self.global_para[key] += net_para[key] * weight

    def update_server(self, server):
        server.net.load_state_dict(self.global_para)
        
    def broadcast(self,clients):
        for client in clients:
            client.net.load_state_dict(self.global_para)

    def evaluate(self,net,dataloader,criterion):
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2470, 0.2435, 0.2615))
        transform_test = transforms.Compose([
                transforms.ToTensor(),
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