import torchvision.transforms as transforms
from torch.nn import CrossEntropyLoss
from torch.nn.functional import one_hot

import torch
from sklearn.metrics import accuracy_score
import numpy as np
from collaboFM.model.tqn_model import TQN_Model
import logging

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class collabo():
    def __init__(self,control_config):
        self.cfg=control_config
        self.n_clients=self.cfg.federate.client_num        
        self.clients_per_round=self.cfg.federate.sample_client_num
        self.n_rounds=self.cfg.federate.total_round_num
        self.epochs=self.cfg.train.local_update_steps
        self.tqn=TQN_Model(embed_dim=512, class_num=2)


    def update_client_iter(self,net,net_id,batch_x,batch_y,criterion,optimizer,label2repre):
        #print(batch_x.shape)
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2470, 0.2435, 0.2615))
        
        transform_train = transforms.Compose([
                normalize
            ])
        batch_x.requires_grad = False
        batch_y.requires_grad = False
        batch_x = batch_x.cuda()
        batch_x = transform_train(batch_x)
        N=batch_y.shape[0]
        batch_y_repre=torch.vstack([label2repre[[batch_y[i]]] for i in range(N)])
        batch_x = batch_x.detach().cuda()
        batch_y = batch_y_repre.detach().cuda()

        optimizer.zero_grad()
        #target = target.long()
        out = net.forward_with_feature(batch_x)[1]
        loss = criterion(out, batch_y)
        
        loss.backward()
        #nn.utils.clip_grad_norm_(net.parameters(),max_norm=5,norm_type=2)
        optimizer.step()
    
    def train_tqn_model(self,net,net_id,batch_x,batch_y,label2repre):
        #print(batch_x.shape)
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2470, 0.2435, 0.2615))
        
        transform_train = transforms.Compose([
                normalize
            ])
        batch_x.requires_grad = False
        batch_y.requires_grad = False
        batch_x=batch_x.cuda()
        batch_x=transform_train(batch_x)
        batch_y=batch_y.cuda()


        with torch.no_grad():
            image_repre = net.forward_with_feature(batch_x)[1]
        self.tqn.cuda()
        from torch.optim import Adam

        optimizer = Adam(filter(lambda p: p.requires_grad, self.tqn.parameters()),lr=0.001)
        criterion = CrossEntropyLoss().cuda()
        optimizer.zero_grad()  
        image_repre=image_repre.unsqueeze(1).detach().cuda()
        label2repre=label2repre.float().detach().cuda()
        out = self.tqn(image_repre, label2repre)##magic_num 1
        batch_y=one_hot(batch_y, num_classes=10)
        batch_y=one_hot(batch_y, num_classes=2).float().detach().cuda()
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer.step()
            
    def evaluate(self,net,dataloader, label2repre):
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2470, 0.2435, 0.2615))
        transform_test = transforms.Compose([
                normalize
            ])
        #net=client.net
        
        criterion = CrossEntropyLoss().cuda()
        true_labels_list, pred_labels_list = np.array([]), np.array([])
        loss_collector = []

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(dataloader):
                x=x.cuda()
                x=transform_test(x)
                image_repre = net.forward_with_feature(x)[1]
                image_repre=image_repre.unsqueeze(1).detach().cuda()
                label2repre=label2repre.float().detach().cuda()
                self.tqn.cuda()
                out = self.tqn(image_repre, label2repre)
                y_true=target
                target=one_hot(target, num_classes=10)
                target=one_hot(target, num_classes=2).float().detach().cuda()
                loss = criterion(out, target)
                _, pred_label = torch.max(out.data[:,:,1], 1)
                loss_collector.append(loss.item())
                pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                true_labels_list = np.append(true_labels_list, y_true.data.cpu().numpy())
                #logger.info(len(pred_labels_list)==len(true_labels_list))
            avg_loss = sum(loss_collector) / len(loss_collector)
            acc=accuracy_score(true_labels_list,pred_labels_list)

        return avg_loss,acc