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
        self.tqn=[TQN_Model(embed_dim=512, class_num=2) for i in range(self.n_clients)]


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
        batch_x = batch_x.detach().cuda()
        batch_y = batch_y.cuda()
        label2repre = label2repre.float().detach().cuda()
        optimizer.zero_grad()
        #target = target.long()
        #logger.debug(batch_x)
        out = net.forward_with_feature(batch_x)[1]
        #logger.debug(out)
        #logger.debug(batch_y)
        loss,angle = self.similarity(out, label2repre,batch_y,2)
        #logger.debug(loss.item())
        loss.backward()
        #nn.utils.clip_grad_norm_(net.parameters(),max_norm=5,norm_type=2)
        optimizer.step()
        return N, loss, angle

    def similarity(self, out, label2repre,target,flag="cluster", tau=20,norm=False):
        loss=torch.tensor([0.0]).cuda()
        angle=[]
        if norm:
            out = torch.nn.functional.normalize(out, dim=-1)
            label2repre = torch.nn.functional.normalize(label2repre, dim=-1)
        if flag == "KADpro":
            label_y = []
            for i in range(len(target)):
                label_y.append(label2repre[target[i]])
            label_y=torch.vstack(label_y).t()
            z = torch.mm(out,label_y)/tau
            z_image = torch.nn.functional.log_softmax(z,dim=1)
            z_text = torch.nn.functional.log_softmax(z,dim=0)
            for i in range(out.shape[0]):
                loss-=z_image[i,i]
                loss-=z_text[i,i]
        if flag == "l2distance":
            label_y = []
            for i in range(len(target)):
                label_y.append(label2repre[target[i]])
            label_y=torch.vstack(label_y)
            pdist = torch.nn.PairwiseDistance()
            loss = torch.sum(pdist(label_y, out))
        for tgt in torch.unique(target): 
            tgt_idx = np.where((target==tgt).cpu())[0]
            #logger.info(tgt_idx)
                #label2repre: [class_num, embed_dim]
                #out: [batchsize, embed_dim]   
                #target: [batchsize]           
            if "KAD" in flag:
                sample_set=label2repre.t()/tau         
                loss-=torch.sum(torch.nn.functional.log_softmax(torch.mm(out[tgt_idx,:],sample_set),dim=1)[:,tgt])  
                non_tgt_idx = np.where((target!=tgt).cpu())[0]
                for i in tgt_idx:
                    z_set = torch.vstack([out[i,:], out[non_tgt_idx,:]]).t()/tau
                    loss-=torch.nn.functional.log_softmax(torch.mm(label2repre[tgt].reshape(1,-1),z_set)/tau,dim=1)[0,0]
            
            if flag in ["KAD+","cluster","cluster-"]:   
                z_set = torch.vstack([out[tgt_idx,:],label2repre[tgt]])
                #logger.info(z_set.shape)
                N = z_set.shape[0]
                conv=torch.mm(z_set, z_set.t())/tau
                conv=torch.nn.functional.log_softmax(conv,dim=1)
                if flag=="cluster":
                    #logger.info(conv)
                    loss-=torch.sum(conv)/N
                    logger.info(torch.sum(conv).item())
                    #logger.info(loss.data)
                elif flag=="cluster-":
                    loss-=torch.sum(conv[:,-1])
                else :
                    loss-=8*torch.sum(conv)/N
            # for i in tgt_idx:
            #     angle.append((torch.dot(out[i,:],label2repre[tgt,:])/torch.linalg.vector_norm(out[i,])/torch.linalg.vector_norm(label2repre[tgt,])).item())
            for i in tgt_idx:
                angle.append(torch.linalg.vector_norm(out[i,]-label2repre[tgt,]).item())
        return loss.cuda(), angle


    def train_tqn_model(self,net,net_id,batch_x,batch_y,label2repre,round):
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
        self.tqn[net_id].cuda()
        from torch.optim import Adam

        optimizer = Adam(filter(lambda p: p.requires_grad, self.tqn[net_id].parameters()),lr=0.000001/(round+1))
        criterion = CrossEntropyLoss().cuda()
        optimizer.zero_grad()  
        image_repre=image_repre.unsqueeze(1).detach().cuda()
        label2repre=label2repre.float().detach().cuda()
        out = self.tqn[net_id](image_repre, label2repre)##magic_num 1
        loss = criterion(out,batch_y)
        loss.backward()
        optimizer.step()
            
    def evaluate(self,net,net_id,dataloader, label2repre):
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
                target=target.cuda()
                self.tqn[net_id].cuda()
                out = self.tqn[net_id](image_repre, label2repre)                
                loss = criterion(out,target)
                _, pred_label = torch.max(out.data, 1)
                loss_collector.append(loss.item())
                pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
            avg_loss = sum(loss_collector) / len(loss_collector)
            acc=accuracy_score(true_labels_list,pred_labels_list)

        return avg_loss,acc