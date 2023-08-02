import torchvision.transforms as transforms
import torch
from sklearn.metrics import accuracy_score
import numpy as np
import logging

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


normalize = transforms.Normalize((0.485, 0.456, 0.406),
                                  (0.229, 0.224, 0.225))
class local_baseline():
    def __init__(self,control_config):
        self.cfg=control_config
        self.n_clients=self.cfg.federate.client_num        
        self.clients_per_round=self.cfg.federate.sample_client_num
        self.n_rounds=self.cfg.federate.total_round_num
        self.epochs=self.cfg.train.local_update_steps
        self.global_para={}


    def update_client_iter(self,net,net_id,batch_x,batch_y,criterion,optimizer):
        #print(batch_x.shape)
        
        
        transform_train = transforms.Compose([
                normalize
            ])
        batch_x=transform_train(batch_x)

        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        #logger.info(torch.cuda.memory_summary(device=0))
        optimizer.zero_grad()
        batch_x.requires_grad = False
        batch_y.requires_grad = False
        #target = target.long()
        out = net(batch_x)
        #print(batch_y)
        loss = criterion(out, batch_y)
        
        loss.backward()
        #nn.utils.clip_grad_norm_(net.parameters(),max_norm=5,norm_type=2)
        optimizer.step()
        
    def evaluate(self,net,dataloader,criterion):
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
                #print(out.shape)
                loss = criterion(out, target)
                _, pred_label = torch.max(out.data, 1)
                loss_collector.append(loss.item())
                #logger.info(loss_collector)
                #total += x.data.size()[0]
                #correct += (pred_label == target.data).sum().item()
                pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
            avg_loss = sum(loss_collector) / len(true_labels_list)
            acc=accuracy_score(true_labels_list,pred_labels_list)

        return avg_loss,acc    
