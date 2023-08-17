import torchvision.transforms as transforms
import torch
from torch.nn import CrossEntropyLoss
from torch.nn.functional import one_hot
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np

from collaboFM.build import *
from collaboFM.data.label_name import get_label_name
from collaboFM.data.dataset import get_mean_std
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class fedclip():
    def __init__(self, control_cfg, client_manager):
        self.cfg = control_cfg
        self.n_clients = self.cfg.federate.client_num
        self.clients_pre_round = self.cfg.federate.sample_client_num
        self.n_rounds = self.cfg.federate.total_round_num
        self.epochs = self.cfg.train.local_update_steps

        self.test_x = None
        self.test_y = None
        self.test_ds = None
        self.test_dl = None
        self.client_manager = client_manager

    def init_server_dataset(self,data_name):
        test_batchsize = self.cfg.eval.batchsize
        num_workers = 8
        _,_,self.test_x,self.test_y = build_data(self.cfg,data_name)
        self.test_ds=build_ds(self.test_x,self.test_y)
        self.test_dl=DataLoader(dataset=self.test_ds, batch_size=test_batchsize, \
            drop_last=False, shuffle=True,num_workers=num_workers)        
        
    def run(self):
        for dataset_name, client_list in self.client_manager.dataset2idx.items():
            self.init_server_dataset(dataset_name)
            normalize = get_mean_std(dataset_name, "fedclip")
            label_name = get_label_name(self.cfg, dataset_name)
            label2token = clip.tokenize(label_name).cuda()
            weights=[len(self.client_manager.clients[idx].train_ds) for idx in range(self.client_manager.n_clients)]
            for round_idx in range(self.n_rounds):
                acc_list = []
                avg_acc = 0
                logger.info(f"-------------Round #{round_idx} start---------------")
                for client_idx in client_list:
                    if (self.cfg.data.load_all_dataset):
                        this_train_dl = self.client_manager.clients[client_idx].trian_dl
                        this_test_dl = self.client_manager.clients[client_idx].test_dl
                    else:
                        _,this_train_dl,_,this_test_dl = \
                        self.client_manager.create_one_dataset(self.client_manager.clients[client_idx],\
                                train_batchsize=self.cfg.train.batchsize,\
                                test_batchsize=self.cfg.eval.batchsize,num_workers=8)
                    net = self.client_manager.clients[client_idx].net
                    criterion4img = CrossEntropyLoss(reduction='sum').cuda()
                    criterion4txt = CrossEntropyLoss(reduction='sum').cuda()
                    optimizer=build_optimizer(net.parameters(),self.cfg.train.optimizer,round_idx)
                    net.cuda()
                    self.client_manager.clip_model.cuda()
                    for epoch in range(self.epochs):
                        net.train()
                        self.client_manager.clip_model.train()
                        loss = 0
                        tot = 0
                        for batch_idx, (batch_x, batch_y) in tqdm(enumerate(this_train_dl)):
                            transform_train = transforms.Compose([
                                    normalize
                                ])
                            batch_x.requires_grad = False
                            batch_y.requires_grad = False
                            N=batch_y.shape[0]
                            batch_x = batch_x.cuda()
                            batch_x = transform_train(batch_x)
                            batch_x = batch_x.detach().cuda()
                            batch_y = batch_y.cuda()
                            optimizer.zero_grad()
                            input_text = torch.vstack([label2token[batch_y[i]] for i in range(N)]).cuda()
                            logits_per_image, logits_per_text = net.forward_AttAI(batch_x, input_text, self.client_manager.clip_model)
                            # logger.info(local_y_raw)
                            # logger.info(batch_y)
                            ground_truth = torch.arange(N, dtype=torch.long).cuda()
                            bloss = (criterion4img(logits_per_image, ground_truth)+criterion4txt(logits_per_text, ground_truth))/2
                            loss+=bloss.item()
                            # logger.info(out.shape)
                            # logger.info(local_y.shape)
                            # if round_idx>=5:
                            bloss.backward()
                            optimizer.step()
                            tot+=N
                        logger.info(loss/tot)
                        net.eval()
                        self.client_manager.clip_model.eval()
                        true_labels_list, pred_labels_list = np.array([]), np.array([])
                        loss_collector = []
                        for batch_idx, (batch_x, batch_y) in enumerate(this_train_dl):
                            transform_train = transforms.Compose([
                                    normalize
                                ])
                            with torch.no_grad():
                                batch_x = batch_x.cuda()
                                batch_x = transform_train(batch_x)
                                N=batch_y.shape[0]
                                batch_x = batch_x.cuda()
                                batch_y = batch_y.cuda()
                                logits_per_image, logits_per_text = net.forward_AttAI(batch_x, label2token,self.client_manager.clip_model)
                                _, pred_label = torch.max(logits_per_image.detach(), -1)
                                loss = criterion4img(logits_per_image, batch_y)
                                # logger.info(criterion4decoder(out, local_y).item()/num_classes)
                                # logger.info(loss.item())
                                loss_collector.append(loss.item())
                                pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                                true_labels_list = np.append(true_labels_list, batch_y.cpu().numpy())
                        loss = sum(loss_collector) / len(true_labels_list)
                        acc=accuracy_score(true_labels_list, pred_labels_list)
                        logger.info(f"--------------client #{client_idx}------------------")
                        logger.info(f"train acc:{acc} train loss:{loss}")
                        true_labels_list, pred_labels_list = np.array([]), np.array([])
                        loss_collector = []
                        for batch_idx, (batch_x, batch_y) in enumerate(this_test_dl):
                            transform_train = transforms.Compose([
                                    normalize
                                ])
                            with torch.no_grad():
                                batch_x = batch_x.cuda()
                                batch_x = transform_train(batch_x)
                                N=batch_y.shape[0]
                                batch_x = batch_x.cuda()
                                batch_y = batch_y.cuda()
                                logits_per_image, logits_per_text = net.forward_AttAI(batch_x, label2token,self.client_manager.clip_model)
                                _, pred_label = torch.max(logits_per_image.detach(), -1)
                                loss = criterion4img(logits_per_image, batch_y)
                                # logger.info(criterion4decoder(out, local_y).item()/num_classes)
                                # logger.info(loss.item())
                                loss_collector.append(loss.item())
                                pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                                true_labels_list = np.append(true_labels_list, batch_y.cpu().numpy())
                        loss = sum(loss_collector) / len(true_labels_list)
                        acc=accuracy_score(true_labels_list, pred_labels_list)
                        logger.info(f"test acc:{acc} test loss:{loss}")
                        acc_list.append(acc)
                    net.cpu()
                global_para = self.client_manager.clients[0].net.state_dict()
                for key in global_para:
                    global_para[key] = global_para[key] * weights[0]/sum(weights)
                for idx in range(1, self.client_manager.n_clients):
                    net_para = self.client_manager.clients[idx].net.state_dict()
                    weight = weights[idx]/sum(weights)
                    for key in net_para:
                        global_para[key] += net_para[key] * weight
                for idx in range(self.client_manager.n_clients):
                    avg_acc += acc_list[idx]*weights[idx]/sum(weights)
                    self.client_manager.clients[idx].net.load_state_dict(global_para)
                net = self.client_manager.clients[0].net.cuda()
                net.eval()
                self.client_manager.clip_model.eval()
                true_labels_list, pred_labels_list = np.array([]), np.array([])
                loss_collector = []
                for batch_idx, (batch_x, batch_y) in enumerate(self.test_dl):
                    transform_train = transforms.Compose([
                            normalize
                        ])
                    with torch.no_grad():
                        batch_x = batch_x.cuda()
                        batch_x = transform_train(batch_x)
                        N=batch_y.shape[0]
                        batch_x = batch_x.cuda()
                        batch_y = batch_y.cuda()
                        logits_per_image, logits_per_text = net.forward_AttAI(batch_x, label2token,self.client_manager.clip_model)
                        _, pred_label = torch.max(logits_per_image.detach(), -1)
                        loss = criterion4img(logits_per_image, batch_y)
                        # logger.info(criterion4decoder(out, local_y).item()/num_classes)
                        # logger.info(loss.item())
                        loss_collector.append(loss.item())
                        pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                        true_labels_list = np.append(true_labels_list, batch_y.cpu().numpy())
                loss = sum(loss_collector) / len(true_labels_list)
                acc=accuracy_score(true_labels_list, pred_labels_list)
                logger.info(f"--------------Round #{round_idx} End------------------")
                logger.info(f"Global Acc:{acc} Global Loss:{loss}")
                logger.info(f"Personalized Acc:{avg_acc}")