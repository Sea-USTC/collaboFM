from collaboFM.build import *
from torch.utils.data.dataloader import DataLoader
import logging
from collaboFM.algorithmzoo.fedavg import FedAvg
from collaboFM.algorithmzoo.collabo import collabo
from collaboFM.algorithmzoo.local import local_baseline
from collaboFM.algorithmzoo.fedclip import fedclip
from collaboFM.algorithmzoo.cliptqn import cliptqn
from collaboFM.data.label_name import get_label_name
from collaboFM.data.dataset import get_mean_std
from tqdm import tqdm

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
        self.epoch_list = self.cfg.tqn_train.epoch_list
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
            if "vit" in self.cfg.client_resource.backbone["0"]:
                self.run_local_vit()
            else:
                self.run_local()
        elif method=="collabo":
            if "vit" in self.cfg.client_resource.backbone["0"]:
                self.run_colla_train_vit()
            else:
                self.run_colla_train()
        elif method == "collabo_wo_tqn":
            self.run_colla_train_vit_head()
        elif method == "fedclip":
            self.run_fedclip()
        elif method == "cliptqn":
            self.run_cliptqn()
    
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
                optimizer=build_optimizer(net,self.cfg.train.optimizer,round_idx)
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
        from torchvision import transforms
        from sklearn.metrics import accuracy_score
        from torch.nn import CrossEntropyLoss
        self.algorithm=local_baseline(self.cfg)
        for dataset_name, client_list in self.client_manager.dataset2idx.items():
            normalize = get_mean_std(dataset_name)
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
                                test_batchsize=self.cfg.eval.batchsize,num_workers=8)
                    net=self.client_manager.clients[client_idx].net
                    criterion=CrossEntropyLoss(reduction='sum').cuda()
                    optimizer=build_optimizer(net,self.cfg.train.optimizer,round_idx)
                    net.cuda()
                    #logger.info(torch.cuda.memory_summary(device=0))
                    for epoch in range(self.training_epochs):                    
                        net.train()
                        for batch_idx, (batch_x, batch_y) in enumerate(this_train_dl):
                            self.algorithm.update_client_iter(net,client_idx,batch_x,batch_y,criterion,optimizer)
                        net.eval()
                        loss,acc=self.algorithm.evaluate(net,this_train_dl,criterion)#training loss
                        logger.info(f"--------------client #{client_idx}------------------")
                        logger.info(f"train acc:{acc} train loss:{loss}")
                        loss,acc=self.algorithm.evaluate(net,this_test_dl,criterion)
                        logger.info(f"test acc:{acc} test loss:{loss}")
                        ### out test acc
                        true_labels_list, pred_labels_list = np.array([]), np.array([])
                        loss_collector = []
                        for i in client_list:
                            if i == client_idx:
                                continue
                            _, all_test_dl=\
                            self.client_manager.create_one_dataset_test(self.client_manager.clients[i],\
                                train_batchsize=self.cfg.train.batchsize,\
                                test_batchsize=self.cfg.eval.batchsize,num_workers=8)
                            for batch_idx, (batch_x, batch_y) in enumerate(all_test_dl):
                                transform_train = transforms.Compose([
                                        normalize
                                    ])
                                with torch.no_grad():
                                    batch_x = batch_x.cuda()
                                    batch_x = transform_train(batch_x)
                                    batch_x = batch_x.cuda()
                                    batch_y = batch_y.cuda()
                                    out = net(batch_x)
                                    _, pred_label = torch.max(out.data, -1)
                                    loss = criterion(out, batch_y)
                                    loss_collector.append(loss.item())
                                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                                    true_labels_list = np.append(true_labels_list, batch_y.data.cpu().numpy())
                            # logger.info(len(true_labels_list))
                        loss = sum(loss_collector) / len(true_labels_list)
                        logger.info(len(true_labels_list))
                        acc=accuracy_score(true_labels_list,pred_labels_list)
                        logger.info(f"out test acc:{acc} out test loss:{loss}")

    def run_with_clip(self):
        self.algorithm=collabo(self.cfg)
        import collaboFM.clip as clip
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.server.net, preprocess = clip.load("/mnt/workspace/colla_group/ViT-B-32.pt", device=device)
        for dataset_name, client_list in self.client_manager.dataset2idx.items():
            label_name = get_label_name(self.cfg, dataset_name)
            label2token=clip.tokenize(label_name).cuda()
            label2repre=self.server.net.encode_text(label2token).to('cpu')
            train_encoder=True
            dump_path="/mnt/workspace/lisiyi/result/backbone.pt"
            if train_encoder:
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
                        optimizer=build_optimizer(net,self.cfg.train.optimizer,round_idx)
                        net.cuda()
                        for epoch in range(self.training_epochs):
                            net.train()
                            loss=0
                            tot=0
                            angle=[]
                            for batch_idx, (batch_x, batch_y) in enumerate(this_train_dl):
                                bnum, bloss,bangle=self.algorithm.update_client_iter(net,client_idx,batch_x,batch_y,criterion,optimizer,label2repre)
                                loss+=bloss.item()
                                angle+=bangle
                                tot+=bnum
                            logger.info(loss/tot)
                            logger.info(sum(angle)/tot)

                model_list={}
                for client_idx in client_list:
                    self.client_manager.clients[client_idx].net.eval()
                    model_list[client_idx]=self.client_manager.clients[client_idx].net.state_dict()
                torch.save(model_list,dump_path)
            else:
                model_list=torch.load(dump_path)
                for client_idx in client_list:
                    self.client_manager.clients[client_idx].net.load_state_dict(model_list[client_idx])
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
                        self.algorithm.tqn[client_idx].train()
                        for batch_idx, (batch_x, batch_y) in enumerate(this_train_dl):
                            self.algorithm.train_tqn_model(net,client_idx,batch_x,batch_y,label2repre,round_idx)
                        self.algorithm.tqn[client_idx].eval()
                        loss,acc=self.algorithm.evaluate(net,client_idx,this_train_dl,label2repre)#training loss
                        logger.info(f"--------------client #{client_idx}------------------")
                        logger.info(f"train acc:{acc} train loss:{loss}")
                        loss,acc=self.algorithm.evaluate(net,client_idx,this_test_dl,label2repre)
                        logger.info(f"test acc:{acc} test loss:{loss}")
    
    def run_colla_train(self):
        import torchvision.transforms as transforms
        import collaboFM.clip as clip
        from sklearn.metrics import accuracy_score
        self.algorithm=collabo(self.cfg)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.server.net, preprocess = clip.load("/mnt/workspace/colla_group/ViT-B-32.pt", device=device)
        for dataset_name, client_list in self.client_manager.dataset2idx.items():
            label_name = get_label_name(self.cfg, dataset_name)
            label2token=clip.tokenize(label_name).cuda()
            label2repre=self.server.net.encode_text(label2token)
            dump_path="/mnt/workspace/lisiyi/result/backbone.pt"
            client_list=[0]
            decay_round=10
            loss_type="cluster"
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
                                test_batchsize=self.cfg.eval.batchsize,num_workers=8)
                    net=self.client_manager.clients[client_idx].net
                    tqn=self.algorithm.tqn[client_idx]
                    net.cuda()
                    tqn.cuda()
                    from torch.nn import CrossEntropyLoss
                    optimizer4decoder = build_optimizer(tqn,self.cfg.tqn_train.tqn_optimizer,round_idx)
                    criterion4decoder = CrossEntropyLoss().cuda()
                    criterion4encoder=self.algorithm.similarity
                    optimizer4encoder=build_optimizer(net,self.cfg.train.optimizer,round_idx)
                    for epoch in range(self.training_epochs):
                        net.train()
                        tqn.train()
                        loss=0
                        tot=0
                        angle=[]
                        for batch_idx, (batch_x, batch_y) in enumerate(this_train_dl):
                            normalize = get_mean_std(dataset_name)
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
                            optimizer4encoder.zero_grad()
                            optimizer4decoder.zero_grad()
                            features = net.forward_with_feature(batch_x)[1]
                            bloss, bangle = criterion4encoder(features, label2repre, batch_y, flag=loss_type,tau=self.cfg.tqn_train.tau,norm=False)
                            loss+=bloss.item()
                            features = features.unsqueeze(1)
                            mu = self.cfg.tqn_train.mu
                            if round_idx>=25:
                                mu = 0
                                #features=features.detach()
                            out = tqn(features, label2repre)##magic_num 1
                            bloss = mu/decay_round*min(round_idx,decay_round)*bloss+criterion4decoder(out,batch_y)
                            bloss.backward()
                            if round_idx<25:
                                optimizer4encoder.step()
                            optimizer4decoder.step()
                            angle+=bangle
                            tot+=batch_y.shape[0]
                        logger.info(loss/tot)
                        logger.info(sum(angle)/tot)
                        net.eval()
                        tqn.eval()
                        true_labels_list, pred_labels_list = np.array([]), np.array([])
                        loss_collector = []
                        for batch_idx, (batch_x, batch_y) in enumerate(this_train_dl):
                            normalize = get_mean_std(dataset_name)
                            transform_train = transforms.Compose([
                                    normalize
                                ])
                            with torch.no_grad():
                                batch_x = batch_x.cuda()
                                batch_x = transform_train(batch_x)
                                N=batch_y.shape[0]
                                batch_x = batch_x.cuda()
                                batch_y = batch_y.cuda()
                                label2repre = label2repre.float().cuda()
                                features = net.forward_with_feature(batch_x)[1]
                                features = features.unsqueeze(1)
                                out = tqn(features, label2repre)##magic_num 1
                                _, pred_label = torch.max(out.data, 1)
                                loss = criterion4decoder(out,batch_y)
                                loss_collector.append(loss.item())
                                pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                                true_labels_list = np.append(true_labels_list, batch_y.data.cpu().numpy())
                        loss = sum(loss_collector) / len(true_labels_list)
                        acc=accuracy_score(true_labels_list,pred_labels_list)
                        logger.info(f"--------------client #{client_idx}------------------")
                        logger.info(f"train acc:{acc} train loss:{loss}")
                        true_labels_list, pred_labels_list = np.array([]), np.array([])
                        loss_collector = []
                        for batch_idx, (batch_x, batch_y) in enumerate(this_test_dl):
                            normalize = get_mean_std(dataset_name)
                            transform_train = transforms.Compose([
                                    normalize
                                ])
                            with torch.no_grad():
                                batch_x = batch_x.cuda()
                                batch_x = transform_train(batch_x)
                                N=batch_y.shape[0]
                                batch_x = batch_x.cuda()
                                batch_y = batch_y.cuda()
                                label2repre = label2repre.float().cuda()
                                features = net.forward_with_feature(batch_x)[1]
                                features = features.unsqueeze(1)
                                out = tqn(features, label2repre)##magic_num 1
                                _, pred_label = torch.max(out.data, 1)
                                loss = criterion4decoder(out,batch_y)
                                loss_collector.append(loss.item())
                                pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                                true_labels_list = np.append(true_labels_list, batch_y.data.cpu().numpy())
                        loss = sum(loss_collector) / len(true_labels_list)
                        acc=accuracy_score(true_labels_list,pred_labels_list)
                        logger.info(f"test acc:{acc} test loss:{loss}")
    
    def run_fedclip(self):
        self.algorithm = fedclip(self.cfg, self.client_manager)
        self.algorithm.run()

    def run_cliptqn(self):
        self.algorithm = cliptqn(self.cfg, self.client_manager)
        self.algorithm.run()

    def run_local_vit(self):
        from torchvision import transforms
        from sklearn.metrics import accuracy_score
        from torch.nn import CrossEntropyLoss
        self.algorithm=local_baseline(self.cfg)
        for dataset_name, client_list in self.client_manager.dataset2idx.items():
            normalize = get_mean_std(dataset_name)
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
                                test_batchsize=self.cfg.eval.batchsize,num_workers=8)
                    net=self.client_manager.clients[client_idx].net
                    from math import sqrt
                    criterion=CrossEntropyLoss(reduction='sum').cuda()
                    optimizer=build_optimizer(net.parameters(),self.cfg.tqn_train.tqn_optimizer,sqrt(round_idx))
                    #net=nn.DataParallel(net.to(f"cuda:{self.cfg.gpus[0]}"), device_ids=self.cfg.gpus, output_device=self.cfg.gpus[0])
                    net.cuda()
                    #logger.info(torch.cuda.memory_summary(device=0))
                    for epoch in range(self.training_epochs):                    
                        net.train()
                        for batch_idx, (batch_x, batch_y) in enumerate(this_train_dl):
                            self.algorithm.update_client_iter(net,client_idx,batch_x,batch_y,criterion,optimizer)
                        net.eval()
                        loss,acc=self.algorithm.evaluate(net,this_train_dl,criterion)#training loss
                        logger.info(f"--------------client #{client_idx}------------------")
                        logger.info(f"train acc:{acc} train loss:{loss}")
                        loss,acc=self.algorithm.evaluate(net,this_test_dl,criterion)
                        logger.info(f"test acc:{acc} test loss:{loss}")
                        ### out test acc
                        true_labels_list, pred_labels_list = np.array([]), np.array([])
                        loss_collector = []
                        for i in client_list:
                            if i == client_idx:
                                continue
                            _, all_test_dl=\
                            self.client_manager.create_one_dataset_test(self.client_manager.clients[i],\
                                train_batchsize=self.cfg.train.batchsize,\
                                test_batchsize=self.cfg.eval.batchsize,num_workers=8)
                            for batch_idx, (batch_x, batch_y) in enumerate(all_test_dl):
                                transform_train = transforms.Compose([
                                        normalize
                                    ])
                                with torch.no_grad():
                                    batch_x = batch_x.cuda()
                                    batch_x = transform_train(batch_x)
                                    batch_x = batch_x.cuda()
                                    batch_y = batch_y.cuda()
                                    out = net(batch_x)
                                    _, pred_label = torch.max(out.data, -1)
                                    loss = criterion(out, batch_y)
                                    loss_collector.append(loss.item())
                                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                                    true_labels_list = np.append(true_labels_list, batch_y.data.cpu().numpy())
                            # logger.info(len(true_labels_list))
                        loss = sum(loss_collector) / len(true_labels_list)
                        logger.info(pred_labels_list[500:530])
                        logger.info(true_labels_list[500:530])
                        acc=accuracy_score(true_labels_list,pred_labels_list)
                        logger.info(f"out test acc:{acc} out test loss:{loss}")

    def run_colla_train_vit(self):
        import torchvision.transforms as transforms
        import collaboFM.clip as clip
        from sklearn.metrics import accuracy_score
        from itertools import chain
        self.algorithm=collabo(self.cfg)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.server.net, preprocess = clip.load("/mnt/workspace/colla_group/ViT-B-32.pt", device=device)
        weights=[len(self.client_manager.clients[idx].train_ds) for idx in range(self.client_manager.n_clients)]
        for dataset_name, client_list in self.client_manager.dataset2idx.items():
            # test_ds_all=build_ds(self.client_manager.test_x,self.client_manager.test_y)
            # test_dl_all=DataLoader(dataset=test_ds_all, batch_size=self.cfg.eval.batchsize,\
            #  drop_last=False, shuffle=False,num_workers=8)
            label_name = get_label_name(self.cfg, dataset_name)
            label2token = clip.tokenize(label_name).cuda()
            label2repre = self.server.net.encode_text(label2token)
            label2repre=label2repre/label2repre.norm(dim=1,keepdim=True)
            dump_path = "/mnt/workspace/lisiyi/result/backbone.pt"
            decay_round = 10
            loss_type="KADpro"
            normalize = get_mean_std(dataset_name)
            local_repre={}
            class_array={}
            for client_idx in client_list:
                class_array[client_idx] = self.client_manager.class_dict[client_idx]
                # logger.info(type(class_array[client_idx]))
                local_repre[client_idx] = torch.vstack([label2repre[i,] for i in class_array[client_idx]])
            for round_idx in range(self.cfg.tqn_train.key_train_round):
                logger.info(f"##########Round #{round_idx} ################")
                for client_idx in client_list:
                    class2idx = {}                
                    for i, j in enumerate(class_array[client_idx]):
                        class2idx[j]=i
                    # logger.info(class2idx.values())
                    logger.info(f"--------------client #{client_idx}------------------\n"
                                f"              train start!")
                    if(self.cfg.data.load_all_dataset):
                        this_train_dl=self.client_manager.clients[client_idx].train_dl
                        this_test_dl=self.client_manager.clients[client_idx].test_dl
                    else:
                        this_train_ds,this_train_dl,this_test_ds,this_test_dl=\
                            self.client_manager.create_one_dataset(self.client_manager.clients[client_idx],\
                                train_batchsize=self.cfg.train.batchsize,\
                                test_batchsize=self.cfg.eval.batchsize,num_workers=8)
                    net=self.client_manager.clients[client_idx].net
                    tqn=self.algorithm.tqn[client_idx]
                    net.cuda()
                    tqn.cuda()
                    from torch.nn import CrossEntropyLoss
                    from math import sqrt
                    #print(type(tqn.parameters()))
                    optimizer = build_optimizer(chain(tqn.parameters(),net.parameters()),self.cfg.tqn_train.tqn_optimizer,sqrt(round_idx))
                    criterion4decoder = CrossEntropyLoss(reduction='sum').cuda()
                    criterion4encoder = CrossEntropyLoss(reduction='sum').cuda()
                    #criterion4encoder=self.algorithm.similarity
                    #optimizer4encoder=build_optimizer(net,self.cfg.train.optimizer,round_idx)
                    for epoch in range(self.epoch_list[client_idx]):
                        net.train()
                        tqn.train()
                        #interior train
                        loss = 0
                        loss2 = 0 
                        tot=0
                        angle=[]
                        num_classes = len(class_array[client_idx])
                        for batch_idx, (batch_x, batch_y) in tqdm(enumerate(this_train_dl)):
                            transform_train = transforms.Compose([
                                    normalize
                                ])
                            batch_x.requires_grad = False
                            batch_y.requires_grad = False
                            N=batch_y.shape[0]
                            local_y_raw = torch.asarray([class2idx[batch_y[i].item()] for i in range(N)]).cuda()
                            local_y = torch.nn.functional.one_hot(
                                torch.nn.functional.one_hot(local_y_raw, num_classes=num_classes)).float()
                            local_y.requires_grad = False
                            batch_x = batch_x.cuda()
                            batch_x = transform_train(batch_x)
                            batch_x = batch_x.detach().cuda()
                            batch_y = batch_y.cuda()
                            local_y = local_y.cuda()
                            label2repre = label2repre.float().detach().cuda()
                            local_label2repre = local_repre[client_idx].float().detach().cuda()
                            #optimizer4encoder.zero_grad()
                            optimizer.zero_grad()
                            features = net.forward_with_feature(batch_x)[1]
                            bloss = criterion4encoder(net.logit_scale.exp()*features@label2repre.t(), batch_y)
                            # logger.info(local_y_raw)
                            # logger.info(batch_y)
                            loss+=bloss.item()
                            features = features.unsqueeze(1)
                            mu = self.cfg.tqn_train.mu
                            out = tqn(features, local_label2repre)##magic_num 1
                            # logger.info(out.shape)
                            # logger.info(local_y.shape)
                            # if round_idx>=5:
                            tqn_loss = - (torch.log_softmax(out,dim=-1)*local_y).flatten(0).sum()/num_classes
                            loss2+=tqn_loss.item()
                            bloss = mu*bloss + tqn_loss
                            bloss.backward()
                            optimizer.step()
                            tot+=batch_y.shape[0]
                        logger.info(loss/tot)
                        logger.info(loss2/tot)
                        # logger.info(sum(angle)/tot)
                        net.eval()
                        tqn.eval()
                        #interior train acc
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
                                local_y_raw = torch.asarray([class2idx[batch_y[i].item()] for i in range(N)])
                                local_y = torch.nn.functional.one_hot(torch.nn.functional.one_hot(local_y_raw, num_classes=num_classes)).float()
                                local_y.requires_grad = False
                                batch_x = batch_x.cuda()
                                batch_y = batch_y.cuda()
                                local_y = local_y.cuda()
                                label2repre = label2repre.float().cuda()
                                local_label2repre = local_repre[client_idx].float().detach().cuda()
                                features = net.forward_with_feature(batch_x)[1]
                                features = features.unsqueeze(1)
                                out = tqn(features, local_label2repre)##magic_num 1
                                _, pred_label = torch.max(torch.squeeze(out[:,:,1]), -1)
                                loss = - (torch.log_softmax(out,dim=-1)*local_y).flatten(0).sum()/num_classes
                                # logger.info(criterion4decoder(out, local_y).item()/num_classes)
                                # logger.info(loss.item())
                                loss_collector.append(loss.item())
                                pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                                true_labels_list = np.append(true_labels_list, local_y_raw.cpu().numpy())
                        loss = sum(loss_collector) / len(true_labels_list)
                        acc=accuracy_score(true_labels_list,pred_labels_list)
                        logger.info(f"--------------client #{client_idx}------------------")
                        logger.info(f"train acc:{acc} train loss:{loss}")
                        # interior test acc
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
                                local_y_raw = torch.asarray([class2idx[batch_y[i].item()] for i in range(N)])
                                local_y = torch.nn.functional.one_hot(torch.nn.functional.one_hot(local_y_raw, num_classes=num_classes)).float()
                                local_y.requires_grad = False
                                batch_x = batch_x.cuda()
                                batch_y = batch_y.cuda()
                                local_y = local_y.cuda()
                                label2repre = label2repre.float().cuda()
                                local_label2repre = local_repre[client_idx].float().detach().cuda()
                                features = net.forward_with_feature(batch_x)[1]
                                features = features.unsqueeze(1)
                                out = tqn(features, local_label2repre)##magic_num 1
                                _, pred_label = torch.max(torch.squeeze(out[:,:,1]), -1)
                                loss = - (torch.log_softmax(out,dim=-1)*local_y).flatten(0).sum()/num_classes
                                loss_collector.append(loss.item())
                                pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                                true_labels_list = np.append(true_labels_list, local_y_raw.cpu().numpy())
                        loss = sum(loss_collector) / len(true_labels_list)
                        acc=accuracy_score(true_labels_list,pred_labels_list)
                        logger.info(f"test acc:{acc} test loss:{loss}")
                        # outerior test acc
                        # true_labels_list, pred_labels_list = np.array([]), np.array([])
                        # loss_collector = []
                        # for i in client_list:
                        #     if i == client_idx:
                        #         continue
                        #     _, all_test_dl=\
                        #     self.client_manager.create_one_dataset_test(self.client_manager.clients[i],\
                        #         train_batchsize=self.cfg.train.batchsize,\
                        #         test_batchsize=self.cfg.eval.batchsize,num_workers=8)
                        #     for batch_idx, (batch_x, batch_y) in enumerate(all_test_dl):
                        #         transform_train = transforms.Compose([
                        #                 normalize
                        #             ])
                        #         with torch.no_grad():
                        #             batch_x = batch_x.cuda()
                        #             batch_x = transform_train(batch_x)
                        #             N=batch_y.shape[0]
                        #             batch_x = batch_x.cuda()
                        #             batch_y = batch_y.cuda()
                        #             batch_y_bce = torch.nn.functional.one_hot(
                        #                 torch.nn.functional.one_hot(batch_y,num_classes=label2repre.shape[0])).float()
                        #             label2repre = label2repre.float().cuda()
                        #             features = net.forward_with_feature(batch_x)[1]
                        #             features = features.unsqueeze(1)
                        #             out = tqn(features, label2repre)##magic_num 1
                        #             _, pred_label = torch.max(torch.squeeze(out[:,:,1]), -1)
                        #             loss = - (torch.log_softmax(out,dim=-1)*batch_y_bce).flatten(0).sum()/label2repre.shape[0]
                        #             loss_collector.append(loss.item())
                        #             pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                        #             true_labels_list = np.append(true_labels_list, batch_y.data.cpu().numpy())
                        #     # logger.info(len(true_labels_list))
                        # loss = sum(loss_collector) / len(true_labels_list)
                        # logger.info(len(true_labels_list))
                        # acc=accuracy_score(true_labels_list,pred_labels_list)
                        # logger.info(f"out test acc:{acc} out test loss:{loss}")
                    tqn.to("cpu")
                # parameter aggregation 
                if self.cfg.fm.use:
                    global_para = self.algorithm.tqn[0].state_dict()
                    for key in global_para:
                        global_para[key] = global_para[key] * weights[0]/sum(weights)
                    for idx in range(1,self.client_manager.n_clients):
                        net_para = self.algorithm.tqn[idx].state_dict()
                        weight = weights[idx] / sum(weights)                       
                        for key in net_para:
                            global_para[key] += net_para[key] * weight
                    for idx in range(self.client_manager.n_clients):
                        self.algorithm.tqn[idx].load_state_dict(global_para)
                    

    def run_colla_train_vit_head(self):
        import torchvision.transforms as transforms
        import collaboFM.clip as clip
        from sklearn.metrics import accuracy_score
        from itertools import chain
        self.algorithm=collabo(self.cfg)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.server.net, preprocess = clip.load("/mnt/workspace/colla_group/ViT-B-32.pt", device=device)
        for dataset_name, client_list in self.client_manager.dataset2idx.items():
            label_name = get_label_name(self.cfg, dataset_name)
            label2token=clip.tokenize(label_name).cuda()
            label2repre=self.server.net.encode_text(label2token)
            label2repre=label2repre/label2repre.norm(dim=1,keepdim=True)
            dump_path="/mnt/workspace/lisiyi/result/backbone.pt"
            # client_list=[0]
            decay_round=10
            loss_type="KADpro"
            normalize = get_mean_std(dataset_name)
            norm=False
            split_train = False
            
            for round_idx in range(self.cfg.tqn_train.key_train_round):
                logger.info(f"##########Round #{round_idx} ################")
                for client_idx in client_list:
                    logger.info(f"--------------client #{client_idx}------------------\n"
                                f"             train start!")
                    if(self.cfg.data.load_all_dataset):
                        this_train_dl=self.client_manager.clients[client_idx].train_dl
                        this_test_dl=self.client_manager.clients[client_idx].test_dl
                    else:
                        this_train_ds,this_train_dl,this_test_ds,this_test_dl=\
                            self.client_manager.create_one_dataset(self.client_manager.clients[client_idx],\
                                train_batchsize=self.cfg.train.batchsize,\
                                test_batchsize=self.cfg.eval.batchsize,num_workers=8)
                    net=self.client_manager.clients[client_idx].net
                    net.cuda()
                    from torch.nn import CrossEntropyLoss
                    from math import sqrt
                    optimizer = build_optimizer(net.parameters(),self.cfg.tqn_train.tqn_optimizer,sqrt(round_idx))
                    criterion4decoder = CrossEntropyLoss().cuda()
                    criterion4encoder=self.algorithm.similarity
                    for epoch in range(self.training_epochs):
                        net.train()
                        loss=0
                        tot=0
                        angle=[]
                        for batch_idx, (batch_x, batch_y) in enumerate(this_train_dl):
                            
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
                            #optimizer4encoder.zero_grad()
                            optimizer.zero_grad()
                            features = net.forward_with_feature(batch_x)[1]
                            # bloss, bangle = criterion4encoder(features.clone(), label2repre.clone(), batch_y, flag=loss_type,tau=self.cfg.tqn_train.tau,norm=norm)
                            # loss+=bloss.item()
                            # angle+=bangle
                            # out = net.head(features)
                            # mu = self.cfg.tqn_train.mu
                            # # if round_idx>=25:
                            # #     mu = 0
                            # if round_idx <0:
                            #     bloss.backward()
                            # else:
                            #     bloss = mu*bloss+criterion4decoder(out,batch_y)
                            #     bloss.backward()
                            bloss = criterion4decoder(net.logit_scale.exp()*features@label2repre.t(), batch_y)
                            loss+=bloss.item()
                            bloss.backward()
                            optimizer.step()
                            tot+=batch_y.shape[0]
                        logger.info(loss/tot)
                        logger.info(sum(angle)/tot)
                        net.eval()
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
                                label2repre = label2repre.float().cuda()
                                features = net.forward_with_feature(batch_x)[1]
                                # out = net.head(features)
                                out = net.logit_scale.exp()*features@label2repre.t()
                                _, pred_label = torch.max(out.data, 1)
                                loss = criterion4decoder(out,batch_y)
                                loss_collector.append(loss.item())
                                pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                                true_labels_list = np.append(true_labels_list, batch_y.data.cpu().numpy())
                        loss = sum(loss_collector) / len(true_labels_list)
                        acc=accuracy_score(true_labels_list,pred_labels_list)
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
                                features = net.forward_with_feature(batch_x)[1]
                                # out = net.head(features)
                                out = net.logit_scale.exp()*features@label2repre.t()
                                _, pred_label = torch.max(out.data, 1)
                                loss = criterion4decoder(out,batch_y)
                                loss_collector.append(loss.item())
                                pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                                true_labels_list = np.append(true_labels_list, batch_y.data.cpu().numpy())
                        loss = sum(loss_collector) / len(true_labels_list)
                        acc=accuracy_score(true_labels_list,pred_labels_list)
                        logger.info(f"test acc:{acc} test loss:{loss}")