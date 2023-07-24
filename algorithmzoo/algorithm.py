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
            self.run_colla_train()
    
    
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
                    optimizer=build_optimizer(net,self.cfg.train.optimizer,round_idx)
                    net=nn.DataParallel(net.to(f"cuda:{gpus[0]}"), device_ids=gpus, output_device=gpus[0])
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
            client_list=[1]
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
                                test_batchsize=self.cfg.test.batchsize,num_workers=8)
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
                            optimizer4encoder.zero_grad()
                            optimizer4decoder.zero_grad()
                            features = net.forward_with_feature(batch_x)[1]
                            bloss, bangle = criterion4encoder(features, label2repre, batch_y, flag=loss_type,tau=self.cfg.tqn_train.tau)
                            loss+=bloss.item()
                            features = features.unsqueeze(1)
                            out = tqn(features, label2repre)##magic_num 1
                            bloss = self.cfg.tqn_train.mu/decay_round*round_idx*bloss+criterion4decoder(out,batch_y)
                            bloss.backward()
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
                            normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2470, 0.2435, 0.2615))
        
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
                        loss = sum(loss_collector) / len(loss_collector)
                        acc=accuracy_score(true_labels_list,pred_labels_list)
                        logger.info(f"--------------client #{client_idx}------------------")
                        logger.info(f"train acc:{acc} train loss:{loss}")
                        loss,acc=self.algorithm.evaluate(net,client_idx,this_test_dl,label2repre)
                        logger.info(f"test acc:{acc} test loss:{loss}")
     

