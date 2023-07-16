import os
import sys

DEV_MODE = True  # simplify the federatedscope re-setup everytime we change
# the source codes
if DEV_MODE:
    file_dir = os.path.join(os.path.dirname(__file__), '..')
    sys.path.append(file_dir)

import yaml
from collaboFM.client import Client_Manager_base
from collaboFM.algorithmzoo.algorithm import Algorithm_Manager
import argparse
import numpy as np
import torch
import random
from collaboFM.configs.config import global_cfg
from collaboFM.auxiliaries.logging import update_logger
from collaboFM.utils import setup_seed

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',
                        dest='cfg_file',
                        help='Config file path',
                        required=False,
                        type=str)
    parser.add_argument('opts',
                        help='See collaboFM/configs for all options',
                        default=None,
                        nargs=argparse.REMAINDER)
    # parser.add_argument('--seed', type=int, default=0,
    #                     help='seed of random,torch,cuda,numpy')
    # parser.add_argument('--basic_config', type=str, default='',
    #                     help='path of fixed config, hardly to change')
    # parser.add_argument('--model', type=str, default="cifar_resnet18",
    #                     help='name of model, like resnet18, resnet50')
    # parser.add_argument('--n_clients', type=int, default=5,
    #                     help='number of clients')
    # parser.add_argument('--clients_per_round', type=int, default=5,
    #                     help='number of clients that participate in each round')
    # parser.add_argument('--n_rounds', type=int, default=10,
    #                     help='the rounds number of communication')
    # parser.add_argument('--training_epochs', type=int, default=10,
    #                     help='the training epochs of local training')
    # parser.add_argument('--client_selection', type=str, default="random",
    #                     help='the client selection strategy')
    # parser.add_argument('--dataset', type=str, default="cifar10",
    #                     help='name of dataset')
    # parser.add_argument('--beta', type=float, default=0.5,
    #                     help='param of dirichlet distribution')
    # parser.add_argument('--partition', type=str, default="dirichlet",
    #                     help='partition strategy to split data, like dirichlet')
    # parser.add_argument('--n_class', type=int, default=3,
    #                     help='param of class partition strategy')
    # parser.add_argument('--method', type=str, default="FedAvg",
    #                     help='algorithm used')
    # parser.add_argument('--load_all_dataset', type=bool, default=False,
    #                     help='whether to load all client datasets in to RAM')
    # parser.add_argument('--train_batchsize', type=int, default=128,
    #                     help='batchsize of traning dataset')
    # parser.add_argument('--test_batchsize', type=int, default=128,
    #                     help='batchsize of test dataset')

    # parser.add_argument('--encoder_list', type=list, default=["identity"],
    #                     help='module list of encoder in model')
    # parser.add_argument('--encoder_para_list', type=dict, default=None,
    #                     help='param list of encoder in model')
    # parser.add_argument('--head_list', type=list, default=["linear"],
    #                     help='module list of head in model')
    # parser.add_argument('--head_para_list', type=dict, default={"in_dim":512,"out_dim":10},
    #                     help='param list of head in model')
    # parser.add_argument('--data_dir', type=str, default="/mnt/workspace/colla_group/data/",
    #                     help='path of data')
    # parser.add_argument('--criterion_dicts', type=dict, default={"type":"cross-entropy"},
    #                     help='param dict of optimizer')
    # parser.add_argument('--optimizer_dicts', type=dict, default={"type":"sgd","lr":0.01,"momentum":0.9,"weight_decay":0.001},
    #                     help='param dict of criterion')

    # parser.add_argument('--partition', type=str, default="class",
    #                     help='partition strategies')

    # parser.add_argument('--partition', type=str, default="class",
    #                     help='partition strategies')
    # parser.add_argument('--basic_config', type=str, default="./test.yaml",
    #                     help='basic configuration')
    # parser.add_argument('--mu', type=float, default=0.01,
    #                     help='param of fedprox or moon')
    # parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    # parser.add_argument('--dataset', type=str, default='cifar10', help='dataset used for training')
    # parser.add_argument('--n_parties', type=int, default=10, help='number of workers in a distributed cluster')
    # parser.add_argument('--model', type=str, default='resnet18', help='neural network used in training')
    # parser.add_argument('--party_per_round', type=int, default=10, help='how many clients are sampled in each round')
    # parser.add_argument('--comm_round', type=int, default=200, help='number of maximum communication roun')
    # parser.add_argument('--load_model_file', type=str, default=None, help='the model to load as global model')
    # parser.add_argument('--load_model_round', type=int, default=None,
    #                     help='how many rounds have executed for the loaded model')
    # parser.add_argument('--epochs', type=int, default=10, help='number of local epochs')
    # parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.1)')
    # parser.add_argument('--reg', type=float, default=1e-4, help="L2 regularization strength")
    # parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    args = parser.parse_args()
    return args

if __name__=="__main__":
    logger.info("FL start!!")
    init_cfg = global_cfg.clone()
    control_config=get_args()
    if(control_config.cfg_file):
        init_cfg.merge_from_file(control_config.cfg_file)
    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)
    init_cfg.freeze()
    logger.info("start training!!!")
    client_manager=Client_Manager_base(init_cfg)

    if init_cfg.federate.use_hetero_model:
        client_manager.create_multi_task_index_datasets()
    else:
        #client_resource is an empty dict
        client_manager.create_fed_split_index()    
        if init_cfg.data.load_all_dataset:
            client_manager.create_all_datasets(train_batchsize=init_cfg.train.batchsize,\
                test_batchsize=init_cfg.eval.batchsize, num_workers=8)
    # if control_config.load_all_model==True:
    client_manager.create_all_models()
    

    algorithm_manager=Algorithm_Manager(init_cfg,client_manager)
    algorithm_manager.run()

    
