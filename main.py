import yaml
from client import Client_Manager_base
from algorithm import Algorithm_Manager
import argparse
import numpy as np
import torch
import random
def set_seed(seed):
    #seed = arinit_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0,
                        help='seed of random,torch,cuda,numpy')
    parser.add_argument('--basic_config', type=str, default='',
                        help='path of fixed config, hardly to change')
    parser.add_argument('--model', type=str, default="cifar_resnet18",
                        help='name of model, like resnet18, resnet50')
    parser.add_argument('--n_clients', type=int, default=5,
                        help='number of clients')
    parser.add_argument('--clients_per_round', type=int, default=5,
                        help='number of clients that participate in each round')
    parser.add_argument('--n_rounds', type=int, default=10,
                        help='the rounds number of communication')
    parser.add_argument('--training_epochs', type=int, default=10,
                        help='the training epochs of local training')
    parser.add_argument('--client_selection', type=str, default="random",
                        help='the client selection strategy')
    parser.add_argument('--dataset', type=str, default="cifar10",
                        help='name of dataset')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='param of dirichlet distribution')
    parser.add_argument('--partition', type=str, default="dirichlet",
                        help='partition strategy to split data, like dirichlet')
    parser.add_argument('--n_class', type=int, default=3,
                        help='param of class partition strategy')
    parser.add_argument('--method', type=str, default="FedAvg",
                        help='algorithm used')
    parser.add_argument('--load_all_dataset', type=bool, default=False,
                        help='whether to load all client datasets in to RAM')
    parser.add_argument('--train_batchsize', type=int, default=128,
                        help='batchsize of traning dataset')
    parser.add_argument('--test_batchsize', type=int, default=128,
                        help='batchsize of test dataset')

    parser.add_argument('--encoder_list', type=list, default=["identity"],
                        help='module list of encoder in model')
    parser.add_argument('--encoder_para_list', type=dict, default=None,
                        help='param list of encoder in model')
    parser.add_argument('--head_list', type=list, default=["linear"],
                        help='module list of head in model')
    parser.add_argument('--head_para_list', type=dict, default={"in_dim":512,"out_dim":10},
                        help='param list of head in model')
    parser.add_argument('--data_dir', type=str, default="/mnt/workspace/colla_group/data/",
                        help='path of data')
    parser.add_argument('--criterion_dicts', type=dict, default={"type":"cross-entropy"},
                        help='param dict of optimizer')
    parser.add_argument('--optimizer_dicts', type=dict, default={"type":"sgd","lr":0.01,"momentum":0.9,"weight_decay":0.001},
                        help='param dict of criterion')

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
    
    control_config=get_args()
    #print(control_config.n_clients)
    basic_config_path=control_config.basic_config
    if(basic_config_path):
        basic_config=yaml.load(basic_config_path,Loader=yaml.FullLoader)
    else:
        basic_config=None
    set_seed(control_config.seed)
    
    
    client_manager=Client_Manager_base(control_config,basic_config)

    if control_config.client_resource:
        client_manager.create_multi_task_index_datasets()
    else:
        #client_resource is an empty dict
        client_manager.create_fed_split_index()    
        if control_config.load_all_dataset==True:
            client_manager.create_all_datasets(train_batchsize=control_config.train_batchsize,\
                test_batchsize=control_config.test_batchsize,num_workers=8)
    # if control_config.load_all_model==True:
    client_manager.create_all_models()
    
    algorithm_manager=Algorithm_Manager(basic_config,control_config,client_manager)
    algorithm_manager.run()

    
