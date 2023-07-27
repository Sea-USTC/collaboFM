import numpy as np
import random
def split_dirichlet(y_train,y_test,n_clients=3,beta=0.5):
    ### input
    # y_train=[0,1,0,3,3,5,7] type:list, a list of image label of train data
    # y_test=[0,7,5,6,4,4,5] type:list, a list of image label of test data
    # n_clients=3, type:int, the number of clients
    # beta=0.5, type:float, the param of Dir(beta)
    ### output
    # train_idx_dict={0:[0,2],1:[1,7]} type:dict, key is the client id, value is the train data id of the clien indexed from the y_train generated by Dir(beta)
    # test_idx_dict={0:[0,4],1:[2,5]} type:dict, key is the client id, value is the test data id of the clien indexed from the y_train generated by Dir(beta)
    min_size = 1  # the sample size lower bound of  each client
    train_label=np.array(y_train)
    test_label=np.array(y_test)
    num = len(train_label)
    n_classes = len(np.unique(train_label))
    assert num > n_clients * min_size, f'The number of sample should be ' \
                                        f'greater than' \
                                        f' {n_clients * min_size}.'
    size = 0
    while size < min_size: 
        #repeat the dirichlet sampling unless the size exceeds the minimum
        idx_slice_train = [[] for _ in range(n_clients)]
        idx_slice_test = [[] for _ in range(n_clients)]
        for k in range(n_classes):
            # for label k, sampling
            idx_k_train = np.where(train_label == k)[0] #np.shape (num,1)
            idx_k_test = np.where(test_label==k)[0]
            np.random.shuffle(idx_k_train)
            np.random.shuffle(idx_k_test)
            prop_raw = np.random.dirichlet(np.repeat(beta, n_clients))
            prop = (np.cumsum(prop_raw) * len(idx_k_train)).astype(int)[:-1]
            idx_slice_train = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_slice_train, np.split(idx_k_train, prop))
            ]
            prop = (np.cumsum(prop_raw) * len(idx_k_test)).astype(int)[:-1]
            idx_slice_test = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_slice_test, np.split(idx_k_test, prop))
            ]
        size = min([len(idx_j) for idx_j in idx_slice_train])
    train_idx_dict={}
    test_idx_dict={}
    for i in range(n_clients):
        # each list is sorted by class label k, needs shuffling
        np.random.shuffle(idx_slice_train[i])
        train_idx_dict[i]=idx_slice_train[i]
        np.random.shuffle(idx_slice_test[i])
        test_idx_dict[i]=idx_slice_test[i]
    
    return train_idx_dict,test_idx_dict

def split_class(y_train,y_test,n_clients=3,n_class=3):
    ### input
    # y_train=[0,1,0,3,3,5,7] type:list, a list of image label of train data
    # y_test=[0,7,5,6,4,4,5] type:list, a list of image label of test data
    # n_clients=3, type:int, the number of clients
    # n_class=3, type:int, the number of class of samples that each client has

    ### output
    # train_idx_dict={0:[0,2],1:[1,7]} type:dict, key is the client id, value is the train data id of the client indexed from the y_train generated by the number of class
    # test_idx_dict={0:[0,4],1:[2,5]} type:dict, key is the client id, value is the test data id of the client indexed from the y_train generated by the number of class
    
    min_size = 1  # the sample size lower bound of  each client
    beta = 1000
    train_label=np.array(y_train)
    test_label=np.array(y_test)
    n_classes = len(np.unique(train_label))
    assert n_classes >= n_class, f'The number of categories per client should'\
                                    f'be less than '\
                                    f'{n_classes+1}.'
    map_cate2client=[[] for _ in range(n_classes)]
    for i in range(n_clients):
        #Client i has n_class kinds of labels
        sampling=np.random.choice(n_classes,n_class,replace=False)#no repeat
        for j in sampling:
            map_cate2client[j].append(i)
    print(map_cate2client)
    size = 0
    while size < min_size: 
        #repeat the dirichlet sampling unless the size exceeds the minimum
        idx_slice_train = [[] for _ in range(n_clients)]
        idx_slice_test = [[] for _ in range(n_clients)]
        for k in range(n_classes):
            # for label k, sampling
            if len(map_cate2client[k])==0: #some categories will not dispatch to any client
                continue
            idx_k_train = np.where(train_label == k)[0] #np.shape (num,1)
            idx_k_test = np.where(test_label == k)[0]
            np.random.shuffle(idx_k_train)
            np.random.shuffle(idx_k_test)
            prop_raw = np.random.dirichlet(np.repeat(beta, len(map_cate2client[k])))
            prop = (np.cumsum(prop_raw) * len(idx_k_train)).astype(int)[:-1]
            for idx_j, idx in zip(map_cate2client[k], np.split(idx_k_train, prop)):
                idx_slice_train[idx_j]=idx_slice_train[idx_j] + idx.tolist()            
            prop = (np.cumsum(prop_raw) * len(idx_k_test)).astype(int)[:-1]
            for idx_j, idx in zip(map_cate2client[k], np.split(idx_k_test, prop)):
                idx_slice_test[idx_j]=idx_slice_test[idx_j] + idx.tolist()
        size = min([len(idx_j) for idx_j in idx_slice_train])
    train_idx_dict={}
    test_idx_dict={}
    for i in range(n_clients):
        # each list is sorted by class label k, needs shuffling
        np.random.shuffle(idx_slice_train[i])
        train_idx_dict[i]=idx_slice_train[i]
        np.random.shuffle(idx_slice_test[i])
        test_idx_dict[i]=idx_slice_test[i]
    return train_idx_dict,test_idx_dict