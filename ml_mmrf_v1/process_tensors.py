import numpy as np
import pandas as pd

def clean_baseline(new_dset, healthy_mins_max):
    # Before
    print ('A] Before cleaning')
    print ('idx, featurename, min, mean, max')
    new_dset['baseline_data_clean']= np.copy(new_dset['baseline_data']).astype('float32')
    for i,f in enumerate(new_dset['baseline_names']):
        print (i,f,new_dset['baseline_data'][:,i].min(),new_dset['baseline_data'][:,i].mean(),new_dset['baseline_data'][:,i].max())
    print ('------')
    # Age = mean and standard deviation normalize
    age_mean = new_dset['baseline_data_clean'][:,[1]].mean(0,keepdims=True)
    age_std  = np.std(new_dset['baseline_data_clean'][:,[1]], keepdims=True)
    new_dset['baseline_data_clean'][:,[1]] = (new_dset['baseline_data_clean'][:,[1]]-age_mean)/age_std

    # Gender = -1,1
    new_dset['baseline_data_clean'][:,2]   = new_dset['baseline_data_clean'][:,2]*2-3
    # Mean and standard deviation normalize PCA features
    pca_mean = new_dset['baseline_data_clean'][:,3:10].mean(0,keepdims=True)
    pca_std  = np.std(new_dset['baseline_data_clean'][:,3:10], keepdims=True)
    new_dset['baseline_data_clean'][:,3:10] = (new_dset['baseline_data_clean'][:,3:10]-pca_mean)/pca_std
    
    print ('------')
    print ('C] After cleaning')
    for i,f in enumerate(new_dset['baseline_names']):
        print (i,f,new_dset['baseline_data_clean'][:,i].min(),new_dset['baseline_data_clean'][:,i].mean(),new_dset['baseline_data_clean'][:,i].max(),np.any(np.isnan(new_dset['baseline_data_clean'])))

        
def clean_labs(new_dset, healthy_mins_max):
    print ('A] Before cleaning')
    print ('idx, featurename, min, mean, max')
    for i,f in enumerate(new_dset['labs_names']):
        print (i,f,np.nanmin(new_dset['labs_data'][...,i]),np.nanmean(new_dset['labs_data'][...,i]),np.nanmax(new_dset['labs_data'][...,i]))
    print ('------')

    new_dset['labs_data_clean']= np.copy(new_dset['labs_data']).astype('float32')
    for idx in range(new_dset['labs_names'].shape[0]):
        healthy_max   = healthy_mins_max[new_dset['labs_names'][idx]][1]
        scale         = healthy_mins_max[new_dset['labs_names'][idx]][2]
        print ('B] Subtracting healthy max ',healthy_max,' from', new_dset['labs_names'][idx],' and scaling: ',scale)
        new_dset['labs_data_clean'][...,idx] = (new_dset['labs_data_clean'][...,idx]-healthy_max)*scale

    print ('------')
    print ('C] After cleaning')
    for i,f in enumerate(new_dset['labs_names']):
        print (i,f,np.nanmin(new_dset['labs_data_clean'][...,i]),np.nanmean(new_dset['labs_data_clean'][...,i]),np.nanmax(new_dset['labs_data_clean'][...,i]))


        
def split_balanced(idx_list, event, trfrac = 0.7):
    print ('Total: ',idx_list.shape[0])
    assert idx_list.shape[0] == event.shape[0],'expecting same shapes'
    idx0 = np.where(event==0)[0]
    idx1 = np.where(event==1)[0]
    
    l0 = idx_list[idx0]
    l1 = idx_list[idx1]
    
    trlist, valist = [], []
    for lidx in [l0, l1]:
        N      = lidx.shape[0]
        shuf   = np.random.permutation(N)
        ntrain = int(trfrac*N)
        idx_train = lidx[shuf[:ntrain]]
        idx_val   = lidx[shuf[ntrain:]]
        trlist+= idx_train.tolist()
        valist+= idx_val.tolist()
    return np.array(trlist), np.array(valist)

def split_balanced_general(idx_list, event, trfrac = .7): 
    print ('Total: ',idx_list.shape[0])
    assert idx_list.shape[0] == event.shape[0],'expecting same shapes'
    idxs = []
    upper_bound = np.max(event)+1
    for i in range(upper_bound): 
        idxs.append(np.where(event==i)[0])
    
    # get l0,l1,l2,...
    lists = []
    for idxi in idxs: 
        lists.append(idx_list[idxi])
    
    trlist, valist = [], []
    for lidx in lists: 
        N      = lidx.shape[0]
        shuf   = np.random.permutation(N)
        ntrain = int(trfrac*N)
        idx_train = lidx[shuf[:ntrain]]
        idx_val   = lidx[shuf[ntrain:]]
        trlist += idx_train.tolist()
        valist += idx_val.tolist()
    return np.array(trlist), np.array(valist)
    
def get_splits(event_obs, nfolds = 5):
    np.random.seed(0)
    idx_list  = np.arange(event_obs.shape[0])
    trainidx, testidx = split_balanced_general(idx_list, event_obs)

    folds_idx = {}
    for fold in range(nfolds):
        idx_list = np.arange(trainidx.shape[0])
        event_fold= event_obs[trainidx]
        fi_idx_train, fi_idx_valid = split_balanced_general(idx_list, event_fold)
        fi_tr, fi_va = trainidx[fi_idx_train], trainidx[fi_idx_valid]
        folds_idx[fold] = (fi_tr, fi_va)
        print ('Fold: ',fold,fi_idx_train.shape[0], fi_idx_valid.shape[0])
        print ('Event obs: ',event_obs[fi_tr].sum(), event_obs[fi_va].sum())
    return folds_idx, testidx


