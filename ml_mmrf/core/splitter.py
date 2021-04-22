import os, sys, glob
import pickle
import pandas as pd
import numpy as np
import warnings
import matplotlib.pylab as plt
import seaborn as sns
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-large')
plt.rc('ytick', labelsize='x-large')
plt.rc('axes', labelsize='x-large')
plt.rc('font', size=12)

from fancyimpute import KNN as KNN_impute
from utils import *


class MMRFSplitter: 
    '''
    The MMRF Splitter class is responsible for generating the test and train sets 
    for the MMRF cohort. It also generates k folds (k specified by the user but 
    generally we have used k=5) of the training data for k-fold cross validation. 
    '''

    def __init__(self, clean_dset, outcomes_type): 
        '''    
        Args: 
            clean_dset: the data dictionary that we get from get_cleaned_data() of MMRFCleaner object, 
            containing the normalized data and mask tensors of the lab, treatment, and baseline data 
            outcomes_type: a string that specifies over which outcome we restricted the patient ids 
        Returns: 
            None
        '''
        self.dset = clean_dset 
        self.outcomes_type = outcomes_type
        self.train_valid_folds = None
        self.testidx = None

    def split_balanced_general(self, idx_list, event, trfrac = .75): 
        '''    
        Helper function that splits a list of indices into training and validate indices balanced on 
        some set of outcomes. 
        
        Args: 
            idx_list: an array containing range of indices [1,...,N], where N is the total number of examples in the dataset to split
            event: an array of event labels that we wish to balance the split on 
            trfrac: fraction of examples that will be in the training set 
        Returns: 
            trlist: list of training indices
            valist: list of held-out indices
        '''
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
        
    def split_data(self, nfolds = 5, recreate_splits=True, seed=0):
        '''    
        Main function that splits the data into training and test sets, and then 
        the training set into five folds. Note that there is a single (global) test 
        set, but each fold has a training and validate set. 
        
        Args: 
            nfolds: int specifiying number of folds to split training data
        Sets: 
            train_valid_folds: dictionary where key is fold number and value is tuple of train
            and valid indices
            testidx: array of patient indices in test set
            nfolds: number of folds to split training data
        '''
        # train_valid_folds, testidx = get_splits(new_dset['event_obs'], nfolds=5)
        
        np.random.seed(0)
        if recreate_splits: 
            event_obs = self.dset['event_obs']
            idx_list  = np.arange(event_obs.shape[0])
            trainidx, testidx = self.split_balanced_general(idx_list, event_obs)

            if seed != 0: 
                np.random.seed(seed)
            folds_idx = {}; pids = {}
            for fold in range(nfolds):
                idx_list = np.arange(trainidx.shape[0])
                event_fold= event_obs[trainidx]
                fi_idx_train, fi_idx_valid = self.split_balanced_general(idx_list, event_fold)
                fi_tr, fi_va = trainidx[fi_idx_train], trainidx[fi_idx_valid]
                folds_idx[fold] = (fi_tr, fi_va)
                pids[fold] = np.concatenate((self.dset['patient_ids'][fi_tr],self.dset['patient_ids'][fi_va]))
                print ('Fold: ',fold,fi_idx_train.shape[0], fi_idx_valid.shape[0])
                print ('Event obs: ',event_obs[fi_tr].sum(), event_obs[fi_va].sum())
            
            train_valid_pids  = pids
            train_valid_folds = folds_idx; test_pids = self.dset['patient_ids'][testidx]
            print('[Saving splits in .pkl]')
            with open(f'../output/folds_{self.outcomes_type}.pkl','wb') as f:
                pickle.dump((train_valid_folds, testidx, train_valid_pids, test_pids),f)
        else: 
            print('[Reading in splits]')
            with open(f'../output/folds_{self.outcomes_type}.pkl', 'rb') as f: 
                train_valid_folds, testidx, train_valid_pids, test_pids = pickle.load(f)
            
        self.train_valid_pids  = train_valid_pids
        self.test_pids         = test_pids
        self.train_valid_folds = train_valid_folds
        self.testidx = testidx
        self.nfolds  = nfolds

    def get_splits(self): 
        '''    
        Function that returns the train_valid_folds dictionary and the test indices. Call 
        only after running split_data().
        
        Args: 
            None
        Sets: 
            train_valid_folds: dictionary where key is fold number and value is tuple of train
            and valid indices
            testidx: array of patient indices in test set
        '''
        assert self.train_valid_folds is not None, 'need to run split_data() function in MMRFSplitter Class'
        assert self.testidx is not None, 'need to run split_data() function in MMRFSplitter Class'
        return self.train_valid_folds, self.testidx, self.train_valid_pids, self.test_pids

    def get_split_data(self): 
        '''    
        This function basically operationalizes the data split by copying over the data into a 
        list of final_datasets (of size nfolds) where each element is a dictionary that contains 
        the global test set (determined by the test indices) and the training and validate sets 
        corresponding to a particular fold.
        
        Args: 
            None
        Sets: 
            final_datasets: list of size nfolds that contains a data dictionary corresponding to 
            each fold
        '''
        assert self.train_valid_folds is not None, 'need to run split_data() function in MMRFSplitter Class'
        assert self.testidx is not None, 'need to run split_data() function in MMRFSplitter Class'
        final_datasets = []
        for fold in range(self.nfolds):
            print ('Saving fold ', fold)
            final_dataset = {}
            final_dataset[fold] = {}
            for tvt in ['train','valid','test']:
                if  tvt =='test':
                    idx = self.testidx
                elif tvt =='train':
                    idx = self.train_valid_folds[fold][0]
                elif tvt == 'valid':
                    idx = self.train_valid_folds[fold][1]
                else:
                    raise NotImplemented()
                final_dataset[fold][tvt] = {}
                final_dataset[fold][tvt]['pids']               = self.dset['patient_ids'][idx]
                # labs
                final_dataset[fold][tvt]['x']                  = self.dset['labs_data_clean'][idx]
                final_dataset[fold][tvt]['m']                  = self.dset['labs_m'][idx]
                final_dataset[fold][tvt]['feature_names_x']    = self.dset['labs_names']
                # outcomes
                final_dataset[fold][tvt]['ys_seq'] = self.dset['y_data'][idx].reshape(-1,1)
                final_dataset[fold][tvt]['ce']     = (1.-self.dset['event_obs'][idx]).reshape(-1,1)
                final_dataset[fold][tvt]['feature_names_y']    = self.dset['names']
                # baseline
                final_dataset[fold][tvt]['b']      = self.dset['baseline_data_clean'][idx]
                final_dataset[fold][tvt]['feature_names']    = self.dset['baseline_names']
                # treatments
                final_dataset[fold][tvt]['a']      = self.dset['treatment_data'][idx]
                final_dataset[fold][tvt]['m_a']    = self.dset['treatment_m'][idx]
                final_dataset[fold][tvt]['feature_names_a']    = self.dset['treatment_names']
            
            # Forward fill missing data in longitudinal lab tensors
            for tvt in ['train','valid','test']:
                x_new  = np.copy(final_dataset[fold][tvt]['x'])
                x_new[final_dataset[fold][tvt]['m']==0] = np.nan
                x_new_filled  = []
                for k in range(x_new.shape[-1]):
                    x_new_filled.append(pd.DataFrame(x_new[...,k]).fillna(method='ffill', axis=1).values[...,None])
                x_new_filled  = np.concatenate(x_new_filled, axis=-1)
                assert not np.any(np.isnan(x_new_filled)),'should not be any nans'
                final_dataset[fold][tvt]['x'] = x_new_filled
            
            # Restrict (in train/valid set) to patients with atleast two longitudinal observations
            T_lb = 2
            print(f'train...')
            for tvt in ['train']:
                M     = final_dataset[fold][tvt]['m']
                M_t   = (M.sum(-1)>1.)*1.
                all_t = M_t.sum(-1)
                keep_idx = np.argwhere(all_t>T_lb).ravel()
                if tvt == 'train':
                    if self.outcomes_type == 'trt_resp' or 'bin' in self.outcomes_type: 
                        Y = final_dataset[fold][tvt]['ys_seq']
                        C = final_dataset[fold][tvt]['ce']
                        print ('Before: N censored/total ',C.sum(), C.shape[0], C.sum()/C.shape[0])
                        for i in range(np.max(Y)+1): 
                            print (f'Before: Y class {i}, N: {len(np.where(Y == i)[0])}')
                    else:
                        C = final_dataset[fold][tvt]['ce']
                        print ('Before: N censored/total ',C.sum(), C.shape[0], C.sum()/C.shape[0])
                for kk in ['a','x','m','ys_seq','ce','b','pids','m_a']:
                    final_dataset[fold][tvt][kk] = np.copy(final_dataset[fold][tvt][kk][keep_idx])
                if tvt == 'train':
                    if self.outcomes_type == 'trt_resp' or 'bin' in self.outcomes_type: 
                        Y = final_dataset[fold][tvt]['ys_seq']
                        C = final_dataset[fold][tvt]['ce']
                        print ('After: N censored/total ',C.sum(), C.shape[0], C.sum()/C.shape[0])
                        for i in range(np.max(Y)+1): 
                            print (f'After: Y class {i}, N: {len(np.where(Y == i)[0])}')
                    else:
                        C = final_dataset[fold][tvt]['ce']
                        print ('After: N censored/total ',C.sum(), C.shape[0], C.sum()/C.shape[0])
            print (final_dataset[fold]['train']['x'].shape)
            for tvt in ['valid', 'test']: 
                print(f'{tvt}...')
                if self.outcomes_type == 'trt_resp' or 'bin' in self.outcomes_type: 
                    Y = final_dataset[fold][tvt]['ys_seq']
                    C = final_dataset[fold][tvt]['ce']
                    print ('Before: N censored/total ',C.sum(), C.shape[0], C.sum()/C.shape[0])
                    for i in range(np.max(Y)+1): 
                        print (f'Before: Y class {i}, N: {len(np.where(Y == i)[0])}')
                else:
                    C = final_dataset[fold][tvt]['ce']
                    print ('Before: N censored/total ',C.sum(), C.shape[0], C.sum()/C.shape[0])

            final_datasets.append(final_dataset)
            
        return final_datasets
