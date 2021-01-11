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

healthy_mins_max = {
    'cbc_abs_neut':(2., 7.5,1/3.), # abs neutrophil count (3.67, 1.), (2.83, 4.51)
    'chem_albumin':(34, 50,1/8.), # chemical albumin (43.62, 2.77), (41.30, 45.94)
    'chem_bun':(2.5, 7.1,1/5.), #BUN # reference range, (4.8, 1.15)
    'chem_calcium':(2.2, 2.7,2.), #Calcium, (2.45, 0.125)
    'chem_creatinine':(66, 112,1/36.), # creatinine, (83., 24.85), (62.22, 103.77)
    'chem_glucose':(3.9, 6.9,1/5.), # glucose, (4.91, 0.40), (4.58, 5.24)
    'cbc_hemoglobin':(13., 17.,1), # hemoglobin (12.90, 15.64), (8.86, 1.02)
    'chem_ldh':(2.33, 4.67,1/3.), #LDH, (3.5, 0.585)
    'serum_m_protein':(0.1, 1.1, 1), # M protein (<3 g/dL is MGUS, any presence of protein is pathological); am just using the data mean/std for this, (0.85, 1.89)
    'urine_24hr_m_protein':(0.0, 0.1, 1), # Urine M protein 
    'cbc_platelet':(150, 400,1/60.), # platelet count (206.42, 334.57), (270.5, 76.63)
    'chem_totprot':(6, 8,1/6.), # total protein, (7, 0.5)
    'urine_24hr_total_protein':(0, 0.23, 1), # 
    'cbc_wbc':(3, 10,1/4.), # WBC  (5.71, 8.44), (7.07, 1.63)
    'serum_iga':(0.85, 4.99, 1.), # IgA, (2.92, 1.035)
    'serum_igg':(6.10, 16.16,1/10.), # IgG, (11.13, 2.515)
    'serum_igm':(0.35, 2.42,1), #IgM, (1.385, 0.518)
    'serum_lambda':(0.57, 2.63, 1/2.), #serum lambda, (1.6, 0.515)
    'serum_kappa':(.33, 1.94,1/8.), #serum kappa , (1.135, 0.403)
    'serum_beta2_microglobulin':(0.7, 1.80, 1/3.), #serum_beta2_microglobulin,
    'serum_c_reactive_protein':(0.0, 1., 1.), #serum_c_reactive_protein,
    'kl_ratio': (.26, 1.65, 1/60.), 
    'SERUMMCOMPONE': (0.,0.,1.), # don't scale the binary features 
    'URINEMCOMPONE': (0.,0.,1.),
    'ONLYINPATIENT': (0.,0.,1.), 
    'ONLYINPATIENT2': (0.,0.,1.),
    'DEVELOPMENTOF': (0.,0.,1.)
}

class MMRFCleaner:
    '''
    The MMRF Cleaner class is responsible for taking the tensors outputted by the MMRF class 
    and normalizing the values (depending on the type of the data: baseline, labs, or treatment).
    '''
    
    def __init__(self, dataset, outcomes_type='mortality'):
        '''
        We copy over the data curated in the Parser class into self.clean_dset.
        
        Args: 
            dataset: the data dictionary that we get from get_parsed_data() in MMRFParser object 
            that contains the data and mask tensors of the lab, treatment, and baseline data 
            outcomes_type: a string that specifies over which outcome we will be restricting
            the patient ids 
        Returns: 
            None
        '''
        self.clean_dset = {}
        self.outcomes_type = outcomes_type
        # we store different outcome data depending on the specified outcomes type (i.e. either mortality or treatment response)
        if outcomes_type == 'mortality':     
            self.clean_dset['patient_ids'] = dataset['outcomes']['pids']
            self.clean_dset['y_data']      = dataset['outcomes']['data']
            self.clean_dset['event_obs']   = dataset['outcomes']['obs']
        elif outcomes_type == 'trt_resp': 
            self.clean_dset['patient_ids'] = dataset['trt_outcomes']['pids']
            self.clean_dset['y_data']      = dataset['trt_outcomes']['data']
            self.clean_dset['event_obs']   = dataset['trt_outcomes']['obs']
            self.clean_dset['tr_names']    = dataset['trt_outcomes']['names']
            self.clean_dset['ym_data']     = dataset['outcomes']['data']
            self.clean_dset['ce']     = dataset['outcomes']['obs']
        
        #  we restrict all data to share a global ordering over patient ids (from outcomes)
        pts = self.clean_dset['patient_ids'].tolist()
        for k in ['treatment','labs','baseline']:
            pts_src = dataset[k]['pids'].tolist()
            idx_map = np.array([pts_src.index(v) for v in pts])
            self.clean_dset[k+'_data'] = dataset[k]['data'][idx_map]
            self.clean_dset[k+'_m']    = dataset[k]['obs'][idx_map]
            self.clean_dset[k+'_names']= dataset[k]['names']
        
        print(f'shape of treatment data: {self.clean_dset["treatment_data"].shape}')
        print(f'shape of lab data: {self.clean_dset["labs_data"].shape}')
        print(f'shape of baseline data: {self.clean_dset["baseline_data"].shape}')

    def get_cleaned_data(self): 
        '''
        Call this function last after cleaning the baseline and lab data. 
        
        Args: 
            None
        Returns: 
            self.clean_dset: final data dictionary containing the normalized tensors
            self.outcomes_type: str that contains the outcome type over which we restricted 
            the patients 
        '''
        return self.clean_dset, self.outcomes_type

    def clean_baseline(self): 
        """
        We z-score normalize a subset of the baseline data (age, gender, and genetic PCA data). 
        TODO: add way for allowing user to specify how they want to normalize the data. 
        
        Args: 
            None
        Sets: 
            We don't return anything, but rather make a copy of the baseline data and set it 
            as the value of 'baseline_data_clean' in the self.clean_dset data dictionary. We 
            proceed to normalize the data and alter 'baseline_data_clean' in place. 
            
        """
        # Before
        print ('A] Before cleaning')
        print ('idx, featurename, min, mean, max')
        self.clean_dset['baseline_data_clean']= np.copy(self.clean_dset['baseline_data']).astype('float32')
        for i,f in enumerate(self.clean_dset['baseline_names']):
            print (i,f,self.clean_dset['baseline_data'][:,i].min(),self.clean_dset['baseline_data'][:,i].mean(),self.clean_dset['baseline_data'][:,i].max())
        print ('------')
        # Age = mean and standard deviation normalize
        age_mean = self.clean_dset['baseline_data_clean'][:,[1]].mean(0,keepdims=True)
        age_std  = np.std(self.clean_dset['baseline_data_clean'][:,[1]], keepdims=True)
        self.clean_dset['baseline_data_clean'][:,[1]] = (self.clean_dset['baseline_data_clean'][:,[1]]-age_mean)/age_std

        # Gender = -1,1
        self.clean_dset['baseline_data_clean'][:,2]   = self.clean_dset['baseline_data_clean'][:,2]*2-3
        # Mean and standard deviation normalize PCA features
        pca_mean = self.clean_dset['baseline_data_clean'][:,3:10].mean(0,keepdims=True)
        pca_std  = np.std(self.clean_dset['baseline_data_clean'][:,3:10], keepdims=True)
        self.clean_dset['baseline_data_clean'][:,3:10] = (self.clean_dset['baseline_data_clean'][:,3:10]-pca_mean)/pca_std
        
        print ('------')
        print ('B] After cleaning')
        for i,f in enumerate(self.clean_dset['baseline_names']):
            print (i,f,self.clean_dset['baseline_data_clean'][:,i].min(),self.clean_dset['baseline_data_clean'][:,i].mean(),self.clean_dset['baseline_data_clean'][:,i].max(),np.any(np.isnan(self.clean_dset['baseline_data_clean'])))

    def clean_labs(self):
        """
        We normalize the lab values in a relative way, such that >0 is in the unhealthy range. 
        Specifically, we subtract the healthy max and then multiply the lab values by a scaling 
        factor. 
        
        TODO: add way for allowing user to specify how they want to normalize the data. 
        
        Args: 
            healthy_mins_max: a dictionary that has the lab value name as the key and a tuple 
            containing the healthy min, healthy max, and the scaling factor as the value.
        Sets: 
            We don't return anything, but rather make a copy of the lab values and set it 
            as the value of 'labs_data_clean' in the self.clean_dset data dictionary. We 
            proceed to normalize the data and alter 'labs_data_clean' in place. 
            
        """
        print ('A] Before cleaning')
        print ('idx, featurename, min, mean, max')
        for i,f in enumerate(self.clean_dset['labs_names']):
            print (i,f,np.nanmin(self.clean_dset['labs_data'][...,i]),np.nanmean(self.clean_dset['labs_data'][...,i]),np.nanmax(self.clean_dset['labs_data'][...,i]))
        print ('------')

        self.clean_dset['labs_data_clean']= np.copy(self.clean_dset['labs_data']).astype('float32')
        for idx in range(self.clean_dset['labs_names'].shape[0]):
            healthy_max   = healthy_mins_max[self.clean_dset['labs_names'][idx]][1]
            scale         = healthy_mins_max[self.clean_dset['labs_names'][idx]][2]
            print ('B] Subtracting healthy max ',healthy_max,' from', self.clean_dset['labs_names'][idx],' and scaling: ',scale)
            self.clean_dset['labs_data_clean'][...,idx] = (self.clean_dset['labs_data_clean'][...,idx]-healthy_max)*scale

        print ('------')
        print ('C] After cleaning')
        for i,f in enumerate(self.clean_dset['labs_names']):
            print (i,f,np.nanmin(self.clean_dset['labs_data_clean'][...,i]),np.nanmean(self.clean_dset['labs_data_clean'][...,i]),np.nanmax(self.clean_dset['labs_data_clean'][...,i]))


