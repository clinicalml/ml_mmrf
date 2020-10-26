import os, sys, glob
import pickle
import pandas as pd
import numpy as np
import warnings
from fancyimpute import KNN as KNN_impute
from utils import *

def gen_pca_embeddings(ia_version='ia15', trainidx=None, testidx=None, write_to_csv=False): 
    
    # TODO 
    FDIR_sh   = '/afs/csail.mit.edu/group/clinicalml/datasets/multiple_myeloma/ia15'
    qc_path   = os.path.join(FDIR_sh,f'MMRF_CoMMpass_{ia_version.upper()}_Seq_QC_Summary.csv')
    fpkm_path = os.path.join(FDIR_sh,f'MMRF_CoMMpass_{ia_version.upper()}a_E74GTF_Cufflinks_Gene_FPKM.txt')
    
    # load in files 
    data_files = {}
    fpkm_df = pd.read_csv(fpkm_path, delimiter=',', encoding='latin-1')
    design_df = pd.read_csv(qc_path, delimiter=',', encoding='latin-1')
    if ia_version == 'ia15': 
        fpkm_df.drop(1, inplace=True, axis=1) # drop gene location column
    design_df['sampleID'] = design_df['QC Link SampleName'].apply(lambda x: x.split('_')).apply(lambda x: '_'.join(x[:4]))
    design_df['cellType'] = design_df['QC Link SampleName'].apply(lambda x: x.split('_')).apply(lambda x: '_'.join(x[3:5]))
    
    # STEP 1: filter IDs in FPKM based on QC data and filter genes based on annotation file (GRCh37)
    
    
    # STEP 2: reproduce Seurat PCA functionality in scanpy (https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html)
    
    
    # STEP 3: write embeddings to csv file and/or return a Pandas dataframe
    if write_to_csv: 
        # write to csv 
    
    # return dataframe with columns: 'PUBLIC_ID', 'PC1', 'PC2', ...
    
    
    