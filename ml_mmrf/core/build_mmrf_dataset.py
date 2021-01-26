'''
Authors: Zeshan M Hussain, Rahul G Krishnan

This file is the high level script that generates the cleaned tensors from the 
MMRF data. Ensure that you have downloaded the flatfiles from the MMRF gateway,
which can be found here: https://research.themmrf.org/. Running this script will 
generate cleaned tensors in .pkl files for each fold, as well as which indices go in 
each fold. This information is stored in folds.pkl. 

'''

import os, sys, glob
import pandas as pd
import numpy as np
import pickle
import warnings
from fancyimpute import KNN as KNN_impute
from distutils.util import strtobool
from parser import MMRFParser
from cleaner import MMRFCleaner
from splitter import MMRFSplitter
from utils import *
from argparse import ArgumentParser


def main(args, save_intermediates=True):
    mm_parser = MMRFParser(**vars(args))
    mm_parser.load_files()
    # parse, parse, parse
    mm_parser.parse_treatments() 
    mm_parser.parse_labs(add_kl_ratio=False, add_pd_feats=False)
    mm_parser.parse_baseline()
    mm_parser.parse_outcomes()
    mm_parser.parse_trt_outcomes()
    dataset = mm_parser.get_parsed_data()
    print(dataset.keys())
    
    if save_intermediates: 
        with open(f'../output/1_mmrf_dataset_{args.granularity//30}mos_type.pkl','wb') as f:
            pickle.dump(dataset, f)

    # clean, clean, clean
    mm_cleaner = MMRFCleaner(dataset, outcomes_type=args.outcomes_type)
    mm_cleaner.clean_baseline()
    mm_cleaner.clean_labs()
    clean_dataset, outcomes_type = mm_cleaner.get_cleaned_data()

    # split, split, split 
    mm_splitter = MMRFSplitter(clean_dataset, outcomes_type)
    nfolds = 5
    mm_splitter.split_data(nfolds=nfolds, recreate_splits=args.recreate_splits)

    final_datasets = mm_splitter.get_split_data()
    if save_intermediates: 
        for fold in range(nfolds): 
            final_dataset = final_datasets[fold]
            fname = '../output/cleaned_mm'+str(fold)+'_2mos.pkl'
            fname = f'../output/cleaned_mm{fold}_2mos_{args.trtrep}.pkl'
            if args.outcomes_type == 'trt_resp': 
                fname = f'../output/cleaned_mm{fold}_2mos_tr_{args.trtrep}.pkl'
            with open(fname,'wb') as f:
                pickle.dump(final_dataset, f)
    print('[---Processing Complete---]')

if __name__ == '__main__':
    # get path to ML_MMRF files 
    parser = ArgumentParser()
    parser.add_argument('--fdir', type=str, default='/afs/csail.mit.edu/group/clinicalml/datasets/multiple_myeloma/ia15', help='path to MMRF flat files')
    parser.add_argument('--ia_version', type=str, default='IA15', help='version of MMRF data')
    parser.add_argument('--outcomes_type', type=str, default='mortality', help='what outcome to balance train/test on + which outcome to store in Y; mortality or trt_resp')
    parser.add_argument('--granularity', type=int, default=60, help='specifies the granularity of time with which you wish to process the data (e.g. 60 means time between time step t and time step t+1 is 60 days or 2 months)')
    parser.add_argument('--maxT', type=int, default=33, help='max time step at which to stop processing longitudinal data at the above granularity')
    parser.add_argument('--recreate_splits', type=strtobool, default=False, help='if you want to recreate folds, then set to True')
    parser.add_argument('--featset', type=str, default='full', help='subset of features to save; support for one of ["full", "serum_igs"]')
    parser.add_argument('--trtrep', type=str, default='ind', help='type of treatment representation to use')
    args = parser.parse_args()
    #assert args.ia_version in args.fdir, 'ia version and version associated with flatfiles do not match'
    np.random.seed(0)
    main(args)

    
    
