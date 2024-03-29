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
import copy
from fancyimpute import KNN as KNN_impute
from distutils.util import strtobool
from parser import MMRFParser
from cleaner import MMRFCleaner
from splitter import MMRFSplitter
from utils import *
from argparse import ArgumentParser


def main(args):
    parser_args = vars(copy.deepcopy(args))
    del parser_args['seed']
    del parser_args['save_intermediates']
    mm_parser = MMRFParser(**parser_args)
    mm_parser.load_files()
    # parse, parse, parse
    mm_parser.parse_treatments() 
    mm_parser.parse_labs(add_kl_ratio=False, add_pd_feats=False)
    mm_parser.parse_baseline()
    mm_parser.parse_os()
    mm_parser.parse_pfs()
    mm_parser.parse_trt_outcomes()
    dataset = mm_parser.get_parsed_data()
    print(dataset.keys())

    if args.save_intermediates: 
        with open(f'../output/1_mmrf_dataset_{args.granularity//30}mos_type.pkl','wb') as f:
            pickle.dump(dataset, f)

    # clean, clean, clean
    mm_cleaner = MMRFCleaner(dataset, outcomes_type=args.outcomes_type)
    mm_cleaner.clean_baseline()
    mm_cleaner.clean_labs()
    clean_dataset, outcomes_type = mm_cleaner.get_cleaned_data()
    if args.save_intermediates: 
        with open(f'../output/1_mmrf_dataset_clean_{args.granularity//30}mos_type.pkl','wb') as f:
            pickle.dump(clean_dataset, f)

    # split, split, split 
    mm_splitter = MMRFSplitter(clean_dataset, outcomes_type)
    nfolds = 5
    mm_splitter.split_data(nfolds=nfolds, recreate_splits=args.recreate_splits, seed=args.seed)

    final_datasets = mm_splitter.get_split_data()
    if args.save_intermediates: 
        for fold in range(nfolds): 
            final_dataset = final_datasets[fold]
            fname = f'../output/cleaned_mm{fold}_2mos_{args.outcomes_type}_{args.trtrep}_seed{args.seed}.pkl'
            with open(fname,'wb') as f:
                pickle.dump(final_dataset, f)
    print('[---Processing Complete---]')

if __name__ == '__main__':
    # get path to ML_MMRF files 
    parser = ArgumentParser()
    parser.add_argument('--fdir', type=str, default='/afs/csail.mit.edu/group/clinicalml/datasets/multiple_myeloma/ia15', help='path to MMRF flat files')
    parser.add_argument('--ia_version', type=str, default='IA15', help='version of MMRF data')
    parser.add_argument('--outcomes_type', type=str, default='os', help='what outcome to balance train/test on + which outcome to store in Y; ["os", "pfs", "pfs_bin", "pfs_asct", "pfs_asct_bin", "pfs_nonasct", "pfs_nonasct_bin", "trt_resp"]')
    parser.add_argument('--granularity', type=int, default=60, help='specifies the granularity of time with which you wish to process the data (e.g. 60 means time between time step t and time step t+1 is 60 days or 2 months)')
    parser.add_argument('--maxT', type=int, default=33, help='max time step at which to stop processing longitudinal data at the above granularity')
    parser.add_argument('--recreate_splits', type=strtobool, default=False, help='if you want to recreate folds, then set to True')
    parser.add_argument('--featset', type=str, default='full', help='subset of features to save; support for one of ["full", "serum_igs"]')
    parser.add_argument('--trtrep', type=str, default='ind', help='type of treatment representation to use')
    parser.add_argument('--rna_seq', type=str, default='bulk', help='type of treatment representation to use')
    parser.add_argument('--seed', type=int, default=0, help='max time step at which to stop processing longitudinal data at the above granularity')
    parser.add_argument('--save_intermediates', type=strtobool, default=True, help='whether to use optuna for optimization')
    args = parser.parse_args()
    #assert args.ia_version in args.fdir, 'ia version and version associated with flatfiles do not match'
    np.random.seed(args.seed)
    main(args)

    
    
