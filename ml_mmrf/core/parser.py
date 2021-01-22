# MMRF parser 
import os, sys, glob
import pickle
import pandas as pd
import numpy as np
import warnings
from fancyimpute import KNN as KNN_impute
from utils import *
from gen_pca import *

class MMRFParser:
    '''
    Authors: Zeshan M Hussain, Rahul G Krishnan

    The MMRF Parser class is responsible for taking the raw data stored in pandas 
    dataframes and converting it into sequential tensors (with associated masks to
    capture the missing data) at a specified granularity of time. 
    
    For example, if we specify granularity to be 60 and maxT to be 33 (which are the 
    default settings), then the treatment and labs tensors will be of size N x 33 x D, 
    where N is the number of and D is the feature dimension. Furthermore, suppose 
    t = 1,...,maxT; the time between t and t+1 is determined by granularity, which in this 
    case is 60 days (2 months).
    '''

    def __init__(self, fdir, ia_version, granularity, maxT, outcomes_type, recreate_splits, featset='full'):
        self.fdir = fdir 
        self.ia_version  = ia_version
        self.granularity = granularity
        self.maxT        = maxT
        self.outcomes_type  = outcomes_type
        self.recreate_splits= recreate_splits
        self.featset = featset

        # set up dictionary for dataset
        self.dataset = {}
        self.dataset['treatment'] = {}
        self.dataset['labs']      = {}
        self.dataset['baseline']  = {}
        self.dataset['outcomes']  = {}
        self.dataset['trt_outcomes'] = {}

    def load_files(self): 
        '''
        This function must be called first before running the parser, since it loads in 
        the flatfiles from the specified directory and stores the raw data in a pandas
        dataframe. Each dataframe from a corresponding file is stored as a value in a 
        dictionary (self.data_files), where the key is the name of the file (e.g. PER_PATIENT, 
        PER_PATIENT_VISIT, STAND_ALONE_TRTRESP, etc.).
        
        Args: 
            None 
        Returns: 
            None
        '''
        self.data_files = {}
        fdir = os.path.join(self.fdir,'CoMMpass_'+self.ia_version.upper()+'_FlatFiles/*.csv') 
        print ('Searching for files that match',fdir)
        for fullname in glob.glob(fdir):
            print (fullname)
            fname = os.path.basename(fullname).split('.')[0]
            if 'CoMMpass_IA13_' in fname:
                kname = fname.split('CoMMpass_IA13_')[1]
            elif 'CoMMpass_IA15_' in fname:
                kname = fname.split('CoMMpass_IA15_')[1]
            else: 
                kname = fname
            self.data_files[kname] = pd.read_csv(fullname, delimiter=',', encoding='latin-1')
        print (self.data_files.keys())

    def get_parsed_data(self): 
        '''
        Call this function at the end after parsing all the data.
        
        Args: 
            None
        Returns: 
            self.dataset: dictionary where key is data type (e.g. treatment, labs, baseline, etc.)
            and value is the data dictionary containing the data tensor, the mask tensor, the names
            of the features, and the patient IDs associated with the examples.
        '''
        return self.dataset

    def parse_treatments(self):
        """
        We use treatment_response files to obtain treatments across time along with the line 
        of therapy that each treatment is associated with. We restrict the treatments to 
        the top 10 treatments with respect to counts over the entire time course. The final 
        treatment tensor is of size N x maxT x num_treat, where each entry at time t for some 
        patient i is a binary vector of size 1 x num_treat. An element in the vector is 1 if 
        that treatment is given to patient i at time t and 0 otherwise.
        
        Args: 
            None
        Sets: 
            We don't return anything, but rather set the 'treatment' key in our 
            dataset dictionary. 
            data_tensor: tensor of size, N x maxT x num_treat
            obs_tensor: mask of size, N x maxT x num_treat (assumed to be all ones)
            pids: list of patient ids
            treatment_fname: names of the treatments includes + other features (such as line of therapy)
        """
        df_trtresp = self.data_files['STAND_ALONE_TRTRESP']; granularity = self.granularity; maxT = self.maxT
        top_10_combinations = df_trtresp.trtshnm.value_counts()[:10].index.to_numpy()
        treatments = []
        for comb in top_10_combinations:
            treatments += [k.strip() for k in comb.split('-')]
        treatment_fname = np.array(np.unique(treatments).tolist())
        print ('parse_treatments: treatments: ',treatment_fname)

        # Restrict to relevant columns
        treatment_df    = df_trtresp[['public_id','line','trtshnm','bmtx_day','trtstdy','trtendy']]

        # Create a binary indicator if treatment is in the set we know and care about
        for fname in treatment_fname:
            treatment_df.loc[:,fname] = np.where(treatment_df.trtshnm.str.contains(fname), 1, 0)

        # Include line in treatment list
        treatment_fname = np.array(treatment_fname.tolist()+['line'])
        print ('parse_treatments: adding line of therapy: ',treatment_fname)
        
        results_pids= {}
        results_data= {}
        for ii,fname in enumerate(treatment_fname):
            print ('parse_treatments:processing...',ii, fname)
            results_pids[fname], results_data[fname] = get_sequential_tensor(treatment_df,'public_id',fname,'trtstdy','trtendy', granularity = granularity, maxT=maxT)
        pids, data_tensor, obs_tensor = merge_on_pids(treatment_df.public_id.unique(), results_pids, results_data)
        print ('parse_treatments:',len(pids), data_tensor.shape, obs_tensor.shape)

        # Add parsed data to dataset dictionary 
        self.dataset['treatment']['pids'] = pids; self.dataset['treatment']['data'] = data_tensor
        self.dataset['treatment']['obs']  = obs_tensor; self.dataset['treatment']['names'] = treatment_fname

    def parse_labs(self, add_kl_ratio=False, add_pd_feats=False):
        """
        Use patient_visit file to extract specified lab data. We include the baseline lab
        values in our final tensors. Note that we do mean imputation for only the baseline 
        lab values. For each lab value, we clip the value based on the median. Although not 
        done in this function, we forward fill the longitudinal missing lab values when loading
        the data on the fly (see load_mmrf() in data.py).
        
        Args: 
            None
        Sets: 
            We don't return anything, but rather set the 'labs' key in our 
            dataset dictionary. 
            data_tensor: tensor of size, N x maxT x num_labs
            obs_tensor: mask of size, N x maxT x num_labs 
            pids: list of patient ids
            lab_names: names of the labs
        """
        df_pp = self.data_files['PER_PATIENT_VISIT']; granularity = self.granularity; maxT = self.maxT
        if self.featset == 'full': 
            lab_names = ['D_LAB_cbc_abs_neut', 'D_LAB_chem_albumin', 'D_LAB_chem_bun', 'D_LAB_chem_calcium', 'D_LAB_chem_creatinine',
                'D_LAB_chem_glucose', 'D_LAB_cbc_hemoglobin', 'D_LAB_serum_kappa', 'D_LAB_serum_m_protein', 'D_LAB_cbc_platelet',
                'D_LAB_chem_totprot', 'D_LAB_cbc_wbc', 'D_LAB_serum_iga', 'D_LAB_serum_igg', 'D_LAB_serum_igm','D_LAB_serum_lambda']
        elif self.featset == 'serum_igs': 
            lab_names = ['D_LAB_serum_kappa', 'D_LAB_serum_m_protein', 'D_LAB_serum_iga', \
                 'D_LAB_serum_igg', 'D_LAB_serum_igm','D_LAB_serum_lambda', 'D_LAB_urine_24hr_m_protein']
        else: 
            assert self.featset in ['full', 'serum_igs'], 'invalid featset specified'
        pd_names = ['AT_SERUMMCOMPONE', 'AT_URINEMCOMPONE', 'AT_ONLYINPATIENT', 'AT_ONLYINPATIENT2', 'AT_DEVELOPMENTOF']
        if add_pd_feats: 
            df = df_pp[['PUBLIC_ID','VISIT','VISITDY']+lab_names+pd_names]
        else: 
            df = df_pp[['PUBLIC_ID','VISIT','VISITDY']+lab_names]
        df.rename(columns = dict([(k,k.split('D_LAB_')[1]) for k in lab_names]), inplace=True)
        lab_names = np.array([k.split('D_LAB_')[1] for k in lab_names])
        
        # truncate based on the median
        medians = df[lab_names].median(0)
        maxval  = (5*(1+medians))
        clipped = df[lab_names].clip(upper = maxval, axis=1)
        print ('parse_labs: clipped values to 5x median (before/after)\n',pd.concat([df[lab_names].max(0), clipped[lab_names].max(0)],axis=1))
        df.loc[:,lab_names] = clipped
        if add_pd_feats: 
            df.rename(columns = dict([(k,k.split('AT_')[1]) for k in pd_names]), inplace=True)
            pd_names  = np.array([k.split('AT_')[1] for k in pd_names])
            for pd_name in pd_names: 
                print(pd_name)
                df[pd_name][df[pd_name] == 'Checked'] = 1
                df[pd_name][df[pd_name].isna()] = 0
            df = df.infer_objects()
            print (f'parse_labs: added and binarized progressive disease features')
        
        # extract baseline labs, impute missing values at baseline with mean 
        df_bl = df[df.VISIT==0].reset_index(drop=True)
        df_bl = df_bl.groupby('PUBLIC_ID').mean()
        df_bl.fillna(df_bl.mean(0), axis=0, inplace = True)
        bline_ids = df_bl.index.tolist()
        
        if add_pd_feats: 
            lab_names = np.concatenate((lab_names,pd_names))
        
        results_pids= {}
        results_data= {}
        for ii,fname in enumerate(lab_names):
            print ('\nparse_labs:processing...',ii, fname)
            results_pids[fname], results_data[fname] = get_sequential_tensor(df,'PUBLIC_ID',fname,'VISITDY', granularity = granularity, maxT=maxT)
            # append baseline results to the beginning of the tensors
            subset_data= df_bl[fname].values.reshape(-1,1)
            idx = [bline_ids.index(pid) for pid in results_pids[fname]]
            ndata        = results_data[fname][0]
            ndata[:,[0]] = subset_data[idx]
            nobs         = results_data[fname][1]
            nobs[:,[0]]  = 1.
            results_data[fname] = (ndata, nobs)
            
        pids, data_tensor, obs_tensor = merge_on_pids(df.PUBLIC_ID.unique(), results_pids, results_data)
        lnames = lab_names.tolist()
        kappa_idx   = lnames.index('serum_kappa'); lambda_idx = lnames.index('serum_lambda')
        data_tensor[...,[kappa_idx]] = np.abs(data_tensor[...,[kappa_idx]])
        if add_kl_ratio:     
            kl_tensor   = data_tensor[...,[kappa_idx]] / (data_tensor[...,[lambda_idx]]+1e-5)
            kl_obs_tensor= obs_tensor[...,[kappa_idx]]*obs_tensor[...,[lambda_idx]]
            kl_tensor[np.where(kl_obs_tensor==0.)] = 0.
            for i in range(kl_tensor.shape[0]): 
                kl_median = np.median(kl_tensor[i][np.where(kl_obs_tensor[i]==1.)].squeeze())
                maxval = (2.5*(1+kl_median))
                np.clip(kl_tensor[i],a_min=None,a_max=500.,out=kl_tensor[i])
            data_tensor = np.concatenate((data_tensor, kl_tensor),axis=-1)
            obs_tensor  = np.concatenate((obs_tensor, kl_obs_tensor),axis=-1)
            lab_names   = np.concatenate((lab_names,np.array(['kl_ratio'])))
        print ('parse_labs:',len(pids), data_tensor.shape, obs_tensor.shape)
        for mn,mx,fn in zip(np.nanmin(data_tensor, axis=(0,1)), np.nanmax(data_tensor, axis=(0,1)), lab_names):
            print ('parse_labs (name/min/max)',fn,mn,mx)

        # Add parsed data to dataset dictionary
        self.dataset['labs']['pids'] = pids; self.dataset['labs']['data'] = data_tensor
        self.dataset['labs']['obs']  = obs_tensor; self.dataset['labs']['names'] = lab_names 

    def parse_baseline(self):
        """
        Extract demographic data (age, gender), ISS stage, beta2_microglobulin (clipped), multiple 
        myeloma subtype (IgG, IgA type, etc.), and output of PCA on genomic data. We do mean imputation
        on all missing baseline data, except for genetic data, for which we do knn (k=5) imputation.
        
        Args: 
            None
        Sets: 
            We don't return anything, but rather set the 'labs' key in our 
            dataset dictionary. 
            bdata: matrix of size, N x num_baseline
            bpids: list of patient ids
            bnames: names of baseline features
        """
        df_pp = self.data_files['PER_PATIENT']; df_ppv = self.data_files['PER_PATIENT_VISIT']
        ia_version = self.ia_version
        df = df_pp[['PUBLIC_ID', 'R_ISS', 'D_PT_age', 'D_PT_gender', 'ecog']] # 'line1sct'
        df.rename(columns={'R_ISS':'iss','D_PT_age':'age','D_PT_gender':'gender'}, inplace=True)
        
        # Add beta2 microglobulin into baseline tensor
        baseline_labs = ['D_LAB_serum_beta2_microglobulin']
        df_bl = df_ppv[['PUBLIC_ID','VISIT','VISITDY']+baseline_labs]
        df_bl.rename(columns = dict([(k,k.split('D_LAB_')[1]) for k in baseline_labs]), inplace=True)
        baseline_labs = np.array([k.split('D_LAB_')[1] for k in baseline_labs])
        df_bl = df_bl[df_bl.VISIT==0].reset_index(drop=True)
        df_bl = df_bl.groupby('PUBLIC_ID').mean()
        merged= pd.merge(df, df_bl, on='PUBLIC_ID', sort=True).drop(['VISIT','VISITDY'],axis=1)
        medians = merged[baseline_labs].median(0)
        maxval  = (5*(1+medians))
        clipped = merged[baseline_labs].clip(upper = maxval, axis=1)
        print ('parse_baselines: clipped values to 5x median (before/after)\n',pd.concat([merged[baseline_labs].max(0), clipped[baseline_labs].max(0)],axis=1))
        merged.loc[:,baseline_labs] = clipped
        
        # add heavy chain/light chain feature to baseline features as well as myeloma type of patients
        serum_labs = ['D_LAB_serum_iga', 'D_LAB_serum_igg', 'D_LAB_serum_igm','D_LAB_serum_lambda', \
                      'D_LAB_serum_kappa', 'D_LAB_serum_m_protein']
        mtype_dfs = get_mtype(df_ppv, serum_labs) #hc_df, igg_df, iga_df, igm_df, kappa_df, lambda_df
        
        print ('parse_baselines: do mean imputation on missing data in baseline')
        merged.fillna(merged.mean(0), axis=0, inplace = True)
        if os.path.exists(f'../output/folds_{self.outcomes_type}.pkl'): 
            print('[generating pca embeddings using pickled folds]')
            genetic_data = gen_pca_embeddings(train_test_file=f'../output/folds_{self.outcomes_type}.pkl', FDIR_sh=self.fdir)
        else: 
            print('[generating pca embeddings using random train/test split]')
            genetic_data = gen_pca_embeddings(train_test_file=None, FDIR_sh=self.fdir)
#         print('[reading in pca embeddings from ia13 csv.]')
#         genetic_data = pd.read_csv('./ia13_pca_embeddings.csv', delimiter=',') 
        genetic_data = genetic_data[['PUBLIC_ID','PC1','PC2','PC3','PC4','PC5']]
        # find indices without genetic data & fill values as missing
        diff_px      = np.setdiff1d(merged.values[:,0], genetic_data.values[:,0])
        missing_gen  = pd.DataFrame(diff_px, columns=['PUBLIC_ID'])
        for k in genetic_data.columns[1:]:
            missing_gen[k] = np.nan
        concat_gen   = pd.concat([genetic_data, missing_gen]).reset_index(drop=True)
        merged_clin_gen = pd.merge(merged, concat_gen, on='PUBLIC_ID', sort=True)
        
        print ('parse_baselines: doing knn(k=5) imputation for missing genomic data')
        X_filled_knn = KNN_impute(k=5).fit_transform(merged_clin_gen.values[:,1:]) 
        clin_gen_imputed = pd.DataFrame(np.concatenate([merged_clin_gen.values[:,[0]], X_filled_knn], axis=1), columns=merged_clin_gen.columns)
        for mtype_df in mtype_dfs:  
            clin_gen_imputed = pd.merge(clin_gen_imputed, mtype_df, on='PUBLIC_ID', sort=True)
        print ('parse_baselines: result',clin_gen_imputed.shape)
        bpids, bdata, bnames = (clin_gen_imputed.values[:,0], clin_gen_imputed.values[:,1:], clin_gen_imputed.columns[1:])
        
        # Add parsed data to dataset dictionary
        self.dataset['baseline']['pids'] = bpids; self.dataset['baseline']['data'] = bdata
        self.dataset['baseline']['obs']  = np.ones_like(bdata); self.dataset['baseline']['names'] = bnames

    def parse_outcomes(self):
        """
        Extract mortality outcomes from the PER_PATIENT file. We differentiate between patients
        who were observed to die after some amount of time ('uncensored') and those who don't 
        have a recorded deathdy but rather have an associated 'lstalive' or 'lastdy' (corresponding
        to their last day in the study). These are ('censored') patients. 
        
        Args: 
            None
        Sets: 
            We don't return anything, but rather set the 'outcomes' key in our 
            dataset dictionary. 
            y: size N x 1, array containing the 'time to event' value for each patient
            e: size N x 1, binary vector where 1 refers to 'uncensored' or 'observed' 
            and 0 refers to 'censored'
            pids: patient ids 
            names: name of feature (i.e. 'mortality')
        """
        df_pp = self.data_files['PER_PATIENT']; granularity = self.granularity
        df_restricted = df_pp[['PUBLIC_ID','D_PT_lstalive', 'D_PT_lastdy', 'D_PT_deathdy']]
        
        # Uncensored outcomes
        # First select patients who have the event (death)
        df_death = df_restricted.dropna(subset=['D_PT_deathdy'])
        assert (df_death.PUBLIC_ID.unique().shape[0]==df_death.shape[0]),'duplicates found'
        death_vals = df_death[['PUBLIC_ID','D_PT_deathdy']].values
        pids_y1  = death_vals[:,0]
        y1       = death_vals[:,1]/float(granularity)
        e1       = np.ones_like(y1)
        
        # Censored outcomes
        # Remove patients who have the event 
        df_subset            = df_restricted[~df_restricted['PUBLIC_ID'].isin(pids_y1.ravel().tolist())].reset_index(drop=True)
        df_alive_or_censored = df_subset[['PUBLIC_ID','D_PT_lstalive','D_PT_lastdy']]
        df_alive_or_censored = df_alive_or_censored.fillna(0)
        df_alive_or_censored['censored_date'] = df_alive_or_censored[['D_PT_lstalive', 'D_PT_lastdy']].values.max(1)
        df_alive_or_censored = df_alive_or_censored[['PUBLIC_ID','censored_date']]
        df_alive_or_censored = df_alive_or_censored[df_alive_or_censored['censored_date']>0].reset_index(drop=True)
        pids_y0  = df_alive_or_censored.values[:,0]
        y0       = df_alive_or_censored.values[:,1]/float(granularity)
        e0       = np.zeros_like(y0)
        assert len(set(pids_y0.tolist()).intersection(set(pids_y1.tolist())))==0,'Found intersection between dead and assumed not dead'
        pids, y, e = np.concatenate([pids_y0.ravel(), pids_y1.ravel()],0), np.concatenate([y0.ravel(), y1.ravel()],0), np.concatenate([e0.ravel(), e1.ravel()],0)
        print ('parse_outcomes: ',pids.shape, y.shape, e.shape)

        self.dataset['outcomes']['pids'] = pids
        self.dataset['outcomes']['data'] = y
        self.dataset['outcomes']['obs']  = e
        self.dataset['outcomes']['names']= np.array(['mortality'])

    def parse_trt_outcomes(self, line=2): 
        """
        TODO: need to change this description; parsing in this function has changed.
        
        xxExtract treatment response from STAND_ALONE_TRTRESP file. There are six canonical 
        xxtreatment responses in MM: stringent Complete Response (sCR), Complete Response (CR),
        xxVery Good Partial Response (VGPR), Partial Response (PR), Stable Disease (SD), 
        xxand Partial Disease (PD). In this function, if from_gateway is set to True, then 
        xxwe label patients as PD or nonPD (after second line therapy). We generate these labels
        xxdirectly from the MMRF gateway and save them in .csv files. Otherwise, we generate 
        xxthe labels from the STAND_ALONE_TRTRESP file.
        
        Args: 
            None
        Sets: 
            We don't return anything, but rather set the 'trt_outcomes' key in our 
            dataset dictionary. 
            y: size N x 1, array containing the treatment response label
            e: size N x 1, binary vector where 1 refers to 'uncensored' or 'observed' 
                and 0 refers to 'censored'
            pids: patient ids 
            names: names of features
        """ 
        trt_df = self.data_files['STAND_ALONE_TRTRESP']; ppv_df = self.data_files['PER_PATIENT_VISIT']
        
        tdf = trt_df[['public_id', 'line', 'therstdy', 'therendy']]
        tdf.rename(columns={'public_id': 'id'}, inplace=True)
        pdf = ppv_df[['PUBLIC_ID', 'VISITDY', 'AT_RESPONSEASSES', 'AT_TREATMENTRESP']]
        pdf.rename(columns={'PUBLIC_ID': 'id', 'VISITDY': 'vday', 'AT_RESPONSEASSES': 'aday', 'AT_TREATMENTRESP': 'tresp'}, inplace=True)

        patlist = []; labellist = []; obslist = []

        pd_pat = ['MMRF 1534', 'MMRF_1570', 'MMRF_1759', 'MMRF_1502', 'MMRF_1822', 'MMRF_2125', 'MMRF_1056', 'MMRF_1842', 'MMRF_1995', 'MMRF_1711', 'MMRF_2445', 'MMRF_2574', 'MMRF_1401', 'MMRF_1637', 'MMRF_1216', 'MMRF_2114', 'MMRF_1569', 'MMRF_1293', 'MMRF_1131', 'MMRF_1042', 'MMRF_1110', 'MMRF_1916', 'MMRF_1122', 'MMRF_2535', 'MMRF_1929', 'MMRF_1550', 'MMRF_1683', 'MMRF_1627', 'MMRF_2170', 'MMRF_1639', 'MMRF_1408', 'MMRF_2008', 'MMRF_1522', 'MMRF_2150', 'MMRF_1555', 'MMRF_1625', 'MMRF_1308', 'MMRF_1889', 'MMRF_1991', 'MMRF_2102', 'MMRF_1479', 'MMRF_1415', 'MMRF_1146', 'MMRF_2328', 'MMRF_1084', 'MMRF_2394', 'MMRF_1981', 'MMRF_1671', 'MMRF_2531', 'MMRF_1269', 'MMRF_1817', 'MMRF_1866', 'MMRF_1196', 'MMRF_1049', 'MMRF_1587', 'MMRF_1974', 'MMRF_1221', 'MMRF_1935', 'MMRF_1971', 'MMRF_1394', 'MMRF_1064', 'MMRF_1189', 'MMRF_1275', 'MMRF_2543', 'MMRF_2055', 'MMRF_1858', 'MMRF_1085', 'MMRF_2026', 'MMRF_1359', 'MMRF_2363', 'MMRF_1499', 'MMRF_1137', 'MMRF_1596', 'MMRF_2572', 'MMRF_2240', 'MMRF_1694', 'MMRF_2373', 'MMRF_1927', 'MMRF_2411', 'MMRF_1163', 'MMRF_1579', 'MMRF_1876', 'MMRF_1112', 'MMRF_1638', 'MMRF_2310', 'MMRF_2106', 'MMRF_1556', 'MMRF_1949', 'MMRF_1810', 'MMRF_1624', 'MMRF_1511', 'MMRF_2072', 'MMRF_1020', 'MMRF_1829', 'MMRF_1228', 'MMRF_2167', 'MMRF_1177', 'MMRF_2225', 'MMRF_1906', 'MMRF_2613', 'MMRF_1513', 'MMRF_1910', 'MMRF_1156', 'MMRF_1201', 'MMRF_1912', 'MMRF_1792', 'MMRF_1074', 'MMRF_2253', 'MMRF_1668', 'MMRF_1157', 'MMRF_2541', 'MMRF_1345', 'MMRF_1188', 'MMRF_2301', 'MMRF_1541', 'MMRF_2401', 'MMRF_1739', 'MMRF_2589', 'MMRF_1149', 'MMRF_1780', 'MMRF_2244', 'MMRF_1774', 'MMRF_2001', 'MMRF_2438', 'MMRF_1933', 'MMRF_1656', 'MMRF_2093', 'MMRF_2404', 'MMRF_1013', 'MMRF_1267', 'MMRF_2126', 'MMRF_1229', 'MMRF_1744', 'MMRF_1236', 'MMRF_1356', 'MMRF_1823', 'MMRF_1700', 'MMRF_1525', 'MMRF_2562', 'MMRF_2196', 'MMRF_1079', 'MMRF_1224', 'MMRF_1041', 'MMRF_1917', 'MMRF_1377', 'MMRF_2119', 'MMRF_1300', 'MMRF_1613', 'MMRF_2185', 'MMRF_1289', 'MMRF_1252', 'MMRF_1564', 'MMRF_1219', 'MMRF_1430', 'MMRF_1474', 'MMRF_2402', 'MMRF_1208', 'MMRF_1031', 'MMRF_1931', 'MMRF_2505', 'MMRF_2087', 'MMRF_2595', 'MMRF_1432', 'MMRF_1992', 'MMRF_2195', 'MMRF_1082', 'MMRF_1433', 'MMRF_1740', 'MMRF_1257', 'MMRF_1790', 'MMRF_2085', 'MMRF_1068', 'MMRF_2605', 'MMRF_2461', 'MMRF_2011', 'MMRF_1501', 'MMRF_1335', 'MMRF_1213', 'MMRF_1462', 'MMRF_1534']

        nonpd_pat = ['MMRF_2388', 'MMRF_1718', 'MMRF_1689', 'MMRF_1039', 'MMRF_1071', 'MMRF_1785', 'MMRF_2557', 'MMRF_1783', 'MMRF_2329', 'MMRF_1060', 'MMRF_2059', 'MMRF_1090', 'MMRF_2089', 'MMRF_1210', 'MMRF_1328', 'MMRF_1510', 'MMRF_2074', 'MMRF_1284', 'MMRF_1117', 'MMRF_1067', 'MMRF_1324', 'MMRF_1466', 'MMRF_1713', 'MMRF_1891', 'MMRF_1957', 'MMRF_2380', 'MMRF_2124', 'MMRF_1048', 'MMRF_2564', 'MMRF_1024', 'MMRF_1286', 'MMRF_2194', 'MMRF_1972', 'MMRF_1030', 'MMRF_1458', 'MMRF_2317', 'MMRF_1947', 'MMRF_1452', 'MMRF_1202', 'MMRF_2153', 'MMRF_2475', 'MMRF_2471', 'MMRF_1886', 'MMRF_1099', 'MMRF_1908', 'MMRF_2268', 'MMRF_1593', 'MMRF_1913', 'MMRF_1730', 'MMRF_1922', 'MMRF_2024', 'MMRF_1440', 'MMRF_1518', 'MMRF_2122', 'MMRF_2064', 'MMRF_1766', 'MMRF_1055', 'MMRF_2281', 'MMRF_1771', 'MMRF_2503', 'MMRF_2477', 'MMRF_1542', 'MMRF_1251', 'MMRF_1434', 'MMRF_2257', 'MMRF_1264', 'MMRF_2305', 'MMRF_2366', 'MMRF_1354', 'MMRF_2336', 'MMRF_1782', 'MMRF_1951', 'MMRF_1763', 'MMRF_1618', 'MMRF_1736', 'MMRF_1011', 'MMRF_2229', 'MMRF_1222', 'MMRF_1670', 'MMRF_1512', 'MMRF_1490', 'MMRF_2386', 'MMRF_1533', 'MMRF_2490', 'MMRF_2608', 'MMRF_2174', 'MMRF_2138', 'MMRF_1092', 'MMRF_1309', 'MMRF_1983', 'MMRF_2172', 'MMRF_1403', 'MMRF_1148', 'MMRF_1127', 'MMRF_1108', 'MMRF_1726', 'MMRF_2419', 'MMRF_2606', 'MMRF_1900', 'MMRF_2367', 'MMRF_1978', 'MMRF_1130', 'MMRF_1735', 'MMRF_1921', 'MMRF_2330', 'MMRF_2538', 'MMRF_2413', 'MMRF_2054', 'MMRF_1097', 'MMRF_2095', 'MMRF_1703', 'MMRF_1413', 'MMRF_2599', 'MMRF_1021', 'MMRF_1772', 'MMRF_1153', 'MMRF_1032', 'MMRF_1424', 'MMRF_1727', 'MMRF_1446', 'MMRF_1861', 'MMRF_2190', 'MMRF_1395', 'MMRF_1178', 'MMRF_2601', 'MMRF_1950', 'MMRF_1825', 'MMRF_1254', 'MMRF_1838', 'MMRF_2412', 'MMRF_1320', 'MMRF_1665', 'MMRF_1577', 'MMRF_1261', 'MMRF_1280', 'MMRF_1086', 'MMRF_1628', 'MMRF_2014', 'MMRF_2344', 'MMRF_1717', 'MMRF_1786', 'MMRF_2250', 'MMRF_2191', 'MMRF_2211', 'MMRF_1496', 'MMRF_1094', 'MMRF_1223', 'MMRF_1965', 'MMRF_2200', 'MMRF_1936', 'MMRF_1205', 'MMRF_1155', 'MMRF_1833', 'MMRF_2198', 'MMRF_2091', 'MMRF_1186', 'MMRF_2453', 'MMRF_1928', 'MMRF_2385', 'MMRF_1193', 'MMRF_2217', 'MMRF_2039', 'MMRF_2313', 'MMRF_2076', 'MMRF_1167', 'MMRF_2306', 'MMRF_1731', 'MMRF_1568', 'MMRF_2292', 'MMRF_1539', 'MMRF_1837', 'MMRF_2082', 'MMRF_2545', 'MMRF_1500', 'MMRF_1519', 'MMRF_1143', 'MMRF_2204', 'MMRF_1565', 'MMRF_2478', 'MMRF_2289', 'MMRF_2501', 'MMRF_1233', 'MMRF_2442', 'MMRF_1418', 'MMRF_2017', 'MMRF_1650', 'MMRF_1491', 'MMRF_1629', 'MMRF_1113', 'MMRF_1061', 'MMRF_1760', 'MMRF_2457', 'MMRF_2427', 'MMRF_1845', 'MMRF_1285', 'MMRF_2273', 'MMRF_1758', 'MMRF_1461', 'MMRF_2118', 'MMRF_1504']

        censored = ['MMRF_1759', 'MMRF_2574', 'MMRF_1131', 'MMRF_1929', 'MMRF_1683', 'MMRF_1408', 'MMRF_1625', 'MMRF_1866', 'MMRF_1221', 'MMRF_2026', 'MMRF_1927', 'MMRF_1112', 'MMRF_1810', 'MMRF_2072', 'MMRF_1829', 'MMRF_1177', 'MMRF_2225', 'MMRF_2253', 'MMRF_2401', 'MMRF_2001', 'MMRF_1933', 'MMRF_1917', 'MMRF_2185', 'MMRF_1430', 'MMRF_2505', 'MMRF_2087', 'MMRF_1432', 'MMRF_1740', 'MMRF_1335', 'MMRF_1213', 'MMRF_2388', 'MMRF_1718', 'MMRF_2557', 'MMRF_1090', 'MMRF_2317', 'MMRF_1202', 'MMRF_1913', 'MMRF_1518', 'MMRF_1055', 'MMRF_2305', 'MMRF_1951', 'MMRF_2386', 'MMRF_1533', 'MMRF_2608', 'MMRF_1148', 'MMRF_2095', 'MMRF_1491', 'MMRF_1629', 'MMRF_1061']

        remove_pat = ['MMRF_1682'] # this is specific for line 2
        eps = 5 # number of days we are willing to look beyond the true range of the therapy line
        for ii,pat_idx in enumerate(np.sort(pdf.id.unique())):
            if pat_idx in remove_pat: 
                continue
            ipdf = pdf[pdf.id == pat_idx]
            itdf = tdf[tdf.id == pat_idx]
            l = itdf[itdf['line'] == line]
            if l.empty: 
                continue 
                
            if pat_idx in pd_pat and pat_idx in censored: 
                labellist.append(1); obslist.append(0); patlist.append(pat_idx)
            elif pat_idx in pd_pat and pat_idx not in censored: 
                labellist.append(1); obslist.append(1); patlist.append(pat_idx)
            elif pat_idx in nonpd_pat and pat_idx in censored: 
                labellist.append(0); obslist.append(0); patlist.append(pat_idx)
            elif pat_idx in nonpd_pat and pat_idx not in censored: 
                labellist.append(0); obslist.append(1); patlist.append(pat_idx)
            else: 
                print(f'{pat_idx} not in lists even though they have line 2 info.')

        pids = np.array(patlist); y = np.array(labellist); names = np.array(['notPD', 'PD']); e = np.array(obslist)
        assert e.shape == y.shape

        self.dataset['trt_outcomes']['pids'] = pids 
        self.dataset['trt_outcomes']['data'] = y
        self.dataset['trt_outcomes']['obs']  = e
        self.dataset['trt_outcomes']['names']= names
    
