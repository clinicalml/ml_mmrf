# MMRF parser 
import os, sys, glob
import pickle
import pandas as pd
import numpy as np
import warnings
from fancyimpute import KNN as KNN_impute
from utils import *

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

    def __init__(self, fdir, ia_version, granularity, maxT):
        self.fdir = fdir 
        self.ia_version  = ia_version
        self.granularity = granularity
        self.maxT        = maxT

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
        for fullname in glob.glob(self.fdir+'/*.csv'):
            print (fullname)
            fname = os.path.basename(fullname).split('.')[0]
            if 'MMRF_CoMMpass_IA13_' in fname:
                kname = fname.split('MMRF_CoMMpass_IA13_')[1]
            elif 'MMRF_CoMMpass_IA15_' in fname:
                kname = fname.split('MMRF_CoMMpass_IA15_')[1]
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

    def parse_labs(self):
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
        lab_names = ['D_LAB_cbc_abs_neut', 'D_LAB_chem_albumin', 'D_LAB_chem_bun', 'D_LAB_chem_calcium', 'D_LAB_chem_creatinine',
            'D_LAB_chem_glucose', 'D_LAB_cbc_hemoglobin', 'D_LAB_serum_kappa', 'D_LAB_serum_m_protein', 'D_LAB_cbc_platelet',
            'D_LAB_chem_totprot', 'D_LAB_cbc_wbc', 'D_LAB_serum_iga', 'D_LAB_serum_igg', 'D_LAB_serum_igm','D_LAB_serum_lambda']
        df = df_pp[['PUBLIC_ID','VISIT','VISITDY']+lab_names]
        df.rename(columns = dict([(k,k.split('D_LAB_')[1]) for k in lab_names]), inplace=True)
        lab_names = np.array([k.split('D_LAB_')[1] for k in lab_names])
        
        # truncate based on the median
        medians = df[lab_names].median(0)
        maxval  = (5*(1+medians))
        clipped = df[lab_names].clip(upper = maxval, axis=1)
        print ('parse_labs: clipped values to 5x median (before/after)\n',pd.concat([df[lab_names].max(0), clipped[lab_names].max(0)],axis=1))
        df.loc[:,lab_names] = clipped
        
        # extract baseline labs, impute missing values at baseline with mean 
        df_bl = df[df.VISIT==0].reset_index(drop=True)
        df_bl = df_bl.groupby('PUBLIC_ID').mean()
        df_bl.fillna(df_bl.mean(0), axis=0, inplace = True)
        bline_ids = df_bl.index.tolist()
        
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
        df = df_pp[['PUBLIC_ID','R_ISS','D_PT_age','D_PT_gender', 'ecog']]
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
        
    #     genetic_data = pd.read_csv('%s_pca_embeddings.csv'%ia_version, delimiter=',')
        # TODO: add function that generates these embeddings
        genetic_data = pd.read_csv('./ia13_pca_embeddings.csv', delimiter=',') 
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

    def parse_trt_outcomes(self, from_gateway=False): 
        """
        Extract treatment response from STAND_ALONE_TRTRESP file. There are six canonical 
        treatment responses in MM: stringent Complete Response (sCR), Complete Response (CR),
        Very Good Partial Response (VGPR), Partial Response (PR), Stable Disease (SD), 
        and Partial Disease (PD). In this function, if from_gateway is set to True, then 
        we label patients as PD or nonPD (after second line therapy). We generate these labels
        directly from the MMRF gateway and save them in .csv files. Otherwise, we generate 
        the labels from the STAND_ALONE_TRTRESP file.
        
        Args: 
            from_gateway: if True, then generate labels from MMRF gateway and save in .csv files
        Sets: 
            We don't return anything, but rather set the 'trt_outcomes' key in our 
            dataset dictionary. 
            y: size N x 1, array containing the treatment response label
            
            pids: patient ids 
            names: names of features
        """ 
        trt_df = self.data_files['STAND_ALONE_TRTRESP']
        if from_gateway: 
            import pdb; pdb.set_trace()
            pd_pids    = pd.read_csv('./cohorts/pd_line2.csv')
            nonpd_pids = pd.read_csv('./cohorts/nonpd_line2.csv')
            pd_np      = [row['Patient'][:4] + '_' + row['Patient'][4:] for _, row in pd_pids.iterrows()]
            nonpd_np   = [row['Patient'][:4] + '_' + row['Patient'][4:] for _, row in nonpd_pids.iterrows()]
            y          = np.zeros(len(pd_np)+len(nonpd_np)).astype(int)
            y[:len(pd_np)] = 1
            pids       = np.array(pd_np + nonpd_np)
            names   = np.array(['notPD', 'PD'])
            print ('parse_outcomes (from MMRF gateway): ', pids.shape, y.shape)
        else: 
            # return best response after first line
            temp  = trt_df[(trt_df['line'] == 1) & (trt_df['trtstdy'] == trt_df['therstdy']) & (trt_df['bestrespsh'].notna())]
            bresp = temp[['public_id', 'trtshnm', 'bestrespsh']]
            pids  = bresp[['public_id']].values.squeeze()
            resp_dict = {
                'sCR': 0,
                'CR': 0, 
                'VGPR': 0, 
                'PR': 0, 
                'SD': 0, 
                'PD': 1
            }
            y = np.array([resp_dict[x] for x in list(bresp[['bestrespsh']].values.squeeze())])
            names = np.array(['CR', 'PR', 'NR'])
            print ('parse_outcomes: ',pids.shape, y.shape)

        self.dataset['trt_outcomes']['pids'] = pids 
        self.dataset['trt_outcomes']['data'] = y
        self.dataset['trt_outcomes']['obs']  = np.ones_like(y)
        self.dataset['trt_outcomes']['names']= names
    