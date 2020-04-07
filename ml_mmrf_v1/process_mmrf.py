import pandas as pd
import numpy as np
import warnings
from fancyimpute import KNN as KNN_impute


def get_sequential_tensor(df, id_col, feature_name, day_start, day_end = None, granularity = 30, maxT = 66):
    """
    Given the index column, feature name column, start day and end day, extract time series data at a prespecified
    granularity and max time (currently 30 days default)
    """
    # Return a tensor where time is along granularity-day intervals and the feature is in the column depicted
    if day_end is None:
        rdf = df[[id_col, feature_name, day_start]]
        rdf.loc[:,'start'] = df[day_start]
        rdf.loc[:,'end']   = df[day_start]
    else:
        rdf = df[[id_col, feature_name]]
        rdf.loc[:,'start'] = df[day_start]
        rdf.loc[:,'end']   = df[day_end]
    rdf.rename(columns={id_col:'id',feature_name:'feature'}, inplace=True)
    maxD = np.max((rdf['start'].values.max(),rdf['end'].values.max()))
    maxTcalc = maxD//granularity
    if maxTcalc>=maxT:
        warnings.warn('\tget_sequential_tensor %s: Found days that exceed set maximum time'%(feature_name))
    rdf.loc[:,'start'] = rdf['start']//granularity
    rdf.loc[:,'end']   = rdf['end']//granularity
    if len(rdf.feature.unique())<10:
        print ('\ttget_sequential_tensor: feature name/values:',feature_name,rdf.feature.unique())
    patlist = []
    flist   = []
    olist   = []
    for ii,pat_idx in enumerate(np.sort(rdf.id.unique())):
        pdf  = rdf[rdf.id==pat_idx]
        vals = np.zeros((maxT,))
        obs  = np.zeros((maxT,))
        for ft, st, en in zip(pdf.feature, pdf.start, pdf.end):
            if st<0:
                continue
            if en+1>=maxT:
                continue
            if np.any(np.isnan([st, en, ft])):
                continue
            st = int(st)
            en = int(en)
            vals[st:en+1] = ft
            obs[st:en+1]  = 1
        patlist.append(pat_idx); flist.append(vals); olist.append(obs)
            
    patient_ids = np.array(patlist); feature_data= np.array(flist); obs_data = np.array(olist)
    print ('\ttget_sequential_tensor: output shapes:',patient_ids.shape, feature_data.shape, obs_data.shape)
    return patient_ids, (feature_data, obs_data)

def parse_labs(df_pp, granularity = 30, maxT=66):
    """
    Use patient_visit file to extract lab data
    """
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
    return pids, data_tensor, obs_tensor, lab_names


def parse_baseline(df_pp, df_ppv, ia_version = 'ia13'):
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
    
    # add heavy chain/light chain feature to baseline features 
    serum_labs = ['D_LAB_serum_iga', 'D_LAB_serum_igg', 'D_LAB_serum_igm','D_LAB_serum_lambda', \
                  'D_LAB_serum_kappa', 'D_LAB_serum_m_protein']
    mtype_dfs = get_mtype(df_ppv, serum_labs) #hc_df, igg_df, iga_df, igm_df, kappa_df, lambda_df

    print ('parse_baselines: do mean imputation on missing data in baseline')
    merged.fillna(merged.mean(0), axis=0, inplace = True)
    
    genetic_data = pd.read_csv('%s_pca_embeddings.csv'%ia_version, delimiter=',')
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
    return clin_gen_imputed.values[:,0], clin_gen_imputed.values[:,1:], clin_gen_imputed.columns[1:]

def get_mtype(df, serum_labs): 
    df_l = df[['PUBLIC_ID', 'VISIT', 'VISITDY']+serum_labs]
    df_l.rename(columns = dict([(k,k.split('D_LAB_')[1]) for k in serum_labs]), inplace=True)
    serum_labs = np.array([k.split('D_LAB_')[1] for k in serum_labs])
    df_l = df_l[df_l.VISIT==0].reset_index(drop=True)
    df_l = df_l.groupby('PUBLIC_ID').mean()
    medians = df_l[serum_labs].median(0)
    maxval  = (5*(1+medians))
    clipped = df_l[serum_labs].clip(upper = maxval, axis=1)
    df_l.loc[:,serum_labs] = clipped
    df_l.fillna(df_l.mean(0), axis=0, inplace = True)
    
    # get heavy chain and light chain flag 
    hc_df = df_l['serum_m_protein']>0.5
    hc_df.rename('heavy_chain', inplace=True)

    # get specific multiple myeloma type 
    df_l['serum_igg'] = df_l['serum_igg'] * 100. 
    df_l['serum_iga'] = df_l['serum_iga'] * 100. 
    df_l['serum_igm'] = df_l['serum_igm'] * 100. 
    df_l['kl_ratio']  = df_l['serum_kappa'] / df_l['serum_lambda']    
    
    igg_df = ((hc_df.astype(int) + (df_l['serum_igg']>1500).astype(int)) > 1).rename('igg_type', inplace=True)
    iga_df = ((hc_df.astype(int) + (df_l['serum_iga']>300).astype(int)) > 1).rename('iga_type', inplace=True)
    iga_df[(igg_df.astype(int) + iga_df.astype(int)) > 1] = 0.
    iga_df = iga_df.astype(bool)
    igm_df = ((hc_df == 1.) & (igg_df == 0.) & (iga_df == 0.)).rename('igm_type', inplace=True)
    kappa_df = (df_l['kl_ratio']>1.5).rename('kappa_type', inplace=True)
    lambda_df = (df_l['kl_ratio']<0.5).rename('lambda_type', inplace=True)
    
    return hc_df, igg_df, iga_df, igm_df, kappa_df, lambda_df
    
def merge_on_pids(all_pids, pdict, ddict):
    """
    Helper function to merge dictionaries
    
    all_pids: list of all patient ids
    pdict, ddict: data dictionaries indexed by feature name
    
    1) pdict[fname]: patient ids
    2) ddict[fname]: data tensor corresponding to each patient
    """
    set_ids = set(all_pids)
    for fname in pdict:
        set_ids = set_ids.intersection(set(pdict[fname]))
    list_ids = list(set_ids)
    list_ids.sort()
    print ('merge_on_pids: intersection of patient ids is',len(list_ids))
    
    maxT = 0
    for fname in ddict:
        maxT = np.max((maxT, ddict[fname][0].shape[1]))
    data = np.zeros((len(list_ids), maxT, len(pdict.keys())))
    obs  = np.zeros_like(data)
    for f_idx, fname in enumerate(pdict):
        pids_f, (data_f, obs_f) = pdict[fname], ddict[fname]
        pids_f    = list(pids_f)
        index_map = [pids_f.index(pid) for pid in list_ids]
        data[:,:maxT, f_idx] = data_f[index_map, :maxT]
        obs[:,:maxT, f_idx]  = obs_f[index_map, :maxT]
    print ('merge_on_pids: after merging, pat_ids, data, obs:', len(list_ids), data.shape, obs.shape)
    return np.array(list_ids), data, obs
        
def parse_treatments(df_trtresp, granularity = 30, maxT=66):
    """
    Use treatment_response files to obtain treatments
    """
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
    return pids, data_tensor, obs_tensor, treatment_fname


def parse_outcomes(df_pp, granularity = 30):
    """
    Extract outcomes from the PER_PATIENT file
    """
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
    return pids, y, e

