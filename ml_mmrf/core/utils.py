import pandas as pd
import numpy as np
import warnings
from fancyimpute import KNN as KNN_impute

def get_sequential_tensor(df, id_col, feature_name, day_start, day_end = None, granularity = 30, maxT = 66):
    """
    Given the index column, feature name column, start day and end day, extract time series data at a prespecified
    granularity and max time (currently 30 days default)
    
    Args: 
        df: the pandas dataframe in which the raw data is stored
        id_col: the column name of the dataframe, df, that IDs the features 
        (usually the patient ID)
        feature_name: longitudinal feature along which we will parse
        day_start: start day to parse
        day_end: final day to parse
        granularity: amount of time between subsequent time steps we wish 
        to enforce in the final tensor (e.g. if 60, then time between t and 
        t+1 is 60 days)
        maxT: max number of time steps to split the data into
    Returns: 
        patient_ids: array containing patient IDs associated with each sequential feature vector; size N x 1
        feature_data: sequential tensor of size N x maxT x D
        obs_data: mask of sequential tensor (indicating missing values); size N x maxT x D
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
    if feature_name == 'line': 
        hit_second_line = False
    for ii,pat_idx in enumerate(np.sort(rdf.id.unique())):
        pdf  = rdf[rdf.id==pat_idx]
        vals = np.zeros((maxT,))
        obs  = np.zeros((maxT,))
        for ft, st, en in zip(pdf.feature, pdf.start, pdf.end):
            if st<0 or st>=maxT:
                continue
            if en+1>=maxT:
                en=maxT-1
            if np.any(np.isnan([st, en, ft])):
                continue
            if feature_name == 'line' and ft == 2: 
                hit_second_line = True
            st = int(st)
            en = int(en)
            if pat_idx == 'MMRF_1064' and ft > 2: # hardcoding in fix for MMRF_1064
                continue
            vals[st:en+1] = ft
            obs[st:en+1]  = 1
        if feature_name == 'line': 
            hit_second_line = False
        patlist.append(pat_idx); flist.append(vals); olist.append(obs)
    
    patient_ids = np.array(patlist); feature_data= np.array(flist); obs_data = np.array(olist)
    print ('\ttget_sequential_tensor: output shapes:',patient_ids.shape, feature_data.shape, obs_data.shape)
    return patient_ids, (feature_data, obs_data)

def get_mtype(df, serum_labs): 
    """
    Helper function that generates the myeloma type of each patient based on their 
    serum Ig values.
    
    Args: 
        df: the pandas dataframe in which the raw data is stored ('PER_PATIENT_VISIT')
        serum_labs: array of serum lab value names that we will use to index the dataframe
    Returns: 
        hc_df: dataframe that contains binary vector for whether or not a patient has heavy chain 
        myeloma or not
        igg_df: binary label for igg type
        iga_df: binary label for iga type
        igm_df: binary label for igm type
        kappa_df: binary label for kappa type
        lambda_df: binary label for lambda type
    """
    df_l = df[['PUBLIC_ID', 'VISIT', 'VISITDY']+serum_labs]
    df_l.rename(columns = dict([(k,k.split('D_LAB_')[1]) for k in serum_labs]), inplace=True)
    serum_labs = np.array([k.split('D_LAB_')[1] for k in serum_labs])
    df_l = df_l[df_l.VISIT==0].reset_index(drop=True)
    df_l = df_l.groupby('PUBLIC_ID').mean()
    medians = df_l[serum_labs].median(0)
    maxval  = (15*(1+medians))
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
    
    # 1500 and 300 are cutoffs for IgG and IgA myeloma types, respectively
    igg_df = ((hc_df.astype(int) + (df_l['serum_igg']>1500).astype(int)) > 1).rename('igg_type', inplace=True)
    iga_df = ((hc_df.astype(int) + (df_l['serum_iga']>300).astype(int)) > 1).rename('iga_type', inplace=True)
    iga_df[(igg_df.astype(int) + iga_df.astype(int)) > 1] = 0.
    iga_df = iga_df.astype(bool)
    igm_df = ((hc_df == 1.) & (igg_df == 0.) & (iga_df == 0.)).rename('igm_type', inplace=True)
    
    # kappa and lambda type myeloma are determined by kappa/lambda ratio (>1.5 is kappa) and (<0.5 is lambda)
    kappa_df = (df_l['kl_ratio']>1.5).rename('kappa_type', inplace=True)
    lambda_df = (df_l['kl_ratio']<0.5).rename('lambda_type', inplace=True)
    
    true_df = df[['PUBLIC_ID', 'VISIT', 'D_IM_IGH_SITE', 'D_IM_IGL_SITE']]
    true_df = true_df[(true_df['VISIT'] == 0) & ((true_df['D_IM_IGH_SITE'].notna()) | (true_df['D_IM_IGL_SITE'].notna()))]
    hc_df = hc_df.to_frame(); igg_df = igg_df.to_frame(); iga_df = iga_df.to_frame()
    igm_df = igm_df.to_frame(); kappa_df = kappa_df.to_frame(); lambda_df = lambda_df.to_frame()
    
    for index, row in true_df.iterrows(): 
        pid = row['PUBLIC_ID']
        if isinstance(row['D_IM_IGH_SITE'], str): 
            if row['D_IM_IGH_SITE'].lower() == 'igg': 
                igg_df.loc[pid] = True; iga_df.loc[pid] = False; hc_df.loc[pid]  = True 
            elif row['D_IM_IGH_SITE'].lower() == 'iga': 
                igg_df.loc[pid] = False; iga_df.loc[pid] = True; hc_df.loc[pid] = True
        if isinstance(row['D_IM_IGL_SITE'], str): 
            if row['D_IM_IGL_SITE'].lower() == 'kappa': 
                kappa_df.loc[pid] = True; lambda_df.loc[pid] = False
            elif row['D_IM_IGL_SITE'].lower() == 'lambda':
                kappa_df.loc[pid] = False; lambda_df.loc[pid] = True
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
