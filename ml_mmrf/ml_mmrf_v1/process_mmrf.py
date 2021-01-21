import pandas as pd
import numpy as np
import warnings
import math
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
            vals[st:en+1] = ft
            obs[st:en+1]  = 1
        if feature_name == 'line' and not hit_second_line: 
            print(f'\ttget_sequential_tensor: did not hit second line for {pat_idx}')
        patlist.append(pat_idx); flist.append(vals); olist.append(obs)
    
    patient_ids = np.array(patlist); feature_data= np.array(flist); obs_data = np.array(olist)
    print ('\ttget_sequential_tensor: output shapes:',patient_ids.shape, feature_data.shape, obs_data.shape)
    return patient_ids, (feature_data, obs_data)

def parse_labs(df_pp, granularity = 30, maxT=66, add_kl_ratio=False, add_pd_feats=False):
    """
    Use patient_visit file to extract lab data
    """
    lab_names = ['D_LAB_cbc_abs_neut', 'D_LAB_chem_albumin', 'D_LAB_chem_bun', 'D_LAB_chem_calcium', 'D_LAB_chem_creatinine',
        'D_LAB_chem_glucose', 'D_LAB_cbc_hemoglobin', 'D_LAB_serum_kappa', 'D_LAB_serum_m_protein', 'D_LAB_cbc_platelet',
        'D_LAB_chem_totprot', 'D_LAB_cbc_wbc', 'D_LAB_serum_iga', 'D_LAB_serum_igg', 'D_LAB_serum_igm','D_LAB_serum_lambda']
#     lab_names = ['D_LAB_serum_kappa', 'D_LAB_serum_m_protein', 'D_LAB_serum_iga', \
#                  'D_LAB_serum_igg', 'D_LAB_serum_igm','D_LAB_serum_lambda', 'D_LAB_urine_24hr_m_protein']
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
    
#     genetic_data = pd.read_csv('%s_pca_embeddings.csv'%ia_version, delimiter=',')
    genetic_data = pd.read_csv('ia13_pca_embeddings.csv', delimiter=',')
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
    
    igg_df = ((hc_df.astype(int) + (df_l['serum_igg']>1500).astype(int)) > 1).rename('igg_type', inplace=True)
    iga_df = ((hc_df.astype(int) + (df_l['serum_iga']>300).astype(int)) > 1).rename('iga_type', inplace=True)
    iga_df[(igg_df.astype(int) + iga_df.astype(int)) > 1] = 0.
    iga_df = iga_df.astype(bool)
    igm_df = ((hc_df == 1.) & (igg_df == 0.) & (iga_df == 0.)).rename('igm_type', inplace=True)
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

def parse_trt_outcomes(trt_df, ppv_df, line=2, granularity=60): 
    """
    Extract treatment response from STAND_ALONE_TRTRESP file
    """ 
    tdf = trt_df[['public_id', 'line', 'therstdy', 'therendy']]
    tdf.rename(columns={'public_id': 'id'}, inplace=True)
    pdf = ppv_df[['PUBLIC_ID', 'VISITDY', 'AT_RESPONSEASSES', 'AT_TREATMENTRESP']]
    pdf.rename(columns={'PUBLIC_ID': 'id', 'VISITDY': 'vday', 'AT_RESPONSEASSES': 'aday', 'AT_TREATMENTRESP': 'tresp'}, inplace=True)
    
    patlist = []
    labellist   = []
    obslist = []
    
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
# #         import pdb; pdb.set_trace()
#         line_start = l.tail(1).therstdy; line_end = l.tail(1).therendy
        
#         last_resp = None; last_day = None
#         for vday, aday, resp in zip(ipdf.vday, ipdf.aday, ipdf.tresp): 
#             if math.isnan(vday) and math.isnan(aday): 
#                 continue    
#             if not math.isnan(aday): 
#                 true_day = aday//granularity
#             else: 
#                 true_day = vday//granularity
#             if true_day >= line_start.values[0]//granularity and true_day <= line_end.values[0]//granularity: 
#                 last_resp = resp
#         if pat_idx in pd_pat: 
#             last_resp = 'Progressive Disease'
#         elif pat_idx in nonpd_pat: 
#             last_resp = 'non Progressive Disease'
#         if last_resp == 'Progressive Disease': 
#             labellist.append(1)
#         else: 
#             labellist.append(0)
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
    return pids, y, e, names



