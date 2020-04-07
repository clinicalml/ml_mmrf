import numpy as np
from lifelines.utils import concordance_index
import pandas as pd
import pickle    
import sys, os, warnings
import pandas as pd
from scipy import stats
from sklearn.preprocessing import OneHotEncoder

def make_censored_oh(Y, C, K):
    Yr,Cr = Y.ravel(), C.ravel()
    oh    = np.zeros((Y.shape[0], K))
    for k in range(Y.shape[0]):
        oh[k, Yr[k]]= 1.
    return oh
    
def digitize_outcomes(Y, Yvalid, Ytest, Ymax, K, method='quantiles'):
    if method=='quantiles':
        K_m_2     = K-1 # correct for bin edges
        probs     = np.arange(K_m_2+1)/float(K_m_2)
        bin_edges = stats.mstats.mquantiles(Y, probs)#[0, 2./6, 4./6, 1])
        bin_edges = bin_edges.tolist()
        bin_edges+= [Ymax]
        bin_edges = bin_edges
    elif method=='uniform':
        bin_edges = np.linspace(0, Ymax, K+1)
        bin_edges = bin_edges.tolist()
    else:
        raise ValueError('bad setting for method')
    predict = []
    for k in range(len(bin_edges)-1):
        predict.append((bin_edges[k]+bin_edges[k+1])/2.)
    predict = np.array(predict)
    
    Ytr  = np.digitize(Y.ravel(), bin_edges)-1
    Yva  = np.digitize(Yvalid.ravel(), bin_edges)-1
    Yte  = np.digitize(Ytest.ravel(), bin_edges)-1
    print (Ytr.max()+1, Yva.max()+1, Yte.max()+1)
    assert predict.shape[0]==K,'Expecting K categories'
    return Ytr, Yva, Yte, predict

def load_mmrf_quick(fold_span = range(5), suffix=''):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    fval     = os.path.join(dir_path, 'cleaned_mm_fold'+suffix+'.pkl')
    print( 'loading from:', fval)
    dset_rest = {}
    for foldnum in fold_span:
        fn = fval.replace('_fold',str(foldnum))
        with open(fn, 'rb') as f:
            dset = pickle.load(f)
        dset_rest[foldnum] = dset[foldnum]
    return dset_rest

def load_mmrf(fold_span = range(5), suffix='', digitize_K = 0, digitize_method = 'uniform', subsample = False):
    new_dset = load_mmrf_quick(fold_span = fold_span, suffix=suffix)
    
    # Make sure we only see data up to maxT
    for fold in fold_span:
        for tvt in ['train', 'valid', 'test']:
            M      = new_dset[fold][tvt]['m']
            m_t    = ((np.flip(np.cumsum(np.flip(M.sum(-1), (1,)), 1), (1,))>1.)*1)
            maxT   = m_t.sum(-1).max()
            new_dset[fold][tvt]['x'] = new_dset[fold][tvt]['x'][:,:maxT,:]
            new_dset[fold][tvt]['a'] = new_dset[fold][tvt]['a'][:,:maxT,:]
            new_dset[fold][tvt]['m'] = new_dset[fold][tvt]['m'][:,:maxT,:]
    
    if subsample: 
        # Transfer data from train to test set
        dir_path = os.path.dirname(os.path.realpath(__file__))
        fval     = os.path.join(dir_path, 'samples.pkl')
        if os.path.exists(fval):
            with open(fval,'rb') as f:
                sample_idx = pickle.load(f)
        else:
            np.random.seed(0)
            sample_idx = {}
            for fold in fold_span:
                N   = new_dset[fold]['train']['x'].shape[0]
                Ns  = int(N*0.15)
                idxshuf = np.random.permutation(N)
                sample_idx[fold] = (idxshuf[:Ns], idxshuf[Ns:])
            with open(fval,'wb') as f:
                pickle.dump(sample_idx, f)
        for fold in fold_span:
            keep, move = sample_idx[fold]
            for kk in ['a','x','m','ys_seq','ce','b','pids','m_a']:
                new_dset[fold]['test'][kk] = np.concatenate([new_dset[fold]['test'][kk], new_dset[fold]['train'][kk][move]], axis=0)
                new_dset[fold]['train'][kk]= new_dset[fold]['train'][kk][keep]
                
    if digitize_K>0:
        for fold in fold_span:
            Ytrain, Yvalid, Ytest = new_dset[fold]['train']['ys_seq'][:,0].ravel(), new_dset[fold]['valid']['ys_seq'][:,0].ravel(), new_dset[fold]['test']['ys_seq'][:,0].ravel()
            Ymax  = np.max([Ytrain.max(), Yvalid.max(), Ytest.max()])+0.1
            print ('Digitizing outcomes ymax:',Ymax)
            ytrain_bin, yvalid_bin, ytest_bin, predictions = digitize_outcomes(Ytrain, Yvalid, Ytest, Ymax, digitize_K, method=digitize_method)
            new_dset[fold]['train']['digitized_y'] = make_censored_oh(ytrain_bin, new_dset[fold]['train']['ce'].ravel(), digitize_K)
            new_dset[fold]['valid']['digitized_y'] = make_censored_oh(yvalid_bin, new_dset[fold]['valid']['ce'].ravel(), digitize_K)
            new_dset[fold]['test']['digitized_y']  = make_censored_oh(ytest_bin,  new_dset[fold]['test']['ce'].ravel(), digitize_K)
            new_dset[fold]['train']['prediction'] = predictions
            new_dset[fold]['valid']['prediction'] = predictions
            new_dset[fold]['test']['prediction']  = predictions
    for fold in fold_span:
        for k in ['train','valid','test']:
            m    = (new_dset[fold][k]['m'].sum(-1)>0.)*1.
            mask = (m[:,::-1].cumsum(1)[:,::-1]>0)*1.
            lot  = new_dset[fold][k]['a'][...,-1]
            lot[:,0]    = 1.
            lot[lot==0] = np.nan
            df = pd.DataFrame(lot)
            df.fillna(method='ffill', axis=1, inplace=True)
            lot = df.values
            lot = lot*mask
            lot[lot>=3] = 3.
            lot_oh      = np.zeros(lot.shape+(4,))
            for i in range(lot.shape[0]):
                for j in range(lot.shape[1]):
                    lot_oh[i,j,lot[i,j].astype(int)] = 1
            lot_oh      = lot_oh[...,1:]
            time_val    = np.ones_like(lot_oh[:,:,[-1]])
            time_val    = np.cumsum(time_val, 1)*0.1
            time_val    = (lot_oh.cumsum(1)*lot_oh*0.1).sum(-1,keepdims=True)
            new_dset[fold][k]['a'] = np.concatenate([time_val, new_dset[fold][k]['a'][...,:-1], lot_oh], -1)
            new_dset[fold][k]['feature_names_a'] = np.array(['time']+new_dset[fold][k]['feature_names_a'].tolist()[:-1]+['line1','line2','line3plus'])
    return new_dset

def get_te_matrix(): 
    ''' 
        5x16 matrix that contains direction of treatment effect on subset of lab features. 
        - 'Bor': PMN: -1, alb: 0, BUN: +1, Ca: -1, Crt: +1, Glc: 0, Hb: -1, Kappa: -1, 
        M-prot: -1, Plt: -1, TotProt: -1, WBC: -1, IgA: -1, IgG: -1, IgM: -1, Lambda: -1 
        (http://chemocare.com/chemotherapy/drug-info/bortezomib.aspx, 
        https://www.ncbi.nlm.nih.gov/pubmed/20061695 [renal], 
        https://www.nature.com/articles/s41598-017-13486-x [renal], 
        https://clinicaltrials.gov/ct2/show/NCT00972959 [calcium])
        
        - 'Car': PMN: -1, alb: 0, BUN: +1, Ca: -1, Crt: +1, Glc: +1, Hb:-1, Kappa: -1, 
        M-prot: -1, Plt: -1, TotProt: -1, WBC: -1, IgA: -1, IgG: -1, IgM: -1, Lambda: -1 
        (https://www.rxlist.com/kyprolis-side-effects-drug-center.htm [side effects])
        
        - 'Cyc': PMN: -1, alb: 0, BUN: +1, Ca: -1, Crt: +1, Glc: 0, Hb: -1, Kappa: -1, 
        M-prot: -1, Plt: -1, TotProt: -1, WBC: -1, IgA: -1, IgG: -1, IgM: -1, Lambda: -1 
        [https://www.rxlist.com/cytoxan-side-effects-drug-center.htm]
        
        - 'Dex': PMN: -1, alb: 0, BUN: 0, Ca: -1, Crt: 0, Glc: +1, Hb: +1, Kappa: 0, 
        M-prot: 0, Plt: +1, TotProt: 0, WBC: -1, IgA: 0, IgG: 0, IgM: 0, Lambda: 0 
        [https://dm5migu4zj3pb.cloudfront.net/manuscripts/108000/108231/JCI75108231.pdf], 
        
        - 'Len': PMN: -1, alb: 0, BUN: +1, Ca: -1, Crt: +1, Glc: -1, Hb: -1, Kappa: -1, 
        M-prot: -1, Plt: -1, TotProt: -1, WBC: -1, IgA: -1, IgG: -1, IgM: -1, Lambda: -1 
        [https://www.revlimid.com/mm-patient/about-revlimid/what-are-the-possible-side-effects/#common, 
        https://themmrf.org/multiple-myeloma/treatment-options/standard-treatments/revlimid/, 
        https://www.webmd.com/drugs/2/drug-94831/revlimid-oral/details/list-sideeffects]
        
        order of columns: array(['cbc_abs_neut', 'chem_albumin', 'chem_bun', 'chem_calcium',
           'chem_creatinine', 'chem_glucose', 'cbc_hemoglobin', 'serum_kappa',
           'serum_m_protein', 'cbc_platelet', 'chem_totprot', 'cbc_wbc',
           'serum_iga', 'serum_igg', 'serum_igm', 'serum_lambda'],
           dtype='<U15')
        order of rows: 'Bor', 'Car', 'Cyc', 'Dex', 'Len'
    '''
    te_matrix = np.array([[-1, 0, 1, -1, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], 
                          [-1, 0, 1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], 
                          [-1, 0, 1, -1, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], 
                          [-1, 0, 0, -1, 0, 1, 1, 0, 0, 1, 0, -1, 0, 0, 0, 0], 
                          [-1, 0, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])
    
    return te_matrix

if __name__=='__main__':
    print ('loading dataset quickly')
    dset = load_mmrf(fold_span=range(5), digitize_K = 20, digitize_method = 'uniform', suffix='_2mos')
    for k in dset[1]['train']:
        print (k, dset[1]['train'][k].shape)
    import ipdb;ipdb.set_trace()
