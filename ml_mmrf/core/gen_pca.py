import anndata
import os, sys, glob
import pickle
import pandas as pd
import numpy as np
import scanpy as sc
import scipy
import warnings

def gen_pca_embeddings(ia_version='ia15', \
                       train_test_file=None, \
                       write_to_csv=False, \
                       write_csv_fname='PCA.csv', \
                       FDIR_sh   = '/afs/csail.mit.edu/group/clinicalml/datasets/multiple_myeloma/ia15'):

    qc_path   = os.path.join(FDIR_sh,f'MMRF_CoMMpass_{ia_version.upper()}_Seq_QC_Summary.csv')
    fpkm_path = os.path.join(FDIR_sh,f'MMRF_CoMMpass_{ia_version.upper()}a_E74GTF_Cufflinks_Gene_FPKM.txt')
    genref_path = os.path.join(FDIR_sh,f'GRCh37.p13_E92_mart_export.txt')

    # load in files
    data_files = {}
    fpkm_df = pd.read_table(fpkm_path, encoding='latin-1')
    design_df = pd.read_csv(qc_path, delimiter=',', encoding='latin-1')
    design_df['sampleID'] = design_df['QC Link SampleName'].apply(lambda x: x.split('_')).apply(lambda x: '_'.join(x[:4]))
    design_df['cellType'] = design_df['QC Link SampleName'].apply(lambda x: x.split('_')).apply(lambda x: '_'.join(x[3:5]))

    # STEP 1: filter IDs in FPKM based on QC data and filter genes based on annotation file (GRCh37)

    # get sampleID's which meet criteria (baseline, bone marrow derived)
    design_df = design_df[design_df.sampleID.isin(fpkm_df.columns)] #only interested in filtering those samples that are in the fpkm matrix
    design_df = design_df[design_df.cellType=='BM_CD138pos']
    design_df = design_df[design_df['Visits::Reason_For_Collection'] == "Baseline"]
    design_df = design_df[design_df.MMRF_Release_Status == "RNA-Yes"]
    design_df.rename(columns={"Patients::KBase_Patient_ID":"PUBLIC_ID"}, inplace=True)

    # drop gene location column and move gene id to index
    fpkm_df.drop(labels="Location", inplace=True, axis=1) # drop gene location column
    fpkm_df.set_index('GENE_ID', inplace=True)
    fpkm_df = fpkm_df.loc[:,design_df.sampleID]

    # sum rows of FPKM with the same gene id
    fpkm_df = fpkm_df.groupby('GENE_ID').sum()

    #if necessary, would filter on % mito here. Not nec here.

    # load gene annotations, will use to update index to gene names
    gtf = pd.read_table(genref_path)

    shared_genes = np.intersect1d(fpkm_df.index, gtf['Gene stable ID']) # make sure gtf and fkpm have same genes
    fpkm_df = fpkm_df.loc[shared_genes,:] #set order
    fpkm_df.index =  gtf.set_index('Gene stable ID').loc[shared_genes,'Gene name'] #update rownames to gene names

    # sum rows of FPKM with the same gene name
    fpkm_df = fpkm_df.groupby('Gene name').sum()

    # STEP 2: reproduce Seurat PCA functionality in scanpy (https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html)
    if train_test_file is None: 
        idxs = [x for x in fpkm_df.columns.values]
        np.random.shuffle(idxs)
        pivot= int(0.7*len(idxs))
        trainidx = idxs[:pivot]; testidx = idxs[pivot:]
    else: 
        with open(train_test_file,'rb') as f:
            _, _, trainidx, testidx = pickle.load(f)
        trainidx = [x + '_1_BM' for x in trainidx]
        testidx  = [x + '_1_BM' for x in testidx]
        remaining_idxs = []
        for x in fpkm_df.columns.values: 
            if x not in trainidx and x not in testidx: 
                remaining_idxs.append(x)
        trainidx += remaining_idxs
    trainfpkm_df  = fpkm_df.loc[:,fpkm_df.columns.isin(trainidx)]
    traindesign_df= design_df[design_df['sampleID'].isin(trainidx)]
    testfpkm_df   = fpkm_df.loc[:,fpkm_df.columns.isin(testidx)]
    testdesign_df = design_df[design_df['sampleID'].isin(testidx)]
    adata = anndata.AnnData(X = scipy.sparse.csr_matrix(trainfpkm_df.T), var=gtf.set_index('Gene name').loc[trainfpkm_df.index,['Gene description']].reset_index().drop_duplicates('Gene name').set_index('Gene name'), obs=traindesign_df.set_index('PUBLIC_ID')[['Batch', 'QC_Percent_Mitochondrial']])

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=1.7, max_mean=5, min_disp=0.5, flavor='seurat') #min is around 5 fpkm
    adata.X = scipy.sparse.csr_matrix(adata.X) #necessary b/c of scanpy bugginess
    adata.raw = adata
    sc.pp.scale(adata, max_value=10)
    adata.X[np.where(np.isnan(adata.X))] = 1e-9
    adata.var['std'].values[np.where(np.isnan(adata.var['std'].values))] = 1e-9
    adata.X = scipy.sparse.csr_matrix(adata.X) #necessary b/c of scanpy bugginess
    sc.tl.pca(adata, svd_solver='arpack', return_info=True)
    
    # project to test data 
    testfpkm_df   = testfpkm_df[testfpkm_df.index.isin(adata.var.index.values)]
    test_adata    = anndata.AnnData(X = scipy.sparse.csr_matrix(testfpkm_df.T), var=gtf.set_index('Gene name').loc[testfpkm_df.index,['Gene description']].reset_index().drop_duplicates('Gene name').set_index('Gene name'), obs=testdesign_df.set_index('PUBLIC_ID')[['Batch', 'QC_Percent_Mitochondrial']])
    sc.pp.filter_cells(test_adata, min_genes=0)
    sc.pp.filter_genes(test_adata, min_cells=0)
    sc.pp.log1p(test_adata)
    test_adata.X = scipy.sparse.csr_matrix((test_adata.X - adata.var['mean'].values[None,:]) / adata.var['std'].values[None,:])
    test_adata.raw = test_adata
    test_pca = (test_adata.X*adata.varm['PCs'])[:,:40]
    train_pca = adata.obsm['X_pca'][:,:40]
    pca_cat   = np.concatenate((train_pca,test_pca),0)
    
    # STEP 3: write embeddings to csv file and/or return a Pandas dataframe
    pca_df = pd.DataFrame(pca_cat, columns=['PC'+str(i) for i in np.arange(40)+1], index=np.concatenate((adata.obs.index.values, test_adata.obs.index.values))).reset_index()
    pca_df = pca_df.rename(columns = {'index':'PUBLIC_ID'})
    if write_to_csv:
        # write to csv
        pca_df.to_csv(write_csv_fname, index=False)

    # return dataframe with columns: 'PUBLIC_ID', 'PC1', 'PC2', ...
    return(pca_df)

if __name__ == '__main__':
    np.random.seed(0)
    gen_pca_embeddings() # use for debugging
