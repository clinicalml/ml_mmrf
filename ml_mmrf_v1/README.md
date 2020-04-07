## ML-MMRF version 1

This is the first version of the ML MMRF dataset. 



### Instructions
* Run the commands in `requirements.sh` to create a conda environment capable of running the setup code
* Download the `IA13` dataset from the [MMRF Researcher Gateway](https://research.themmrf.org/) and place the folders named into `ia13`
* Run `1_MMRF_files.ipynb` to create the various tensors
* Run `2_CleanData.ipynb` to preprocess the tensors
* Run `3_SanityCheckData.ipynb` for some sanity checks

### Notes
* `ia13_pca_embeddings.csv` was created using PCA on the RNA-Seq data from MMRF.