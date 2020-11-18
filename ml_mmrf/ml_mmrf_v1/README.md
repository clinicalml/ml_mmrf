## ML-MMRF version 1

This is the first version of the ML MMRF dataset. 


### Instructions
* Run the commands in `requirements.sh` to create a conda environment capable of running the setup code
* Download the `IA13` dataset from the [MMRF Researcher Gateway](https://research.themmrf.org/) and place the folders named into `ia13`
* Run `1_MMRF_files.ipynb` to create the various tensors
* Run `2_CleanData.ipynb` to preprocess the tensors
* Run `3_SanityCheckData.ipynb` for some sanity checks
* Look at the main function in `data.py` for an example on how to load the data. Running `python data.py` should yield the following output:

```
$ python data.py
....
x (439, 32, 16)
m (439, 32, 16)
feature_names_x (16,)
ys_seq (439, 1)
ce (439, 1)
b (439, 16)
feature_names (16,)
a (439, 32, 9)
m_a (439, 33, 6)
feature_names_a (9,)
digitized_y (439, 20)
prediction (20,)
```

### Notes
* `ia13_pca_embeddings.csv` was created using PCA on the RNA-Seq data from MMRF.


### Reference
If you use this version of the dataset, please cite the following paper where it was introduced:

