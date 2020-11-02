# Machine Learning with Multiple Myeloma Research Foundation (MMRF) CoMMpass Dataset

## Overview

## Data Access 
Access to the MMRF CoMMpass data is through the [MMRF gateway](https://research.themmrf.org/). You must first register using your institutional email to receive acess.

## Methods 


## Data Description 


This repository is organized into <strong>data folders</strong>. Each such folder contains code to setup MMRF datasets from various versions of the raw MMRF files. The datasets from each data folder may be used in multiple different projects. 

Follow the instructions in the data folder to setup the data. You are free to use the data as is. To run machine learning models on the data, you will also need to setup the github repositories for the code. 


## Instructions 
* Enter the desired data folder
* Run the code in `requirements.sh` to setup the relevant packages you will need in order to setup the data
* Follow the instructions to download the relevant MMRF dataset files. You will need access to the [MMRF dataset](https://research.themmrf.org/) so please sign up for it.

### Version 1: `ml_mmrf_v1`
* The following research papers use this data:
  * `Inductive Biases for Unsupervised, Sequential models of Cancer Progression`
* MMRF Data Version: IA13
* Goal: Unsupervised learning of high-dimensional patient data

### Version 2: `ml_mmrf_v2`
* The following research papers use this data: 
  * `Attentive, Pharmacodynamic State Space Modeling` 
* MMRF Data Version: IA13, IA15 
* To create data tensors from raw MMRF flatfiles, run python ml_mmrf_v2/core/build_mmrf_dataset.py. You can also go through the jupyter notebook, "3_SanityCheckData.ipynb" to verify that the data has been created properly.
