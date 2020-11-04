# Machine Learning with Multiple Myeloma Research Foundation (MMRF) CoMMpass Dataset

## Overview
The MMRF CoMMpass registry contains longitudinal data for over 1000 newly diagnosed multiple myeloma patients. Researchers track these patients from initial diagnosis through their course of treatment over a minimum of 5 years, capturing patient lab values, treatments, and complications. In addition to these data, there is a rich store of biological and demographic data of each patient at baseline, including RNA-seq, cytogenetics, and survey (e.g. Quality of Life) data. 

ML-MMRF is a repository built to process the MMRF CoMMpass Dataset and allows researchers to use these data for machine learning. We first parse the raw MMRF files into tensors (stored in numpy matrices), then clean and normalize the tensors, and finally provide a notebook for validation of the procedure. 

## Data Access 
Access to the MMRF CoMMpass data is through the [MMRF Researcher Gateway](https://research.themmrf.org/). You must first register using your institutional email to receive access.

## Methods 
At a high level, run ```python ml_mmrf_v2/core/build_mmrf_dataset.py``` to create data tensors from raw MMRF flatfiles. After data processing, we recommend going through the provided jupyter notebook, "3_SanityCheckData.ipynb" to verify that the tensors have been created properly. Below, we detail the specific aspects of the outer level build script. 

### Parsing of Sequential Tensors 
The Parser class in ```core/parser.py``` is responsible for taking the raw data stored in pandas dataframes and converting it into sequential data tensors of size ```N x maxT x D```, where N is the number of examples, maxT is the max number of time steps, and D is the feature dimension. The user specifies the granularity of time at which we parse, i.e. the number of days between subsequent time steps, and also specifies maxT, the max time for which we parse. Finally, we also return a binary mask tensor, where we store a value of 1 if it is observed and 0 if it is missing. As an example, if we specify granularity to be 60 and maxT to be 33, which are the default settings, then the treatment and labs tensors will be of size N x 33 x D. Furthermore, suppose ```t = 1,...,maxT```; the time between ```t``` and ```t+1``` is determined by granularity, which in this case is 60 days (2 months). 

There are three data types that we parse from the raw MMRF files: treatments, patient labs, and baseline data (basic demographics and cytogenetics). We detail the specific features available in the raw files in the *Data Description* section below. For now, we will give a brief overview of how we actually do the parsing for each data type. 

#### Patient Lab Values 
The subset of lab values that we select from the raw files at each visit include a patient's blood chemistry values (i.e. albumin, BUN, calcium, creatinine, glucose, and total protein), complete blood counts (i.e. absolute neutrophils, hemoglobin, WBC, and plateletes), and finally, their serum immunoglobuins (i.e. IgG, IgA, IgM, lambda, kappa, and M-protein). We then clip the values of these labs to 5 times the median value, and then build the sequential tensor by processing the values based on the user-specified granularity and maxT. Each visit and lab collection panel is labeled with the day in which it was taken, allowing us to bucket the lab value into a particular time step. The lab values at a patient's baseline visit are mean imputed across the cohort at baseline.

#### Patient Treatments
We obtain the treatments given to a patient across time along with the line of therapy that each treatment is associated with. Additionally, we restrict the treatments to those that appear in the top 10 treatment combinations with respect to raw counts over the entire time course. The final treatment representation that we use is a 9-dimensional binary vector where five dimensions refer to whether or not one of Dexamethasone, Lenalidomide, Bortezomib, Carfilzomib, and Cyclophosphamide are given. The sixth thru ninth dimensions of the binary vector are a one-hot representation of the line of therapy, which we categorize into one of three buckets: Line 1, Line 2, or >= Line 3. When parsing the treatments, we leverage the start and end days which are given for each of the regimens to construct the tensor, whose final size is N x maxT x 9. 

#### Patient Baseline Data 



### Selecting Clinical Outcome
outcomes (e.g. time to death, treatment response)

### Cleaning and Normalization of Data 

### Train/Test Split 

### Processing of Genetic Data 

### Validation and Sanity Checks



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
