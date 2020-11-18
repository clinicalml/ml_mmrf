conda create -n data_v1 python=3.7.3
conda activate data_v1
conda install matplotlib scipy numpy pandas h5py ipython jupyterlab cython seaborn scikit-learn r
conda install -c conda-forge tslearn ipdb
pip install pyro-ppl tensorflow_probability jupyter-tensorboard lifelines scanpy
conda install -c brittainhard fancyimpute
conda install -c conda-forge ncurses
conda install pytorch cudatoolkit=10.1 torchvision -c pytorch 
