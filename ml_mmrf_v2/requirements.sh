conda create -n data_v2 python=3.7.3
conda activate data_v2
conda install matplotlib scanpy scipy numpy pandas h5py ipython jupyterlab cython seaborn scikit-learn r
conda install -c conda-forge tslearn ipdb
pip install pyro-ppl tensorflow_probability jupyter-tensorboard lifelines scanpy[leiden]
conda install -c brittainhard fancyimpute
conda install -c conda-forge ncurses
conda install pytorch cudatoolkit=10.1 torchvision -c pytorch
