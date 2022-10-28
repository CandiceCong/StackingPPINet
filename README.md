# StackingPPINet
Protein-protein interaction site prediction by self-attention integrated convolutional neural networks with multi-view features

# Requirements

PyTorch==0.4.0

numpy==1.15.0

scikit-learn==0.19.1

# Usage

  In this GitHub project, we give a demo to show how it works. The three benchmark datasets are given, i.e., Dset_186, Dset_72 and PDBset_164. Dset_186 consists of 186 protein sequences with the resolution less than 3.0 Å with sequence homology less than 25%. Dset_72 and PDBset_164 were constructed as the same as Dset_186. Dset_72 has 72 protein sequences and PDBset_164 consists of 164 protein sequences. These protein sequences in the three benchmark datasets have been annotated. Thus, we have 422 different annotated protein sequences. 
  The features data, such as PSSMs, raw sequences, secondary structures, etc., are given in data_cache folder. You can split the raw three datasets by yourself. In our study, we use the 60% as training dataset , 20% as validation dataset, and 20% as testing dataset. The detail of the  dataset division can see the paper.
  
  You can run the train.py file to train StackingPPINet and use the predcit.py file to see the predictive resluts. If you want to tune some hyper-parameters, you can change some values of hyper-parameters in config.py in utils folder.

  The other details can see the paper and the codes.

# License
See the LICENSE.txt file for details
