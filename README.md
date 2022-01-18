# Iterate

## Cyclic improvement of molecular inhibitors of alpha-synuclein

The repository contains the code (Python 3.9.7) used to carry out the iterative molecule improvement as applied to docking scores and then to the aggregation data using a Junction Tree Variational Autoencoder https://arxiv.org/abs/1802.04364 to generate molecular features for molecules based on JTNN latent vectors. The original, full repository can be found here: https://github.com/wengong-jin/icml18-jtnn

The data file contains:
  - ~8000 molecules of the test set under zinc.csv and their AutoDock Vina binding scores to asyn fibrils in kcal / mol
  - the initial molecules found via docking and similarity searches and their normalised thalves

The bin file contains some example functions for carrying out the methods used in the submitted paper including:
  - dimensionality_reduction.py creates PCA and UMAP representations of the molecular space (tSNE not included here as the package is deprecated and requires a custom environment)
  - cross_val.py produces a learning curve for the docking scores
  - iteration_simulation.py produces summary data files and box plots for a simulated version of the iterative cycle carried out on the docking scores
  - predict.py carries out a prediction on the aggregation data

Notes: 
- For use of the jtnnencoder, a separate environment using Python 3.6.13 is required, and the module versions specified in requirements_encoder.txt
- Values required for the CNS_MPO function can be acquired via JChem


