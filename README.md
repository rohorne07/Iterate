# Iterate

## Cyclic improvement of molecular inhibitors of alpha-synuclein

The repository contains the code (Python 3.9.7) used to carry out the iterative molecule improvement as applied to docking scores and then to the aggregation data using an easy-to-install version of the Junction Tree Variational Autoencoder https://arxiv.org/abs/1802.04364 to generate molecular features for molecules based on JTNN latent vectors. The original, full repository can be found here: https://github.com/wengong-jin/icml18-jtnn

The simulation of the experimental process carried out can be be run via bin/iteration_simulation.py

Note: Depending on OS, Python 3.6.13 may be required for jtnnencoder functioning
