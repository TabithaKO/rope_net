# Robotic Manipulation of Deformable Objects using Point Cloud Observations

This repository contains the training and evaluation scripts for both the baseline and the grasp-conditioned models.

### Datasets from Isaac Sim:

- joint_positions_3000.npy
- rope_positions_3000.npy
- grasp_point_3000.npy

### Training Scripts:

- train_baseline_reg.py
- train_conditioned_reg.py

### Evaluation Scripts:

- evaluate_baseline.py
- evaluate_conditional_model.py

### Pretrained Scripts:

- checkpoints_baseline.zip
- checkpoints_conditional_model.zip

## Rope IsaacSim Environment:

- Place files in IsaacSim_env inside root directory of an IsaacSim 4.5.0 see: https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/download.html to download and: https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/install_workstation.html for installation

- Use sample_ropes.py to generate data

- Use test_model.py to test model predictions