### This is a README dedicated to the paper Achieving Limited Adaptivity for MNL Bandits.

To run the algorithms, use the following command: 

`python3 main.py --alg_name [ALG_NAME] --num_contexts [NUM_CONTEXTS] --num_outcomes [NUM_OUTCOMES] --arm_seed [ARM_SEED] --theta_seed [THETA_SEED] --reward_seed [REWARD_SEED] --theta_star [THETA_STAR]  --reward_vec [REWARD_VEC] --horizon [HORIZON] --failure_level [FAILURE_LEVEL] --dim_arms [DIM_ARMS] --num_arms [NUM_ARMS] --param_norm_ub [PARAM_NORM_UB] --reward_norm_ub [REWARD_NORM_UB]`

You can follow the instructions in `Reproducible_Experiments.ipynb` to replicate the experiments shown in the paper.

We implement the following algorithms:

- `RS-MNL`: Our Algorithm
- `OFUL-MLogB`: [Zhang & Sugiyama (2023)](https://proceedings.neurips.cc/paper_files/paper/2023/file/5ef04392708bb2340cb9b7da41225660-Paper-Conference.pdf)
- `ada-OFU-ECOLog`: [Faury et al. (2022)](https://proceedings.mlr.press/v151/faury22a/faury22a.pdf)
- `OFULog+`: [Lee et al. (2023)](https://arxiv.org/abs/2310.18554)
- `RS-GLinCB`: [Sawarni et al. (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/0faa0019b0a8fcab8e6476bc43078e2e-Paper-Conference.pdf)
