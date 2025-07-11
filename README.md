### This is a README dedicated to the paper Achieving Limited Adaptivity for MNL Bandits.

To run the algorithms, use the following command: 

`python3 main.py --alg_name [ALG_NAME] --num_contexts [NUM_CONTEXTS] --num_outcomes [NUM_OUTCOMES] --arm_seed [ARM_SEED] --theta_seed [THETA_SEED] --reward_seed [REWARD_SEED] --theta_star [THETA_STAR]  --reward_vec [REWARD_VEC] --horizon [HORIZON] --failure_level [FAILURE_LEVEL] --dim_arms [DIM_ARMS] --num_arms [NUM_ARMS] --param_norm_ub [PARAM_NORM_UB] --reward_norm_ub [REWARD_NORM_UB]`

where the arguments are as follows:

1. ALG_NAME: One of ["rs_glincb" , "rs_mnl" , "mlogb" , "ada_ofu_ecolog" , "ofulogplus"]

2. NUM_CONTEXTS: 1 for the non-contextual setting and None for the contextual setting

3. NUM_OUTCOMES: the number of outcomes for the multinomial logistic function

4. ARM_SEED: Seed to control the generation of arms

5. THETA_SEED: Seed to control the generation of theta_star (if random, check point 7)

6. REWARD_SEED: Seed to control the generation of the reward vector (if random, check point 8) and the rewards

7. THETA_STAR: Random or path to the .npy file containing theta_star, the hidden optimal parameter

8. REWARD_VEC: Random or path to the .npy file containing the reward_vec

9. HORIZON: The number of rounds for which the algorithm is run

10. FAILURE_LEVEL: The probability with which the confidence bounds fail, Default : 0.05

11. DIM_ARMS: the dimension of the arm set

12. num_arms: the number of arms in the arm set

13. PARAM_NORM_UB: the desired upper bound for the parameter vectors (in case randomly generated, check point 7)

14. REWARD_NORM_UB: the desired upper bound for the reward vector (in case randomly generated, check point 8)

You can follow the instructions in `Reproducible_Experiments.ipynb` to replicate the experiments shown in the paper.