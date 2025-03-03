### This is a README dedicated to MNL_Limited_Adaptivity.

To run the algorithms, use the following command: 

`python3 main.py --alg_name [ALG_NAME] --num_contexts [NUM_CONTEXTS] --num_outcomes [NUM_OUTCOMES] --seed [SEED] --theta_star [THETA_STAR] --normalize_theta_star --reward_vec [REWARD_VEC] --horizon [HORIZON] --failure_level [FAILURE_LEVEL] --dim_arms [DIM_ARMS] --num_arms [NUM_ARMS]`

where the arguments are as follows:

1. ALG_NAME: One of ["rs_glincb" , "rs_mnl" , "mlogb" , "ada_ofu_ecolog" , "ofulogplus"]

2. NUM_CONTEXTS: 1 for the non-contextual setting and None for the contextual setting

3. NUM_OUTCOMES: the number of outcomes in the experiment

4. SEED: Random seed

5. THETA_STAR: Random or path to the .npy file containing theta_star, the hidden optimal parameter

6. NORMALIZE_THETA_STAR: if theta_star is to be converted into a unit vector

7. REWARD_VEC: Random or path to the .npy file containing the reward_vec, fixed and known

8. HORIZON: The number of rounds for which the algorithm is run

9. FAILURE_LEVEL: The probability with which the confidence bounds fail, Default : 0.05

10. DIM_ARMS: the dimension of the arm set

11. num_arms: the number of arms in the arm set