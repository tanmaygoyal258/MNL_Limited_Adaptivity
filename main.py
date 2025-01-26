import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse
from datetime import datetime
from MNLEnv import MNLEnv
from LinearEnv import LinearEnv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg_name' , type = str , default = "B_MNL" , help = "algorithm to run, choose from [B_MNL , BatchLinUCB]")
    parser.add_argument('--num_arms' , type = int , default = 10 , help = "number of arms")
    parser.add_argument('--dim_arms' , type = int , default = 5 , help = "dimensions of the arm")
    parser.add_argument('--num_outcomes' , type = int , default = 2 , help = "number of outcomes")
    parser.add_argument('--optimal_design_alg' , type = str , default = "barycentric_spanner" , help = "algorithm to use for optimal design")
    parser.add_argument('--seed', type = int, default = 123, help = 'random seed')
    parser.add_argument('--theta_star', type = str, default = "random", help = 'file containing optimal parameter')
    parser.add_argument('--reward_vec', type = str, default = "random", help = 'file containing reward_vec')
    parser.add_argument('--normalize_thetastar', action = 'store_true')
    parser.add_argument('--horizon', type = int, default = '10000', help = 'time horizon')
    parser.add_argument('--num_batches' , type = int , help = "number of batches")
    parser.add_argument('--failure_level', type = float, default = 0.05, help = 'delta')
    parser.add_argument('--barycentric_spanner_constant' , type = float , default = 2 , help = 'constant for the barycentric spanner')
    return parser.parse_args()


    
if __name__ ==  "__main__":

    # read the arguments
    args = parse_args()

    # check alg_name
    assert args.alg_name in ["B_MNL" , "BatchLinUCB"]
    if args.alg_name == "BatchLinUCB":
        args.num_outcomes = 1

    # set the seed before any randomization occurs
    np.random.seed(args.seed)
    
    # create the params dictionary
    params = {}
    params["alg_name"] = args.alg_name
    params["num_arms"] = args.num_arms
    params["dim_arms"] = args.dim_arms
    params["optimal_design_alg"] = args.optimal_design_alg
    params["seed"] = args.seed
    params["horizon"] = args.horizon
    params["failure_level"] = args.failure_level
    params["BS_constant"] = args.barycentric_spanner_constant
    params["num_batches"] = args.num_batches
    params["num_outcomes"] = args.num_outcomes

    # generate theta_star
    if args.theta_star != "random" and "npy" in args.theta_star:
        theta_star = np.load(args.theta_star)
        assert len(theta_star) == args.dim_arms * params["num_outcomes"]
    elif args.theta_star == "random":
        theta_star = np.array([np.random.random()*2-1 for i in range(args.dim_arms * params["num_outcomes"])])
        if args.normalize_thetastar:
            theta_star /= np.linalg.norm(theta_star)
        params["thetastar"] = theta_star.tolist()
        params["param_norm_ub"] = int(np.linalg.norm(theta_star)) + 1

    # generate reward_vec : ensure first element is zero to correspond to no action chosen
    if args.alg_name == "BatchLinUCB":
        reward_vec = None
    else:
        if args.reward_vec != "random" and "npy" in args.reward_vec:
            reward_vec = np.load(args.reward_vec)
            assert len(reward_vec) == args.num_outcomes , "Reward vec should have either num_outcomes elements"
        elif args.reward_vec == "random":
            reward_vec = np.array([np.random.random() for i in range(args.num_outcomes)])
        else:
            assert False , "Incorrect choice for Reward Vector"
        params["reward_vec"] = reward_vec.tolist()
        params["reward_vec_norm_ub"] = int(np.linalg.norm(reward_vec)) + 1

    print(params)
    
    # generate the arms for the instance
    all_arms = []
    for t in range(args.horizon):
        arms = [[np.random.random()*2-1 for i in range(args.dim_arms)] for j in range(args.num_arms)]
        arms = [arm/np.linalg.norm(arm) for arm in arms]
        all_arms.append(arms)

    # check validity of the data path
    data_path = f"Results_{args.alg_name}"
    if not os.path.exists(data_path):
            os.makedirs(data_path)
    data_path_log = data_path + "/logs"
    if not os.path.exists(data_path_log):
        os.makedirs(data_path_log)
    data_path_with_details = f"{data_path_log}/T={args.horizon}_K={args.num_outcomes}_d={args.dim_arms}_N={args.num_arms}_seed={args.seed}"
    if not os.path.exists(data_path_with_details):
        os.makedirs(data_path_with_details)
    params["data_path"] = data_path_with_details

    # dump the json
    with open(data_path_with_details + "/params.json", "w") as outfile:
        json.dump(params, outfile)

    # initialize the environment
    if args.alg_name == "B_MNL":
        env = MNLEnv(params , all_arms , theta_star , reward_vec)
    else:
        env = LinearEnv(params , all_arms, theta_star)
    
    # obtain the regret array and save it
    regret_arr = env.regret_arr
    np.save(data_path_with_details + "/regret.npy", regret_arr)

