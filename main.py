import numpy as np
import os
import json
import argparse
from MNLEnv import MNLEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg_name', type = str)
    parser.add_argument("--num_contexts" , type = int , default = None , help = "number of contexts: 1 for non-contextual setting and None for infinite context setting")
    parser.add_argument("--num_outcomes" , type = int , default = 1)
    parser.add_argument('--arm_seed' , type = int ,default = 456)
    parser.add_argument('--theta_seed' , type = int , default = 0)
    parser.add_argument('--reward_seed' , type = int , default = 123)
    parser.add_argument('--theta_star', type = str, default = "random", help = 'file containing optimal parameter')
    parser.add_argument('--reward_vec', type = str, default = "random", help = 'file containing reward vector parameter')
    parser.add_argument('--horizon', type = int, default = '10000', help = 'time horizon')
    parser.add_argument('--failure_level', type = float, default = 0.05, help = 'delta')
    parser.add_argument('--dim_arms', type = int, default = 3, help = 'dimension of arm')
    parser.add_argument('--num_arms', type = int, default = 10, help = 'number of items per slot')
    parser.add_argument('--param_norm_ub' , type = int , default = 5 , help = "upper_bound for norm of theta")
    parser.add_argument('--reward_norm_ub' , type = int , default = 2 , help = "upper bound on norm of reward_vector")
    return parser.parse_args()

if __name__ ==  "__main__":

    # read the arguments
    args = parse_args()

    # create the params dictionary
    params = {}
    params["alg_name"] = args.alg_name
    params["num_contexts"] = args.horizon if args.num_contexts is None else args.num_contexts
    params["horizon"] = args.horizon
    params["failure_level"] = args.failure_level
    params["dim_arms"] = args.dim_arms
    params["num_arms"] = args.num_arms
    params["arm_seed"] = args.arm_seed
    params["theta_seed"] = args.theta_seed
    params["reward_seed"] = args.reward_seed
    params["num_outcomes"] = args.num_outcomes
    params["param_norm_ub"] = args.param_norm_ub
    params["reward_norm_ub"] = args.reward_norm_ub

    # generate theta_star
    if args.theta_star != "random" and "npy" in args.theta_star:
        params["theta_star"] = np.load(params["theta_star"])
        assert len(params["theta_star"]) == params["dim_arms"] * params["num_outcomes"]
    else:
        theta_rng = np.random.default_rng(params["theta_seed"])
        params["theta_star"] = np.array([theta_rng.uniform()*2 - 1 for i in range(params["dim_arms"] * params["num_outcomes"])])
        params["theta_star"] = params["theta_star"] / np.linalg.norm(params["theta_star"]) * params["param_norm_ub"]
        params["theta_star"] = params["theta_star"].tolist()

    # generate the reward vector
    if args.num_outcomes == 1:
        params["reward_vec"] = np.array([1])
        params["reward_vec"] = params["reward_vec"].tolist()
        params["reward_norm_ub"] = 1
    else:
        reward_rng = np.random.default_rng(params["reward_seed"])
        params["reward_vec"] = [reward_rng.uniform() for _ in range(args.num_outcomes)]
        params["reward_vec"] = params["reward_vec"] / np.linalg.norm(params["reward_vec"]) * params["reward_norm_ub"]
        params["reward_vec"] = params["reward_vec"].tolist()

    assert params["alg_name"] in ["rs_glincb" , "rs_mnl" , "mlogb" , "ada_ofu_ecolog" , "ofulogplus"] , "Incorrect algorithm name"

    # set the environment
    if args.alg_name in ["rs_glincb" , "rs_mnl" , "mlogb" , "ada_ofu_ecolog" , "ofulogplus"]:
        env = MNLEnv(params)

    # check validity of the data path
    data_path = f"Results/v2"
    if not os.path.exists(data_path):
            os.makedirs(data_path)
    data_path_with_alg = f"{data_path}/{args.alg_name.lower()}"
    if not os.path.exists(data_path_with_alg):
        os.makedirs(data_path_with_alg)
    suffix = f"N={args.num_arms}_K={args.num_outcomes}_T={args.horizon}_reward_seed={args.reward_seed}"
    data_path_with_details = f"{data_path_with_alg}/{suffix}"
    if not os.path.exists(data_path_with_details):
        os.makedirs(data_path_with_details)
    params["data_path"] = data_path_with_details

    # dump the json file with the params
    with open(data_path_with_details + "/params.json", "w") as outfile:
        json.dump(params, outfile)
    
    # obtain the regret and number of switches
    regret_arr = env.regret_arr
    np.save(data_path_with_details + "/regret.npy", regret_arr)
    try:
        switches_arr = env.switch_arr
        np.save(data_path_with_details + "/switches.npy", switches_arr)
    except:
        pass

    print(f"Algorithm {args.alg_name} with reward_seed {args.reward_seed} : Regret {np.sum(regret_arr)}")
    
