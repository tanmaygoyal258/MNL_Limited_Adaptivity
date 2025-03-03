import numpy as np
import os
import json
import argparse
from MNLEnv import MNLEnv
from MNLEnv_Batched import MNLEnv_Batched


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg_name', type = str)
    parser.add_argument("--num_contexts" , type = int , default = None , help = "number of contexts: 1 for non-contextual setting and None for infinite context setting")
    parser.add_argument("--num_outcomes" , type = int , default = 2)
    parser.add_argument('--seed', type = int, default = 123, help = 'random seed')
    parser.add_argument('--theta_star', type = str, default = "random", help = 'file containing optimal parameter')
    parser.add_argument('--normalize_theta_star', action = "store_true")
    parser.add_argument('--reward_vec', type = str, default = "random", help = 'file containing optimal parameter')
    parser.add_argument('--horizon', type = int, default = '10000', help = 'time horizon')
    parser.add_argument('--failure_level', type = float, default = 0.05, help = 'delta')
    parser.add_argument('--dim_arms', type = int, default = 5, help = 'arm dimensions')
    parser.add_argument('--num_arms', type = int, default = 4, help = 'number of items per slot')
    return parser.parse_args()

def generate_slot_arms(params):
    num_contexts = params["horizon"] if params["num_contexts"] is None else params["num_contexts"]
    all_arms = []
    for c in range(num_contexts):
        context_arms = []
        for _ in range(args.num_arms):
            context_arms.append([np.random.random()*2-1 for i in range(args.dim_arms)])
            context_arms = [(arm/np.linalg.norm(arm)).reshape(-1,1) for arm in context_arms]
        all_arms.append(context_arms)
    return all_arms

    
def generate_theta_star(args):
    # generate theta_star
    if args.theta_star != "random" and "npy" in args.theta_star:
        theta_star = np.load(args.theta_star)
        assert len(theta_star) == args.dim_arms * args.num_outcomes
    elif args.theta_star == "random":
        theta_star = np.array([np.random.random()*2-1 for i in range(args.dim_arms * args.num_outcomes)])
        if args.normalize_theta_star:
            theta_star /= np.linalg.norm(theta_star)
            # theta_star *= 3
    return theta_star

def generate_reward_vec(args):
    return [np.random.random() for _ in range(args.num_outcomes)]

if __name__ ==  "__main__":


    # read the arguments
    args = parse_args()

    # set the seed before any randomization occurs
    np.random.seed(args.seed)
    
    # create the params dictionary
    params = {}
    params["alg_name"] = args.alg_name
    params["num_contexts"] = args.horizon if args.num_contexts is None else args.num_contexts
    params["horizon"] = args.horizon
    params["failure_level"] = args.failure_level
    params["dim_arms"] = args.dim_arms
    params["num_arms"] = args.num_arms
    params["seed"] = args.seed
    params["num_outcomes"] = args.num_outcomes

    theta_star = generate_theta_star(args)
    params["thetastar"] = theta_star.tolist()
    params["param_norm_ub"] = int(np.linalg.norm(theta_star)) + 1
    
    reward_vec = generate_reward_vec(args)
    params["reward_vec"] = reward_vec
    params["reward_vec_norm_ub"] = int(np.linalg.norm(reward_vec))+1

    print(params)

    # generate the arms for each slot
    slot_arms = generate_slot_arms(params)
    
    # check validity of the data path
    data_path = f"Results"
    if not os.path.exists(data_path):
            os.makedirs(data_path)
    data_path_with_alg = f"{data_path}/{args.alg_name}" if args.alg_name in ["rs_mnl" , "mlogb"] else data_path
    if not os.path.exists(data_path_with_alg):
        os.makedirs(data_path_with_alg)
    suffix = f"contexts={args.num_contexts}_N={args.num_arms}_K={args.num_outcomes}_T={args.horizon}_seed={args.seed}" if args.alg_name in ["rs_mnl" , "mlogb"] else \
        f"{args.alg_name}_contexts={args.num_contexts}_N={args.num_arms}_K={args.num_outcomes}_T={args.horizon}_seed={args.seed}"
    data_path_with_details = f"{data_path_with_alg}/{suffix}"
    if not os.path.exists(data_path_with_details):
        os.makedirs(data_path_with_details)
    params["data_path"] = data_path_with_details

    # dump the json file with the params
    with open(data_path_with_details + "/params.json", "w") as outfile:
        json.dump(params, outfile)

    # set the environment
    if args.alg_name in ["rs_glincb" , "rs_mnl" , "mlogb" , "ada_ofu_ecolog" , "ofulogplus"]:
        env = MNLEnv(params , slot_arms , theta_star, reward_vec)
    # else set the batched algorithm: TODO
    else:
        env = MNLEnv_Batched(params , slot_arms , theta_star, reward_vec)

    # obtain the regret, reward, and time arrays, and save them
    regret_arr = env.regret_arr
    np.save(data_path_with_details + "/regret.npy", regret_arr)
    time_arr = env.time_arr
    try:
        switches_arr = env.switch_arr
        np.save(data_path_with_details + "/switches.npy", switches_arr)
    except:
        pass
    try:
        np.save(data_path_with_details + "/time.npy" , time_arr)
    except:
        # different sizes batches have different arrays
        new_time_arr = []
        batch_lengths = []
        for batch in time_arr:
            batch_lengths.append(len(batch))
            new_time_arr += batch
        np.save(data_path_with_details + "/time.npy" , new_time_arr)
        np.save(data_path_with_details + "/batch_lengths.npy" , batch_lengths)
        
    try:
        reward_arr = env.reward_arr
        np.save(data_path_with_details + "/reward.npy", reward_arr)
        pull_time_arr = env.pull_time_arr
        update_time_arr = env.update_time_arr
        np.save(data_path_with_details + "/pull_time.npy" , pull_time_arr)
        np.save(data_path_with_details + "/update_time.npy" , update_time_arr)
    except: 
        # the time and reward arrays were not instantiated for the algorithms
        pass
