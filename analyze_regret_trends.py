import numpy as np
import os
import matplotlib.pyplot as plt
import argparse



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg_name' , type = str , default = "B_MNL" , help = "algorithm to run, choose from [B_MNL , BatchLinUCB]")
    parser.add_argument('--horizon', type = int, default = '10000', help = 'time horizon')
    parser.add_argument('--num_contexts' , type = int , default = None)
    parser.add_argument('--num_outcomes', type = int, default = 5, help = 'number of outcomes')
    parser.add_argument('--num_arms', type = int, default = 4, help = 'number of arms')
    parser.add_argument('--dim_arms' , type = int , help = "dimension of the arms")
    parser.add_argument('--seed', type = int , default = 123)
    parser.add_argument("--xlim" , type = float , nargs = 2 , default = [None , None])
    parser.add_argument("--ylim" , type = float , nargs = 2 , default = [None , None])
    return parser.parse_args()

if __name__ == "__main__":
    
    # read the arguments
    args = parse_args()
    if args.alg_name == "BatchLinUCB":
        args.num_outcomes = 1

    # check for all the files that exist and store corresponding regret and time arrays
    regret_arrays = []
    folder_name = f"Results_{args.alg_name}/logs"
    suffix = f"/T={args.horizon}_K={args.num_outcomes}_d={args.dim_arms}_N={args.num_arms}_seed={args.seed}_contexts={args.num_contexts}"
    regret_arrays.append(np.load(folder_name + suffix + "/regret.npy"))

    # plot the regret
    plt.plot(regret_arrays[0].cumsum() , '-' , label = args.alg_name)
    plt.title(f"Cumulative Regret for T = {args.horizon}, K = {args.num_outcomes}, d = {args.dim_arms}, and N = {args.num_arms}")
    plt.grid()
    plt.xlabel("Number of Rounds")
    plt.ylabel("Cumulative Regret")
    plt.legend()
    if args.xlim != [None , None]:
        plt.xlim(args.xlim)
    if args.ylim != [None , None]:
        plt.ylim(args.ylim)
    plt.savefig(f"{folder_name}/regret_{suffix[1:]}.png")


