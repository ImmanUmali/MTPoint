import os, pickle, sys, glob, random
from pathlib import Path
from copy import deepcopy
import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm
from joblib import Parallel, delayed
from datetime import datetime

sys.path.append("..")

from amort.simulator import Simulator
from configs.config import *
from configs.path import *
from utils.utils import *
from utils.mymath import *

# used for recorded data
class UserDataset(object):
    def __init__(self, config=None):
        config = deepcopy(mta_dd_config["simulator"]) if config is None else config
        self.targeted_stat = config["targeted_y"]

        data_paths = [f"{PATH_EXP_DATA}/P_{i}_results.csv" for i in range(1, 22)]
        missing_paths = [path for path in data_paths if not os.path.exists(path)]
        if missing_paths:
            self.dataset = []
            self.n_user = 0
            print("[ user dataset ] experiment data not found; user-data validation disabled")
        else:
            self.dataset = [pd.read_csv(path) for path in data_paths]
            self.n_user = len(self.dataset)
            print(f"[ user dataset ] {self.n_user} loaded")
        self.stat_range = np.vstack([stat_range[v] for v in self.targeted_stat])
    
    def sample(self, n_user, shuffle=True):
        data = self.dataset[n_user]
        # task = np.array([
        #     task_index(t_cue, p, n) for (t_cue, p, n) in zip(data["t_cue"], data["p"], data["n"])
        # ])
        stat = data[[
            "DD_Cor",
            "mean",
            "std",
            "t_cue",
            "p", 
            "n"
        ]].to_numpy()
        stat = v_normalize(stat, *self.stat_range.T)
        if shuffle: np.random.shuffle(stat)
        return stat




class TrainDataset(object):
    def __init__(self, n_ep=50, sim_config=None, load_existing_data=True):
        if sim_config is None:
            self.sim_config = deepcopy(mta_dd_config["simulator"])
        else:
            self.sim_config = deepcopy(sim_config)
        self.sim_config["seed"] = 100
        self.n_ep = n_ep
        if load_existing_data: self._get_dataset()
    

    def _get_dataset(self):
        """Load an existing dataset from file."""
        self.datafilelist = glob.glob(f"{PATH_AMORT_SIM_DATASET}train_*_step_{self.n_ep}ep.pkl")
        # No data found
        if len(self.datafilelist) == 0:
            print("WARNING!!! NO DATASET EXIST!!!")
            return

        self.datafilelist.sort()
        self.n_param = 0

        params = list()
        stats = list()
        for df in self.datafilelist:
            d = pickle_load(df)
            p, s = d["params"], d["stat_data"]
            self.n_param += p.shape[0]
            params.append(p)
            stats.append(s)
        
        params = np.concatenate(params, axis=0, dtype=np.float32)
        stats = np.concatenate(stats, axis=0, dtype=np.float32)
        self.dataset = dict(
            params=params,
            stat_data=stats
        )

        print(f"[ simulated dataset ] {self.n_param} parameters, {self.n_ep} trials per condition ")


    def _generate_dataset(self, total_param=2**23, save_param=2**21, num_cpu=12):
        """Generate simulation dataset"""
        self.simulator = Simulator(self.sim_config)
        os.makedirs(PATH_AMORT_SIM_DATASET, exist_ok=True)

        save_freq = int(np.ceil(total_param / save_param))

        def get_simul_res(simulator, i):
            np.random.seed(datetime.now().microsecond + i)
            args = simulator.simulate(
                n_param=1,
                sim_per_cond=self.n_ep,
                verbose=False,
                verbose_simul=False
            )
            return args

        for _ in range(save_freq):
            sub_param = total_param // save_freq

            eps = Parallel(n_jobs=num_cpu)(
                delayed(get_simul_res)(self.simulator, i) for i in tqdm(range(sub_param))
            )

            params_arr = np.concatenate([eps[i][0] for i in range(sub_param)], axis=0, dtype=np.float32)
            stats_arr = np.concatenate([eps[i][1] for i in range(sub_param)], axis=0, dtype=np.float32)

            pickle_save(
                f"{PATH_AMORT_SIM_DATASET}train_{now_to_string(omit_year=True)}_{sub_param:08d}_step_{self.n_ep}ep.pkl", 
                dict(
                    params=params_arr,         # np.array (n_param, param_sz)
                    stat_data=stats_arr,       # np.array (n_param, 24, stat_sz)
                )
            )


    def sample(self, batch_sz):
        # Parameter selection
        indices = np.random.choice(self.n_param, batch_sz, replace=False)
        
        return (
            self.dataset["params"][indices],
            self.dataset["stat_data"][indices]
        )


        # ep_indices = np.random.choice(self.n_ep, sim_per_param)
        # rows = np.repeat(indices, sim_per_param).reshape((-1, sim_per_param))
        # cols = np.tile(ep_indices, (batch_sz, 1))
        # if sim_per_param == 1:
        #     return (
        #         self.dataset["params"][indices],
        #         self.dataset["stat_data"][rows, cols].squeeze(1)
        #     )
        # else:
        # p = self.dataset["params"][indices]
        # p = np.repeat(p, sim_per_param, axis=0)
        # s = self.dataset["stat_data"][rows, cols]
        # s = np.concatenate(s, axis=0)
        # return (p, s)



class ValidDataset(object):
    def __init__(self, total_user=100, trial_per_cond=50, sim_config=None, load_existing_data=True):
        self.total_user = total_user
        self.trial_per_cond = trial_per_cond
        if sim_config is None:
            self.sim_config = deepcopy(mta_dd_config["simulator"])
        else:
            self.sim_config = deepcopy(sim_config)
        self.sim_config["seed"] = 121
        
        self.fpath = f"{PATH_AMORT_SIM_DATASET}valid_{total_user}_param_{trial_per_cond}ep.pkl"
        if load_existing_data: self._get_dataset()


    def _get_dataset(self):
        """
        Load an existing dataset from file or create a new dataset using the PnCSimulator.
        """
        if not os.path.exists(self.fpath): self._generate_dataset()
        self.dataset = pickle_load(self.fpath)
        self.n_param = self.dataset["params"].shape[0]


    def _generate_dataset(self, num_cpu=12):
        self.simulator = Simulator(self.sim_config)
        os.makedirs(PATH_AMORT_SIM_DATASET, exist_ok=True)

        def get_simul_res(simulator, i):
            np.random.seed(datetime.now().microsecond + i)
            args = simulator.simulate(
                n_param=1,
                sim_per_cond=self.trial_per_cond,
                verbose=False,
                verbose_simul=False
            )
            return args

        # Parallelize the creation of the dataset.
        eps = Parallel(n_jobs=num_cpu)(
            delayed(get_simul_res)(self.simulator, i) for i in tqdm(range(self.total_user))
        )
        params_arr = np.concatenate([eps[i][0] for i in range(self.total_user)], axis=0, dtype=np.float32)
        stats_arr = np.concatenate([eps[i][1] for i in range(self.total_user)], axis=0, dtype=np.float32)

        pickle_save(
            self.fpath,
            dict(
                params=params_arr,         # np.array (n_param, param_sz)
                stat_data=stats_arr,          # np.array (n_param, n_ep, stat_sz
            )
        )


    def sample(self, n_user=None):
        """
        Returns a sample from the dataset with the specified number of users (sampled parameters and stats).
        If the number of users is not specified, it defaults to the total number of parameters in the dataset.
        """
        if n_user is None:
            n_user = self.total_user
        params = list()
        stats = list()
        for user in range(n_user):
            params.append(self.dataset["params"][user])
            stats.append(self.dataset["stat_data"][user, :])
        return np.array(params, dtype=np.float32), np.array(stats, dtype=np.float32)
        

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Data synthesis')

    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--valid', type=bool, default=False)

    parser.add_argument('--n_ep', type=int, default=50)
    parser.add_argument('--cpu', type=int, default=16)
    parser.add_argument('--exp', type=int, default=20)
    parser.add_argument('--save_exp', type=int, default=14)
    parser.add_argument('--mul', type=int, default=1)

    args = parser.parse_args()

    if args.valid:
        ValidDataset(load_existing_data=False, trial_per_cond=args.n_ep)._generate_dataset(num_cpu=args.cpu)

    if args.train: 
        TrainDataset(load_existing_data=False, n_ep=args.n_ep)._generate_dataset(
            total_param=2**args.exp * args.mul, 
            save_param=2**args.save_exp,
            num_cpu=args.cpu
        )
    
