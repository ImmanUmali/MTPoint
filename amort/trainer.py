"""
Trainer for amortized inference
Orignial code written by Hee-seung Moon (https://github.com/hsmoon121/amortized-inference-hci)

Code modified by June-Seop Yoon
"""

import enum
import os, sys
from time import time
from copy import deepcopy
from pathlib import Path
from abc import ABC, abstractmethod
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import torch
from joblib import Parallel, delayed

sys.path.append("..")

from nets.amortizer import AmortizerForTrialData, RegressionForTrialData
from amort.simulator import Simulator
from amort.dataset import UserDataset, TrainDataset, ValidDataset
from utils.schedulers import CosAnnealWR
from configs.config import *
from configs.path import *
from utils.loggers import Logger
from utils.plot import *
from utils.utils import now_to_string
from utils.mymath import *

class Trainer(ABC):
    def __init__(self, config=None):
        if config is None:
            self.config = deepcopy(mta_dd_config)
        else:
            self.config = config
        self.iter = 0
        self.name = self.config["name"]

        # Initialize the amortizer, simulator, and datasets
        self.point_estimation = self.config["point_estimation"]
        amortizer_fn = RegressionForTrialData if self.point_estimation else AmortizerForTrialData
        self.amortizer = amortizer_fn(config=self.config["amortizer"])
        self.simulator = Simulator(config=self.config["simulator"])

        self.user_dataset = UserDataset()
        self.valid_dataset = ValidDataset(sim_config=self.config["simulator"])
        self.train_dataset = TrainDataset(sim_config=self.config["simulator"])

        self.targeted_params = self.config["simulator"]["targeted_params"]
        self.targeted_stat = self.config["simulator"]["targeted_y"]
        self.param_symbol = [param_symbol[v] for v in self.targeted_params]
        self.stat_range = np.array([stat_range[v] for v in self.targeted_stat])

        # Initialize the optimizer and scheduler
        self.lr = self.config["learning_rate"]
        self.lr_gamma = self.config["lr_gamma"]
        self.clipping = self.config["clipping"]
        self.optimizer = torch.optim.Adam(self.amortizer.parameters(), lr=1e-9)
        self.scheduler = CosAnnealWR(self.optimizer, T_0=10, T_mult=1, eta_max=self.lr, T_up=1, gamma=self.lr_gamma)

        self.datetime_str = now_to_string(omit_year=True, omit_ms=True)
        self.model_path = PATH_AMORT_MODEL
        self.board_path = PATH_AMORT_BOARD
        self.result_path = PATH_AMORT_RESULT
        self.clipping = float("Inf")

    def train(
        self,
        n_iter=300,     # 20~, 100~200
        step_per_iter=2048,
        batch_sz=64,    # As maximum as possible (memory & speed)
        board=True,
        save_freq=10,
    ):
        """
        Training loop

        n_iter (int): Number of training iterations
        step_per_iter (int): Number of training steps per iteration
        batch_sz (int): Batch size
        n_trial (int): Number of trials for each user data (default: 1)
        board (bool): Whether to use tensorboard (default: True)
        """
        iter = self.iter
        last_step = self.iter * step_per_iter

        self.logger = Logger(
            self.name, 
            self.datetime_str,
            last_step=last_step, 
            board=board, 
            board_path=self.board_path
        )

        # Training iterations
        losses = dict()
        print(f"\n[ Training - {self.name} ]")
        for iter in range(self.iter + 1, n_iter + 1):
            losses[iter] = []

            # Training loop
            with tqdm(total=step_per_iter, desc=f" Iter {iter}") as progress:
                for step in range(step_per_iter):
                    batch_args = self.train_dataset.sample(batch_sz=batch_sz)

                    # Training step
                    loss = self._train_step(*batch_args)
                    losses[iter].append(loss)

                    # Logging
                    if step % 10 == 0:
                        self.logger.write_scalar(train_loss=loss, lr=self.scheduler.get_last_lr()[0])
                    progress.set_postfix_str(f"Avg.Loss: {np.mean(losses[iter]):.3f}")
                    progress.update(1)
                    self.logger.step()
                    self.scheduler.step((iter-1) + step/step_per_iter)
                    
                    if np.isnan(loss):
                        raise RuntimeError("Nan loss computed.")

            # Save model
            if iter % save_freq == 0:
                self.save(iter)
                valid_res = self.valid()
                self.logger.write_scalar(**valid_res)
            self.iter = iter

        print("\n[ Training Done ]")
        if iter in losses:
            print(f"  Training Loss: {np.mean(losses[iter])}\n")
        


    def _train_step(self, params, stat_data, traj_data=None):
        """
        Training step

        params (ndarray): [n_sim, n_param] array of parameters
        stat_data (list): [n_sim] list of static data
        traj_data (list): [n_sim] list of trajectories (default: None)
        """
        self.amortizer.train()
        if self.point_estimation:
            params_tensor = torch.FloatTensor(params).to(self.amortizer.device)
            loss = F.mse_loss(self.amortizer(stat_data, traj_data), params_tensor)
        else:
            z, log_det_J = self.amortizer(params, stat_data, traj_data)
            loss = torch.mean(0.5 * torch.square(torch.norm(z, dim=-1)) - log_det_J)
        return self._optim_step(loss)
    

    def _optim_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.amortizer.parameters():
            param.grad.data.clamp_(-self.clipping, self.clipping)
        self.optimizer.step()
        return loss.item()


    def save(self, iter, path=None):
        """
        Save model, optimizer, and scheduler with iteration number
        """
        if path is None:
            os.makedirs(f"{self.model_path}/{self.name}/{self.datetime_str}", exist_ok=True)
            ckpt_path = f"{self.model_path}/{self.name}/{self.datetime_str}/iter{iter:03d}.pt"
        else:
            os.makedirs(path, exist_ok=True)
            ckpt_path = path + f"iter{iter:03d}.pt"
        torch.save({
            "iteration": iter,
            "model_state_dict": self.amortizer.state_dict(),
            "optim_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }, ckpt_path)
        

    def load(self, model_name, model_session):
        """
        Load model, optimizer, and scheduler from the latest checkpoint
        """
        import glob
        ckpt_paths = glob.glob(f"{self.model_path}/{model_name}/{model_session}/iter*.pt")
        ckpt_paths.sort()
        ckpt_path = ckpt_paths[-1]

        self.name = model_name
        self.datetime_str = model_session

        ckpt = torch.load(ckpt_path, map_location=self.amortizer.device.type)
        self.amortizer.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optim_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        iter = ckpt["iteration"]
        self.scheduler.step(iter)

        print(f"[ amortizer - loaded checkpoint ]\n\t{model_name} - {model_session} - {iter}")

        self.iter = iter


    def _find_last_ckpt(self, save_path):
        """
        Find the latest checkpoint
        """
        os.makedirs(save_path, exist_ok=True)
        ckpts = os.listdir(save_path)
        ckpts_with_iter = [f for f in ckpts if f.startswith("iter")]
        if ckpts_with_iter:
            return os.path.join(save_path, max(ckpts_with_iter))
        elif ckpts:
            return os.path.join(save_path, max(ckpts))
        else:
            return None


    def valid(
        self,
        n_sample=200,   # Sample for distribution estimation
        infer_type="mode",
        verbose=True
    ):
        self.amortizer.eval()
        valid_res = dict()

        ### 1) Parameter recovery from simulated
        start_t = time()

        sim_gt_params, sim_valid_data = self.valid_dataset.sample()
        self.parameter_recovery(
            valid_res,
            sim_gt_params,
            sim_valid_data,
            n_sample,
            infer_type,
            surfix="_sim",
        )
        if verbose:
            print(f"- parameter recovery (simulated) ({time() - start_t:.3f}s)")
        
        ### 2) Compare simulation and human players
        start_t = time()
        self.model_fitting(
            n_sample, 
            infer_type
        )
        if verbose:
            print(f"- user simulation ({time() - start_t:.3f}s)")

        return valid_res


    def parameter_recovery(
        self,
        res,
        gt_params,
        valid_data,
        n_sample,
        infer_type,
        surfix="",
    ):
        # Note: all parameters are normalized: -1 ~ 1
        n_param = gt_params.shape[0]
        inferred_params = list()

        for param_i in range(n_param):
            stat_i = valid_data[param_i] # valid_data := list[(stat, traj)]
            norm_param = self.amortizer.infer(stat_i, n_sample=n_sample, type=infer_type)
            norm_param = self._clip_params(norm_param)
            inferred_params.append(self.simulator.convert_from_output(norm_param)[0])
        gt_params = self.simulator.convert_from_output(gt_params)
        inferred_params = np.array(inferred_params)
        
        r_squared = plot_parameter_recovery(
            gt_params,
            inferred_params,
            fname="r2_params",
            param_labels=self.param_symbol,
            fpath=f"{self.result_path}/{self.name}/{self.datetime_str}/iter{self.iter:03d}/"
        )

        for i, l in enumerate(self.targeted_params):
            res["Parameter_Recovery/r2_" + l + surfix] = r_squared[i]
    

    def model_fitting(
        self,
        n_sample, 
        infer_type
    ):
        if self.user_dataset.n_user == 0:
            print("- user simulation skipped: no experiment data available")
            return

        inferred_param = list()
        merged_result = list()
        exp_result = list()

        for p in range(self.user_dataset.n_user):
            stat = self.user_dataset.sample(p, shuffle=False)
            norm_param = self.amortizer.infer(stat, n_sample=n_sample, type=infer_type)
            norm_param = self._clip_params(norm_param)
            denorm_param = self.simulator.convert_from_output(norm_param)[0]
            inferred_param.append(denorm_param)

            denorm_stat = v_denormalize(stat, *self.stat_range.T)
            exp_result.append(dict(
                dd_acc=denorm_stat[:,0],
                mta_mean=denorm_stat[:,1],
                mta_std=denorm_stat[:,2],
            ))

        inferred_param = np.array(inferred_param)

        def run_user_simul(user_no):
            simul_result, ta_result = self.simulator._simulate(
                *inferred_param[user_no],
                sim_per_cond=500,
                task_order=np.arange(len(condition_list)),
                return_ta=True
            )
            return pd.DataFrame(dict(
                user=[user_no+1 for _ in range(simul_result.shape[0])],
                task_index=np.arange(len(condition_list)),
                t_cue=condition_list[:,0],
                p=condition_list[:,1],
                n=condition_list[:,2],
                dd_acc_exp=exp_result[user_no]["dd_acc"],
                dd_acc_sim=simul_result[:,0],
                mta_mean_exp=exp_result[user_no]["mta_mean"],
                mta_mean_sim=simul_result[:,1],
                mta_std_exp=exp_result[user_no]["mta_std"],
                mta_std_sim=simul_result[:,2],
                ta=ta_result
            ))
        
        # eps = Parallel(n_jobs=2)(
        #     delayed(run_user_simul)(i) for i in tqdm(range(21))
        # )
        # merged_result = pd.concat(eps)
        merged_result = pd.concat([run_user_simul(p) for p in tqdm(range(21))])
        merged_result.to_csv(f"{self.result_path}/{self.name}/{self.datetime_str}/iter{self.iter:03d}/simul.csv", index=False)

        plot_fitting_per_user(
            merged_result, 
            f"{self.result_path}/{self.name}/{self.datetime_str}/iter{self.iter:03d}/"
        )

        for mt in ["dd_acc", "mta_mean", "mta_std"]:
            merged_result[f"{mt}_mae"] = np.abs(merged_result[f"{mt}_exp"] - merged_result[f"{mt}_sim"])
        merged_result = merged_result.groupby("user", as_index=False).mean()

        pd.DataFrame(merged_result[["user", "dd_acc_mae", "mta_mean_mae", "mta_std_mae"]]).to_csv(f"{self.result_path}/{self.name}/{self.datetime_str}/iter{self.iter:03d}/mae.csv", index=False)

        df_param = dict(user=np.arange(1, 22))
        for i, v in enumerate(self.targeted_params):
            df_param[v] = inferred_param[:,i]
        
        df_param = pd.DataFrame(df_param)
        df_param.to_csv(f"{self.result_path}/{self.name}/{self.datetime_str}/iter{self.iter:03d}/param.csv", index=False)


    def _clip_params(self, params):
        return np.clip(
            params,
            np.array([-1.] * len(self.targeted_params)),
            np.array([1.] * len(self.targeted_params))
        )