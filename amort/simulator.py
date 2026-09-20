import os, sys
from pathlib import Path
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from scipy.optimize import minimize
from skopt import forest_minimize
import warnings
import random
warnings.filterwarnings("ignore")
sys.path.append("..")

from configs.config import *
from utils.mymath import *


class Simulator(object):

    def __init__(self, config=None):
        config = deepcopy(mta_dd_config["simulator"]) if config is None else config
        self.targeted_params = config["targeted_params"]
        self.targeted_stat = config["targeted_y"]
        self.param_range = np.vstack([param_range[v] for v in self.targeted_params])
        self.stat_range = np.vstack([stat_range[v] for v in self.targeted_stat])
        self.condition_index = np.arange(len(condition_list))
    

    def simulate(
        self, 
        n_param=1,
        sim_per_param = 1,
        sim_per_cond=50,
        verbose=False,
        verbose_simul=False
    ):
        """
        Moving Target Acquisition + Drift Diffusion

        Arguments (Inputs):
        - n_param: no. of parameter sets to sample to simulate (only used when fixed_params is not given)
        - sim_per_param: no. of simulation per parameter set
        =======
        Free params in MMTA model
        1) c2   (min=0.01, max=0.3)
        2) c3   (min=0.01, max=0.1)
        3) c4   (min=5, max=50)
        4) nu_c (min=0.2, max=0.8)      Correct drift rate
        5) nu_e (min=0.0, max=0.8)      Error drift rate
        6) Wa   (min=0.0, max=10.0)      Anticipation Weight
        7) Ter  (min=0.0, max=0.3)      Non-Decision Time
        8) Negative_w   (min=0.0, max=5)        
        9) Positive_w   (min=0.0, max=5)
        =======
        Outputs:
        - norm_params: free parameter sets (with normalized values) used for simulation
            > ndarray with size ((n_param), (dim. of free parameters))
        - stats: static (fixed-size) behavioral outputs for every trial (see below)
            > ndarray with size ((n_param), (sim_per_param), (dim. of static behavior))
        =======
        Static behavioral output (normalized)
        1) Drift Diffusion: accuracy (min=0.25, max=1)
        2) MTA Press timing mean (min=-0.6, max=0.6)
        3) MTA Press timing std (min=0, max=0.5)

        4) t_cue (0.05 / 0.175 / 0.35 / 0.6)
        5) P (1.25 / 1.8)
        6) N (1 / 2 / 4)

        Per parameter, there are always 24 trial sets.
        On each task condition, 50 trials proceeded.
        =======
        """
        norm_params = np.random.uniform(low=-1, high=1, size=(n_param, len(self.targeted_params)))
        #norm_params = np.repeat(norm_params, sim_per_param, axis=0)

        stats = []
        for i in (tqdm(range(n_param)) if verbose else range(n_param)):
        #for i in (tqdm(range(n_param*sim_per_param)) if verbose else range(n_param*sim_per_param)):
            z = norm_params[i]
            w = v_denormalize(z, *self.param_range.T)
            s = self._simulate(*w, sim_per_cond=sim_per_cond, verbose=verbose_simul)
            s = v_normalize(s, *self.stat_range.T)

            stats.append(s)
        
        return norm_params, np.array(stats, dtype=np.float32)
        

    def _simulate(
        self, 
        c2, c3, c4, nu_c, nu_e, Wa, Ter,
        
        ##############################################################################################  -------------------------  Penalty - N
        # Y
        #Negative_w, Positive_w,
        ##############################################################################################
        
        
        s=0.1, step_size=0.00005, 
        sim_per_cond=100,
        task_order=None,
        verbose=False,
        return_ta=False
    ):
        """
        Return:
        Drift Diffusion result, MTA Press timing, T_cue, P, N
        """
        dd_acc, mta_mean, mta_std = list(), list(), list()
        ta_list = list()

        if task_order is None:
            np.random.shuffle(self.condition_index)
            task_order = np.copy(self.condition_index)

        delta = s * np.sqrt(step_size)

        for i, cond_i in enumerate(tqdm(task_order)) if verbose else enumerate(task_order):
            dd = list()
            mta = list()

            [t_cue, p, n] = condition_list[cond_i]
            n = int(n)
            
            ##################################################################################### ------------------------- Penalty - N
            # Y
            #ta = self._get_optimal_ta(c2, c3, c4, nu_c, nu_e, Wa, Ter, t_cue, p, n, Negative_w, Positive_w)
            
            # N
            ta = self._get_optimal_ta(c2, c3, c4, nu_c, nu_e, Wa, Ter, t_cue, p, n)
            #####################################################################################


            for _ in range(sim_per_cond):
                s_int = self._integrated_sigma(c2, c3, c4, ta, p, Ter)
                m_sampling = np.random.normal(ta, s_int)
                
                ##################################################################################### ------------------------- Ter - Both
                # DD
                T_sampling = m_sampling - Ter
                
                # MTA
                #T_sampling = m_sampling 
                ########################################################################################
                
                if n == 1:
                    d_result = 1

                
                else:
                   
                    if T_sampling < 0:
                      rands = random.random()
                      if rands < 1/n:
                        d_result = 0
                      else:
                        d_result = 1

                    
                    elif T_sampling - t_cue > 0.151:

                        delta = s * np.sqrt(step_size)
                        num_step = int(np.floor( (t_cue + 0.151 ) / step_size))
                        num_step_out = int(np.floor( (T_sampling - 0.151-t_cue) / step_size))
                        num_step_er = num_step + num_step_out
                        cor_p = (1 + nu_c * np.sqrt(step_size) / s) / 2
                            
                        ####################################################################################   -------------------------  ER - Log                 
                        # Log
                        err_p = (1 + (nu_e / np.log2(n)) * np.sqrt(step_size) / s) / 2
                            
                        # Sqrt
                        #err_p = (1 + (nu_e / np.sqrt(n)) * np.sqrt(step_size) / s) / 2                    
                        ######################################################################################
                            
                        b = np.random.binomial(num_step, cor_p)
                        c = np.random.binomial(num_step_out, cor_p)
                        d_sampling = [b - (num_step - b) + c - (num_step_out-c)]

                        for _ in range(n-1):
                            b = np.random.binomial(num_step_er, err_p)
                            d_sampling.append(b - (num_step_er - b))
                        d_sampling = np.array(d_sampling)

                        evidence = d_sampling * delta
                        max_index = np.argmax(evidence)
                        max_val = np.max(evidence)

                        if max_index == 0:
                            evidence = evidence[1:]
                            max_val2 = np.max(evidence)

                            if max_val != max_val2:
                                d_result = 0
                            else:
                                while max_val == max_val2:
                                    max_val += np.random.binomial(1, cor_p) * delta
                                    max_val2 += np.random.binomial(1, err_p) * delta
                                
                                d_result = 0 if max_val > max_val2 else 1
                        else:
                            d_result = 1
        

                    else:
                        evidence = np.zeros(n)

                        cor_p = (1 + nu_c * np.sqrt(step_size) / s) / 2
                        
                        ###################################################################################   ------------------------- ER - Log
                        
                        # Log
                        err_p = (1 + (nu_e / np.log2(n)) * np.sqrt(step_size) / s) / 2
                        
                        # Sqrt
                        #err_p = (1 + (nu_e / np.sqrt(n)) * np.sqrt(step_size) / s) / 2
                        
                        ######################################################################################
                        
                        num_step = max(1, int(np.floor(T_sampling / step_size)))

                        b = np.random.binomial(num_step, cor_p)
                        d_sampling = [b - (num_step - b)]

                        for _ in range(n-1):
                            b = np.random.binomial(num_step, err_p)
                            d_sampling.append(b - (num_step - b))
                        
                        d_sampling = np.array(d_sampling)
                        evidence = delta * d_sampling

                        max_index = np.argmax(evidence)
                        max_val = np.max(evidence)

                        if max_index == 0:
                            evidence = evidence[1:]
                            max_val2 = np.max(evidence)

                            if max_val != max_val2:
                                d_result = 1
                            else:
                                while max_val == max_val2:
                                    max_val += np.random.binomial(1, cor_p) * delta
                                    max_val2 += np.random.binomial(1, err_p) * delta
                                
                                d_result = 1 if max_val > max_val2 else 0
                        else:
                            d_result = 0
              
                dd.append(d_result)
                mta.append(m_sampling - t_cue)

            dd_acc.append(np.mean(dd))
            mta_mean.append(np.mean(mta))
            mta_std.append(np.std(mta))
            ta_list.append(ta)

        if not return_ta:
            return np.vstack((dd_acc, mta_mean, mta_std, *condition_list[task_order].T)).T
        
        return np.vstack((dd_acc, mta_mean, mta_std, *condition_list[task_order].T)).T, np.array(ta_list)
    
    ##################################################################################################################################################  -----------------------  Penalty - N
    # Y
    #def _get_optimal_ta(self, c2, c3, c4, nu_c, nu_e, Wa, Ter, t_cue, p, n, Negative_w, Positive_w, sample=500, interval=100):
        
    # N     
    def _get_optimal_ta(self, c2, c3, c4, nu_c, nu_e, Wa, Ter, t_cue, p, n, sample=200, interval=100):  
        #bounds = [(0, 0.8)]
        #initial_ta = t_cue
        ta = np.linspace(0.0, 0.8, sample)        
        left = -0.3
        right = 0.8

        
        def Ternary_search_min(f, left, right, epsilon=0.001):

            while right - left > epsilon:
                mid1 = left + (right - left) / 3
                mid2 = right - (right - left) / 3

                if f(mid1) < f(mid2):
                   right = mid2
                else:
                   left = mid1
                
            return (left + right) / 2            

        min_x = Ternary_search_min(lambda x: self._penalty(x, c2, c3, c4, nu_c, nu_e, Wa, Ter, t_cue, p, n), left, right)

        
        """

        from scipy.optimize import differential_evolution

        result = differential_evolution(
        lambda x: self._penalty(x, c2, c3, c4, nu_c, nu_e, Wa, Ter, t_cue, p, n), 
        bounds=[(max(t_cue-0.3, 0), t_cue+0.3)],
        strategy='best1bin',
        maxiter=20  # 조정 가능한 반복 횟수
          # 병렬 처리 (scipy 1.2.0 이상에서 사용 가능)
        )

        """
        #x0 = [[t_cue]]
        #y0 = [self._penalty(t_cue, c2, c3, c4, nu_c, nu_e, Wa, Ter, t_cue, p, n)]  


        #result = forest_minimize(lambda x: self._penalty(x, c2, c3, c4, nu_c, nu_e, Wa, Ter, t_cue, p, n), [(max(t_cue-0.3,0),t_cue+0.3)],  x0=x0, y0=y0, acq_func="EI", n_calls=30, n_random_starts=0, base_estimator="RF")
        
    ##################################################################################################################################################  -----------------------  Penalty - N
        # Y
        #res = np.array([self._penalty(_ta, c2, c3, c4, nu_c, nu_e, Wa, Ter, t_cue, p, n, Negative_w, Positive_w) for _ta in ta])        
        # N
        #result = minimize(self._penalty, initial_ta, args=(c2, c3, c4, nu_c, nu_e, Wa, Ter, t_cue, p, n), method='COBYLA', bounds=bounds)
        #res = np.array([self._penalty(_ta, c2, c3, c4, nu_c, nu_e, Wa, Ter, t_cue, p, n) for _ta in ta]) 
        #res = apply_gaussian_filter(res, size=interval-1, sigma=10)
        #if float(result.x[0]) > 0 : print (float(result.x[0]), n, t_cue, Wa, result.success)
        return min_x #ta[np.argmin(res)]  #min_x #result.x[0]
    ################################################################################################################################################## ----------------------  Penalty - N
    # Y
    #def _penalty(self, ta, c2, c3, c4, nu_c, nu_e, Wa, Ter, t_cue, p, n, Negative_w, Positive_w, s=0.1, step_size=0.00005, sample=150):

    # N
    def _penalty(self, ta, c2, c3, c4, nu_c, nu_e, Wa, Ter, t_cue, p, n, s=0.1, step_size=0.00005, sample=200):
    ########################################################################################################################################    
        s_int = self._integrated_sigma(c2, c3, c4, ta, p, Ter)

        dd_penalty_list = list()

        for _ in range(sample):
            m_sampling = np.random.normal(ta, s_int)

            ##################################################################################### ------------------------- Normalize - N
            # Y
            mta_penalty = (np.abs(m_sampling - t_cue)) / 0.151
            #mta_penalty = (np.abs(m_sampling - t_cue))
            # N
            # mta_penalty = (np.abs(m_sampling - t_cue))
            #####################################################################################
            
            ##################################################################################### -----------------------  Penalty - N
            #Y
            #mta_penalty = Negative_w * mta_penalty if m_sampling - t_cue <= 0 else Positive_w * mta_penalty
            
            # N
            
            #####################################################################################
                      
            #####################################################################################  ------------------------- Ter - Both                
            # DD,Both
            T_sampling = m_sampling - Ter
            # MTA
            #T_sampling = m_sampling 
            ######################################################################################

            if n == 1: dd_penalty = 0
            
            else:
               
                if T_sampling < 0:
                    rands = random.random()
                    if rands < 1/n:
                        dd_penalty = 0
                    else:
                        dd_penalty = 1

                elif T_sampling - t_cue > 0.151:

                    delta = s * np.sqrt(step_size)
                    num_step = int(np.floor( (t_cue + 0.151 ) / step_size))
                    num_step_out = int(np.floor( (T_sampling - 0.151 - t_cue) / step_size))
                    num_step_er = num_step + num_step_out
                    cor_p = (1 + nu_c * np.sqrt(step_size) / s) / 2
                        
                    ####################################################################################   -------------------------  ER - Log                 
                    # Log
                    err_p = (1 + (nu_e / np.log2(n)) * np.sqrt(step_size) / s) / 2
                        
                    # Sqrt
                    #err_p = (1 + (nu_e / np.sqrt(n)) * np.sqrt(step_size) / s) / 2                    
                    ######################################################################################
                        
                    b = np.random.binomial(num_step, cor_p)
                    c = np.random.binomial(num_step_out, err_p)
                    d_sampling = [b - (num_step - b) + c - (num_step_out-c)]

                    for _ in range(n-1):
                        b = np.random.binomial(num_step_er, err_p)
                        d_sampling.append(b - (num_step_er - b))
                    d_sampling = np.array(d_sampling)

                    evidence = d_sampling * delta
                    max_index = np.argmax(evidence)
                    max_val = np.max(evidence)

                    if max_index == 0:
                        evidence = evidence[1:]
                        max_val2 = np.max(evidence)

                        if max_val != max_val2:
                            dd_penalty = 0
                        else:
                            while max_val == max_val2:
                                max_val += np.random.binomial(1, cor_p) * delta
                                max_val2 += np.random.binomial(1, err_p) * delta
                            
                            dd_penalty = 0 if max_val > max_val2 else 1
                    else:
                        dd_penalty = 1


                else:
                    delta = s * np.sqrt(step_size)

                    cor_p = (1 + nu_c * np.sqrt(step_size) / s) / 2
                        
                    ####################################################################################   -------------------------  ER - Log                 
                    # Log
                    err_p = (1 + (nu_e / np.log2(n)) * np.sqrt(step_size) / s) / 2
                        
                    # Sqrt
                    #err_p = (1 + (nu_e / np.sqrt(n)) * np.sqrt(step_size) / s) / 2                    
                    ######################################################################################
                    num_step = int(np.floor( (T_sampling ) / step_size))
    
                    b = np.random.binomial(num_step, cor_p)
                    d_sampling = [b - (num_step - b)]

                    for _ in range(n-1):
                        b = np.random.binomial(num_step, err_p)
                        d_sampling.append(b - (num_step - b))
                    d_sampling = np.array(d_sampling)

                    evidence = d_sampling * delta
                    max_index = np.argmax(evidence)
                    max_val = np.max(evidence)

                    if max_index == 0:
                        evidence = evidence[1:]
                        max_val2 = np.max(evidence)

                        if max_val != max_val2:
                            dd_penalty = 0
                        else:
                            while max_val == max_val2:
                                max_val += np.random.binomial(1, cor_p) * delta
                                max_val2 += np.random.binomial(1, err_p) * delta
                            
                            dd_penalty = 0 if max_val > max_val2 else 1
                    else:
                        dd_penalty = 1
     
            dd_penalty_list.append((1-Wa) * dd_penalty + Wa * mta_penalty)

        return np.mean(dd_penalty_list)


    def _integrated_sigma(self, c2, c3, c4, ta, p, Ter):
        
        s_t = c2 * p
        #######################################################################################################################  -----------------------  Ter - Both       
        # DD
        #s_v = c3 + 1 / (np.exp(c4 * ta) - 1)
        
        # MTA,Both
        if ta - Ter > 0:
            s_v = c3 + 1 / (np.exp(c4 * (ta - Ter)) - 1)
        else: 
             s_v = c3 + 1 / (np.exp(c4 * 0.000000001) - 1)
        #######################################################################################################################        
        s_int = np.sqrt(s_t ** 2 * s_v ** 2 / (s_t ** 2 + s_v ** 2))
        return s_int
    

    def convert_from_output(self, outputs):
        param_in = deepcopy(outputs)
        if len(np.array(param_in).shape) == 1:
            param_in = np.expand_dims(param_in, axis=0)

        param_out = v_denormalize(param_in, *self.param_range.T)
        return param_out  


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    s = Simulator()
