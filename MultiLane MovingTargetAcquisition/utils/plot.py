import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde, pearsonr, ttest_ind
from scipy.optimize import minimize_scalar
from time import time
from tqdm import tqdm
from sklearn.metrics import r2_score

sys.path.append("..")

from utils.mymath import *


def fig_save(dir, fn, DPI=100, save_svg=False):
    os.makedirs(dir, exist_ok=True)
    if dir[-1] == '/':
        fullpath = f"{dir}{fn}"
    else:
        fullpath = f"{dir}/{fn}"
    plt.savefig(f"{fullpath}.png", dpi=DPI, bbox_inches='tight', pad_inches=0)
    if save_svg:
        while True:
            try:
                plt.savefig(f"{fullpath}.pdf", dpi=DPI, bbox_inches='tight', pad_inches=0)
                break
            except PermissionError:
                _ = input("The file seems to be opened... close and press enter to proceed.")
    plt.close('all')


def plot_parameter_recovery(
    p_true,     # np.ndarray with shape (batch_sz, param_sz)
    p_pred,     # np.ndarray with shape (batch_sz, param_sz)
    fname,
    param_labels=None,
    fpath=None,
):
    """
    ##### https://github.com/hsmoon121/amortized-inference-hci
    Plot the comparison between true and inferred parameter values, and return R2 values

    Code modification: now all R2 plots of parameters are drawn in a single figure
    """
    plt.rcParams["font.family"] = "sans-serif"
    # plt.rcParams["font.size"] = 18
    # plt.rcParams["axes.linewidth"] = 2
    sns.set_style("white")
    r2_list = []

    # Grid configuration
    _r, _c = find_divisors(p_true.shape[1])
    fig, axs = plt.subplots(_r, _c, figsize=np.array([_c,_r])*3, constrained_layout=True)
    
    for i in range(p_true.shape[1]):
        y_true = p_true[:, i]
        y_pred = p_pred[:, i]
        y_fit = np.polyfit(y_true, y_pred, 1)
        y_func = np.poly1d(y_fit)
        r_squared = r2_score(y_pred, y_func(y_true))
        r2_list.append(r_squared)

        max_val = max(max(y_true), max(y_pred))
        min_val = min(min(y_true), min(y_pred))
        dist_val = (max_val - min_val) * 0.1

        label = f"$({y_fit[0]:.2f})x + ({y_fit[1]:.2f}), R^2={r_squared:.2f}$"

        r, c = i // _c, i % _c
        ax = axs[r][c] if _r > 1 else axs[c]

        sns.regplot(
            x=y_true,
            y=y_pred,
            scatter_kws={"color": "black", "alpha": 0.3},
            line_kws={"color": "red", "lw": 2.5},
            ax=ax
        )
        ax.plot(
            [min_val - dist_val, max_val + dist_val],
            [min_val - dist_val, max_val + dist_val],
            color="gray",
            linestyle="--"
        )
        ax.set_title(f"Parameter {param_labels[i]}")
        if r == _r // 2 and c == 0:
            ax.set_ylabel(f"Inferred parameter")
        if c == _c // 2 and r == _r-1:
            ax.set_xlabel(f"True parameter")
        
        ax.set_xlim([min_val - dist_val, max_val + dist_val])
        ax.set_ylim([min_val - dist_val, max_val + dist_val])
        ax.legend([Line2D([0], [0], color="red", lw=2.5)], [label,], fontsize=8, loc="lower right")
        ax.grid(linestyle="--", linewidth=0.5)
        ax.set_aspect("equal")

    fig_save(fpath, fname, DPI=300, save_svg=True)
    plt.close(fig)

    return r2_list


def plot_fitting_per_user(
    data,
    fpath
):
    user_list = data["user"].unique()
    r2_table = {"user": [*(i+1 for i in range(len(user_list))), "mean"]}
    _r, _c = find_divisors(len(user_list))
    for key in ["dd_acc", "mta_mean", "mta_std"]:
        r2s = []
        fig, axs = plt.subplots(_r, _c, figsize=np.array([_c, _r])*3, constrained_layout=True)

        total_max = -9999
        total_min = 9999

        for user in user_list:
            target_data = data[data["user"] == user]
            y_true = target_data[f"{key}_exp"].to_numpy()
            y_pred = target_data[f"{key}_sim"].to_numpy()

            r_squared = compute_r2(y_true, y_pred)
            r2s.append(r_squared)
            max_val = max(max(y_true), max(y_pred))
            min_val = min(min(y_true), min(y_pred))
            dist_val = (max_val - min_val) * 0.1

            if total_max < max_val: total_max = max_val
            if total_min > min_val: total_min = min_val

            label = f"$R^2={r_squared:.2f}$"

            ax = axs[(user-1)//_c][(user-1)%_c]

            sns.regplot(
                x=y_true,
                y=y_pred,
                scatter_kws={"color": "black", "alpha": 0.3},
                line_kws={"color": "red", "lw": 2.5},
                ax=ax
            )
            ax.plot(
                [min_val - dist_val, max_val + dist_val],
                [min_val - dist_val, max_val + dist_val],
                color="gray",
                linestyle="--"
            )
            ax.set_title(f"User No. {user:02d}")
            
            ax.legend(
                [Line2D([0], [0], color="red", lw=2.5)], 
                [label,], 
                fontsize=8, loc="lower right"
            )
        
        d_val = 0.1 * (total_max - total_min)
        
        ###

        ax.set_xlim([min_val - dist_val, max_val + dist_val])
        ax.set_ylim([min_val - dist_val, max_val + dist_val])
        ax.grid(True)
        
        
        """
        for r in range(_r):
            for c in range(_c):
                axs[r][c].set_xlim(total_min - d_val, total_max + d_val)
                axs[r][c].set_ylim(total_min - d_val, total_max + d_val)
                axs[r][c].grid(True)

        """
        
        
        fig_save(fpath, f"user_{key}", DPI=300, save_svg=True)
        plt.close(fig)

        r2_table[key] = [*r2s, np.mean(r2s)]
    
    pd.DataFrame(r2_table).to_csv(f"{fpath}/r2_table.csv", index=False)


# def plot_fitting_per_cond(
#     data,
#     fpath
# ):
#     for key in ["dd_result", "mta_press"]:
#         fig, axs = plt.subplots(4, 6, figsize=np.array([6, 4])*3, constrained_layout=True)
#         d = data.groupby(["user", "task_index"], as_index=False).mean()
#         for cond in range(24):
#             target_data = d[d["task_index"] == cond]
#             y_true = target_data[f"{key}_exp"].to_numpy()
#             y_pred = target_data[f"{key}_sim"].to_numpy()

#             r_squared = compute_r2(y_true, y_pred)
#             max_val = max(max(y_true), max(y_pred))
#             min_val = min(min(y_true), min(y_pred))
#             dist_val = (max_val - min_val) * 0.1

#             label = f"$R^2={r_squared:.2f}$"

#             ax = axs[cond//6][cond%6]

#             sns.regplot(
#                 x=y_true,
#                 y=y_pred,
#                 scatter_kws={"color": "black", "alpha": 0.3},
#                 line_kws={"color": "red", "lw": 2.5},
#                 ax=ax
#             )
#             ax.plot(
#                 [min_val - dist_val, max_val + dist_val],
#                 [min_val - dist_val, max_val + dist_val],
#                 color="gray",
#                 linestyle="--"
#             )
#             t_cue, p, n = task_cond(cond)
#             ax.set_title(f"$t_cue$={t_cue:.2f}, p={p:.2f}, n={n}")
#             if key == 'dd_result':
#                 ax.set_xlim(0, 1)
#                 ax.set_ylim(0, 1)
#                 ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
#                 ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
#             else:
#                 ax.set_xlim(-0.3, 1)
#                 ax.set_ylim(-0.3, 1)
#                 ax.set_xticks([-0.3, 0, 0.3, 0.6, 0.9])
#                 ax.set_yticks([-0.3, 0, 0.3, 0.6, 0.9])
#             ax.grid(True)
            
#             ax.legend(
#                 [Line2D([0], [0], color="red", lw=2.5)], 
#                 [label,], 
#                 fontsize=8, loc="lower right"
#             )

#         fig_save(fpath, f"cond_{key}", DPI=300, save_svg=True)
#         plt.close(fig)