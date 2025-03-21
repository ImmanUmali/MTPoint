import numpy as np

param_range = {
    "c2": [0.01, 0.2],
    "c3": [0.01, 0.1],
    "c4": [5, 50],
    "nu_c": [0.2, 0.7],
    "nu_e": [0.01, 0.4],
    "Wa": [0.001, 1.0],
    "Ter": [0.0,0.2]#,
    #"Negative_w" : [0.0, 5],
    #"Positive_w" : [0.0, 5]
}

param_symbol = {
    "c2": r"$c_2$",
    "c3": r"$c_3$",
    "c4": r"$c_4$",
    "nu_c": r"$\nu_c$",  
    "nu_e": r"$\nu_e$",
    "Wa": r"$W_a$",
    "Ter": r"$T_er$"#Y,
    ##############################################################################################-----------------------  Penalty - N
    #Y
    #"Negative_w": r"$Negative_w",
    #"Positive_w": r"$Positive_w$"
}

stat_range = {
    "dd_acc": [0.0 , 1],
    "mta_mean": [-0.6, 0.6],
    "mta_std": [0, 0.6],
    "t_cue": [0.05, 0.6],
    "P": [1.25, 1.8],
    "N": [1, 4]
}

#t_cue_random = np.random.uniform(0.05, 0.81, 4)
#p_random = np.random.uniform(0.5, 2.5, 2)
#n_random = np.random.choice([1, 2, 3, 4], 3, replace=False)

#condition_list = np.array([[t_cue, p, n] for t_cue in t_cue_random for p in p_random for n in n_random])

condition_list = np.array([[t_cue, p, n] for t_cue in [0.05, 0.175, 0.35, 0.6] for p in [1.25, 1.8] for n in [1, 2, 4]])

mta_dd_config = dict(
    name = "default",
    point_estimation = True,
    learning_rate = 0.00005,
    lr_gamma = 0.9,
    clipping = 1.0,
    amortizer = dict(
        device = None,
        trial_encoder_type = "attention",
        encoder = dict(     # Generate static size latent vector from simul.
            traj_sz = 0,
            stat_sz = len(stat_range.keys()),
            batch_norm = True,
            mlp = dict(     # Static
                feat_sz = 64,   # 64~128
                out_sz = 32,    # mlp out_sz : encoder out_sz -> reliability weight of data (stat, traj)
                depth = 3,
            ),
            transformer = dict( # Trajectory
                num_latents = 4,
                n_block = 2,
                query_sz = 8,   # 8~16
                out_sz = 8,     # 8~16
                head_sz = 8,
                n_head = 4,
                attn_dropout = 0.4, # 0.2~0.4 if low, training may be unstable
                res_dropout = 0.4,
                max_freq = 10,
                n_freq_bands = 2,
                max_step = 204,
            ),
            conv1d = [],
            rnn = dict(     # Trajectory
                type = "LSTM",
                bidirectional = True,
                dropout = 0.2,
                feat_sz = 8,     ### Experiment
                depth = 2,        ### Optional: 2 or 3
            ),
        ),
        ### DUMMY CONFIG
        trial_encoder = dict(
            attention = dict(
                num_latents = 4,
                n_block = 2,
                query_sz = 32,
                out_sz = 32,
                head_sz = 8,
                n_head = 4,
                attn_dropout = 0.4,
                res_dropout = 0.4,
            )
        ),
        ### DUMMY CONFIG END
        invertible = dict(
            param_sz = len(param_range.keys()),
            n_block = 5,
            act_norm = True,
            invert_conv = True,
            batch_norm = False,
            block = dict(
                permutation = False,
                head_depth = 2,
                head_sz = 32,
                cond_sz = 32,
                feat_sz = 32,
                depth = 2,
            )
        ),
        linear = dict(
            in_sz = 32,
            out_sz = len(param_range.keys()),
            hidden_sz = 256,
            hidden_depth = 2,
            batch_norm = False,
            activation = "relu",
        )
    ),
    simulator = dict(
        seed = None,
        targeted_params = list(param_range.keys()),
        targeted_y = list(stat_range.keys()),
    ),
)
