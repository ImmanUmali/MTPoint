import argparse

import sys, copy
sys.path.append('..')

from configs.config import *
from amort.trainer import Trainer

parser = argparse.ArgumentParser(description='Training option')


parser.add_argument('--amort', type=str, default='pte', choices=['inn', 'pte'])

# MLP for stat data
parser.add_argument('--mlp_feat', type=int, default=64)
parser.add_argument('--mlp_out', type=int, default=32)

# INN
parser.add_argument('--inn_block_feat', type=int, default=16)

# PTE
parser.add_argument('--pte_hid_sz', type=int, default=256)
parser.add_argument('--pte_hid_depth', type=int, default=2)

# Training setup
parser.add_argument('--step_per_iter', type=int, default=1024)
parser.add_argument('--batch_sz', type=int, default=128)
parser.add_argument('--n_iter', type=int, default=100)
parser.add_argument('--save_freq', type=int, default=10)

parser.add_argument('--load_ckpt', type=bool, default=False)
parser.add_argument('--load_model', type=str, default="mlp_f64_o32-pte_256x2-tr_it1024_b128")
parser.add_argument('--load_session', type=str, default="0208_012206")

args = parser.parse_args()

### Configuration setup
cfg = copy.deepcopy(mta_dd_config)

if args.amort == 'inn': cfg["point_estimation"] = False
elif args.amort == 'pte': cfg["point_estimation"] = True
cfg["amortizer"]["encoder"]["mlp"]["feat_sz"] = args.mlp_feat
cfg["amortizer"]["encoder"]["mlp"]["out_sz"] = args.mlp_out
cfg["amortizer"]["invertible"]["block"]["feat_sz"] = args.inn_block_feat
cfg["amortizer"]["linear"]["hidden_sz"] = args.pte_hid_sz
cfg["amortizer"]["linear"]["hidden_depth"] = args.pte_hid_depth


name = f"mlp_f{args.mlp_feat}_o{args.mlp_out}"
if args.amort == 'inn':
    name += f"-inn_f{args.inn_block_feat}"
elif args.amort == 'pte':
    name += f"-pte_{args.pte_hid_sz}x{args.pte_hid_depth}"
name += f"-tr_it{args.step_per_iter}_b{args.batch_sz}"

cfg["name"] = name
trainer = Trainer(config=cfg)

if args.load_ckpt:
    trainer.load(
        args.load_model,
        args.load_session
    )

trainer.train(
    n_iter=args.n_iter,
    step_per_iter=args.step_per_iter,
    batch_sz=args.batch_sz,
    save_freq=args.save_freq,
)