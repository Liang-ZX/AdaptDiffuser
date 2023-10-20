import numpy as np
import torch
import pdb
import argparse
import os

# import gym
# import d4rl

from denoising_diffusion_pytorch.datasets.tamp import KukaDataset
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from denoising_diffusion_pytorch.mixer import MixerUnet
# from denoising_diffusion_pytorch.temporal import TemporalMixerUnet
from denoising_diffusion_pytorch.temporal_attention import TemporalUnet
from denoising_diffusion_pytorch.utils.rendering import KukaRenderer

parser = argparse.ArgumentParser()
parser.add_argument('--suffix', default='0', type=str, help='save dir suffix')
# parser.add_argument('--env_name', default="multiple_cube_kuka_temporal_convnew_real2_128", type=str, help='env name')
parser.add_argument('--data_path', default="kuka_dataset", type=str, help='dataset root path')  # 11649 numbers
parser.add_argument('--visualization', action='store_true', help='visualization')
parser.add_argument('--train_step', default=700000, type=int, help='train steps')
parser.add_argument('--old_data', default=None, type=str, help='old dataset root path')
parser.add_argument('--pretrain_path', default=None, type=str, help='pretrain diffusion model path')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

H = 128
diffusion_path = f'logs/multiple_cube_kuka_pick_conv_new_real2_{H}/{args.suffix}'

if args.old_data is None:
    args.old_data = os.path.join(os.environ['ADAPTDIFF_ROOT'], "kuka_dataset")  # TODO: You can use it if needed

#### dataset
# env = gym.make('hopper-medium-v2')
dataset = KukaDataset(H, data_path=args.data_path)
renderer = KukaRenderer()

old_dataset = KukaDataset(H, data_path=args.old_data) if args.old_data is not None else None

## dimensions
obs_dim = dataset.obs_dim

print(args)

#### model
# model = Unet(
#     width = H,
#     dim = 32,
#     dim_mults = (1, 2, 4, 8),
#     channels = 2,
#     out_dim = 1,
# ).cuda()

if not os.path.exists(diffusion_path):
    os.makedirs(diffusion_path)

diffusion_epoch = 0

# model = MixerUnet(
#     dim = 32,
#     image_size = (H, obs_dim),
#     dim_mults = (1, 2, 4, 8),
#     channels = 2,
#     out_dim = 1,
# ).cuda()

# model = MixerUnet(
#     horizon = H,
#     transition_dim = obs_dim,
#     cond_dim = H,
#     dim = 32,
#     dim_mults = (1, 2, 4, 8),
# ).cuda()

model = TemporalUnet(
    horizon = H,
    transition_dim = obs_dim,
    cond_dim = H,
    dim = 128,
    dim_mults = (1, 2, 4, 8),
).cuda()


diffusion = GaussianDiffusion(
    model,
    channels = 1,
    image_size = (H, obs_dim),
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()

# #### test
# print('testing forward')
# x = dataset[0][0].view(1, H, obs_dim).cuda()
# mask = torch.zeros(1, H).cuda()
#
# loss = diffusion(x, mask)
# loss.backward()
# print('done')
# # pdb.set_trace()
# ####

trainer = Trainer(
    diffusion,
    dataset,
    renderer,
    train_batch_size = 32,
    train_lr = 2e-5,
    train_num_steps = args.train_step,  # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    fp16 = False,                     # turn on mixed precision training with apex
    results_folder = diffusion_path,
    save_and_sample_every = 1000,  # 1000
    visualization=args.visualization,
    old_dataset=old_dataset,
)


if args.pretrain_path is not None:
    print("Loading pretrain")
    trainer.load(loadpath=args.pretrain_path)


trainer.train()
