import os
import os.path as osp
import numpy as np
import torch
import pdb
import pybullet as p
import argparse

# import gym
# import d4rl

from denoising_diffusion_pytorch.denoising_diffusion_pytorch import GaussianDiffusion
from denoising_diffusion_pytorch import Trainer
from denoising_diffusion_pytorch.datasets.tamp import KukaDataset
from denoising_diffusion_pytorch.mixer_old import MixerUnet
from denoising_diffusion_pytorch.mixer import MixerUnet as MixerUnetNew
from denoising_diffusion_pytorch.temporal_attention import TemporalUnet
from denoising_diffusion_pytorch.utils.rendering import KukaRenderer
import diffusion.utils as utils
# import environments
from imageio import get_writer
import torch.nn as nn

from diffusion.models.mlp import TimeConditionedMLP
from diffusion.models import Config

from gym_stacking.utils import get_bodies, sample_placement, pairwise_collision, \
    RED, GREEN, BLUE, BLACK, WHITE, BROWN, TAN, GREY, connect, get_movable_joints, set_joint_position, set_pose, add_fixed_constraint, remove_fixed_constraint, set_velocity, get_joint_positions, get_pose, enable_gravity

from gym_stacking.env import StackEnv, get_env_state
from tqdm import tqdm


def execute(samples, env, idx=0):
    # postprocess_samples = []

    states = [get_env_state(env.robot, env.cubes, env.attachments)]
    rewards = 0
    ims = []

    for sample in samples[1:]:
        joint_pos = sample[:7]
        contact = [sample[14+j*8] for j in range(4)]
        action = np.concatenate([joint_pos, contact], dtype=np.float32)

        state, reward, done, _ = env.step(action)
        if not env.use_gui:
            im = env.render()
            ims.append(im)
            # writer.append_data(im)

        states.append(get_env_state(env.robot, env.cubes, env.attachments))
        rewards = rewards + reward

    env.attachments[:] = 0
    env.get_state()  # env.get_state is also set_state
    reward = env.compute_reward()
    rewards = rewards + reward
    state = get_env_state(env.robot, env.cubes, env.attachments)

    # writer.close()

    return state, states, ims, rewards


def eval_episode(guide, env, dataset, idx=0, args=None):
    state = env.reset()
    # states = [state]

    # idxs = [(0, 3), (1, 0), (2, 1)]
    # cond_idxs = [map_tuple[idx] for idx in idxs]
    # stack_idxs = [idx[0] for idx in idxs]
    # place_idxs = [idx[1] for idx in idxs]

    # samples_full_list = []
    obs_dim = dataset.obs_dim

    samples = torch.Tensor(state)
    samples = (samples - dataset.mins) / (dataset.maxs - dataset.mins + 1e-8)
    samples = samples[None, None, None].cuda()
    samples = (samples - 0.5) * 2

    conditions = [
           (0, obs_dim, samples),
    ]

    rewards = 0
    frames = []

    total_samples = []

    for i in range(3):
        # samples = samples_orig = trainer.ema_model.guided_conditional_sample(model, 1, conditions, cond_idxs[i], stack_idxs[i], place_idxs[i])
        samples = samples_orig = trainer.ema_model.conditional_sample(1, conditions)

        samples = torch.clamp(samples, -1, 1)
        samples_unscale = (samples + 1) * 0.5
        samples = dataset.unnormalize(samples_unscale)

        samples = to_np(samples.squeeze(0).squeeze(0))

        samples, samples_list, frames_new, reward = execute(samples, env, idx=i)
        frames.extend(frames_new)

        total_samples.extend(samples_list)

        # samples_full_list.extend(samples_list)

        samples = (samples - dataset.mins) / (dataset.maxs - dataset.mins + 1e-8)
        samples = torch.Tensor(samples[None, None, None]).to(samples_orig.device)
        samples = (samples - 0.5) * 2


        conditions = [
               (0, obs_dim, samples),
        ]

        samples_list.append(samples)

        rewards = rewards + reward

        print("reward: %.4f   " % reward)

    if args is not None:
        save_dir = os.path.join(args.savepath, "uncond_samples")
    else:
        save_dir = "uncond_samples"

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    if 'DISPLAY' not in os.environ:
        writer = get_writer(os.path.join(save_dir, "uncond_video_writer{}.mp4".format(idx)))
        for frame in frames:
            writer.append_data(frame)

    np.save(os.path.join(save_dir, "uncond_sample_{}.npy".format(idx)), np.array(total_samples))


    return rewards


class PosGuide(nn.Module):
    def __init__(self, cube, cube_other):
        super().__init__()
        self.cube = cube
        self.cube_other = cube_other

    def forward(self, x, t):
        cube_one = x[..., 64:, 7+self.cube*8: 7+self.cube*8]
        cube_two = x[..., 64:, 7+self.cube_other*8:7+self.cube_other*8]

        pred = -100 * torch.pow(cube_one - cube_two, 2).sum(dim=-1)
        return pred


def to_np(x):
    return x.detach().cpu().numpy()

# def pad_obs(obs, val=0):
#     state = np.concatenate([np.ones(1)*val, obs])
#     return state
#
# def set_obs(env, obs):
#     state = pad_obs(obs)
#     qpos_dim = env.sim.data.qpos.size
#     env.set_state(state[:qpos_dim], state[qpos_dim:])

parser = argparse.ArgumentParser()
parser.add_argument('--suffix', default='0', type=str, help='save dir suffix')
parser.add_argument('--env_name', default="multiple_cube_kuka_temporal_convnew_real2_128", type=str, help='env name')
parser.add_argument('--data_path', default="kuka_dataset", type=str, help='dataset root path')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

#### dataset
env_name = args.env_name
H = 128
T = 1000
dataset = KukaDataset(H, data_path=args.data_path)

diffusion_path = f'logs/{env_name}/'
diffusion_epoch = 650

weighted = 5.0

savepath = f'logs/{env_name}/plans_weighted{weighted}_{H}_{T}/{args.suffix}'
utils.mkdir(savepath)

args.savepath = savepath

## dimensions
obs_dim = dataset.obs_dim
# act_dim = 0

#### model
# model = MixerUnet(
#     dim = 32,
#     image_size = (H, obs_dim),
#     dim_mults = (1, 2, 4, 8),
#     channels = 2,
#     out_dim = 1,
# ).cuda()

# model = MixerUnetNew(
#     H,
#     obs_dim * 2,
#     0,
#     dim = 32,
#     dim_mults = (1, 2, 4, 8),
# #     out_dim = 1,
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
    channels = 2,
    image_size = (H, obs_dim),
    timesteps = T,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()

#### load reward and value functions
# reward_model, *_ = utils.load_model(reward_path, reward_epoch)
# value_model, *_ = utils.load_model(value_path, value_epoch)
# value_guide = guides.ValueGuide(reward_model, value_model, discount)
env = StackEnv(conditional=False)

trainer = Trainer(
    diffusion,
    dataset,
    env,
    train_batch_size = 32,
    train_lr = 2e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    fp16 = False,                     # turn on mixed precision training with apex
    results_folder = diffusion_path,
)


print(f'Loading: {diffusion_epoch}')
trainer.load(diffusion_epoch)
render_kwargs = {
    'trackbodyid': 2,
    'distance': 10,
    'lookat': [10, 2, 0.5],
    'elevation': 0
}

x = dataset[0][0].view(1, 1, H, obs_dim).cuda()
conditions = [
       (0, obs_dim, x[:, :, :1]),
]
trainer.ema_model.eval()
hidden_dims = [128, 128, 128]


config = Config(
    model_class=TimeConditionedMLP,
    time_dim=128,
    input_dim=obs_dim,
    hidden_dims=hidden_dims,
    output_dim=12,
    savepath=savepath,
)

device = torch.device('cuda')
guide = config.make()
guide.to(device)

guide_model_ckpt = "./logs/kuka_cube_stack_classifier_new3/value_0.99/state_80.pt"
ckpt = torch.load(guide_model_ckpt)

guide.load_state_dict(ckpt)

# samples_list = []
# frames = []

# models = [PosGuide(1, 3), PosGuide(1, 4), PosGuide(1, 2)]

#####################################################################
# TODO: Color
# Red = block 0
# Green = block 1
# Blue = block 2
# Yellow block 3
#####################################################################

rewards =  []

for i in tqdm(range(100)):
    reward = eval_episode(guide, env, dataset, idx=i, args=args)
    # assert False
    rewards.append(reward)
    print("rewards mean: ", np.mean(rewards))
    print("rewards std: ", np.std(rewards) / len(rewards) ** 0.5)
