"""
train_parallel.py
Train multiple agents in parallel using tmux
"""

import subprocess as sp
from classopt import classopt, config as classopt_config
from typing import Optional
import random


class TmuxSession:
    def __init__(self, session_name: str, n_of_windows: int = 1, attach: bool = False):
        self.session_name = session_name
        if not attach:
            sp.run(f'tmux kill-session -t {session_name}'.split(), stderr=sp.DEVNULL)
            sp.run(f'tmux new-session -d -s {session_name}'.split())

        self.windows = [
            TmuxWindow(self.session_name, window=0)
        ]
        self.new_window(n_of_windows - 1, attach=attach)

    def new_window(self, n_of_windows: int, attach: bool = False):
        for _ in range(n_of_windows):
            if not attach:
                sp.run(f'tmux new-window -t {self.session_name}'.split())
            self.windows.append(TmuxWindow(self.session_name, len(self.windows)))

    def get_window(self, window: int):
        assert 0 <= window < len(self.windows)
        return TmuxWindow(self.session_name, window)


class TmuxWindow:
    def __init__(self, session_name: str, window: int):
        self.session_name = session_name
        self.window = window

    def send_cmd(self, cmd: str):
        print(f'[{self.session_name}:{self.window}] Sending command: {cmd}')
        sp.run(f'tmux send-keys -t {self.session_name}:{self.window}'.split() + [cmd, 'C-m'])


# ==================== #


@classopt(default_long=True)
class Config:
    kill: bool = classopt_config(default=False, help='kill all tmux sessions')
    session_name: str = 'drl-train'


def get_train_cmd(cuda_device: int,
                  eta: float,
                  penalty: float,
                  curiosity: bool,
                  batch_size: int = 32,
                  reward_scaling: bool = False,
                  name: Optional[str] = None,
                  group: Optional[str] = None):

    if name is None:
        name = f'eta{eta}'
        if abs(penalty) < 1:
            name += '-wo-pen'
        else:
            name += f'-pen{-penalty}'
        name = name.replace('.0', '').replace('.', '_')

    seed = random.randint(1000, 9999)
    name += f'-seed{seed}'

    cmd = f'CUDA_VISIBLE_DEVICES={cuda_device}'
    cmd += f' python -m src --train --name {name}'
    cmd += f' --icm_eta {eta}'
    cmd += f' --truncated_negative_reward {penalty}'
    cmd += f' --batch_size {batch_size}'
    cmd += f' --seed {seed}'
    if not curiosity:
        cmd += ' --disable_curiosity'
    if reward_scaling:
        cmd += ' --reward_scaling'
    if group is not None:
        cmd += f' --group {group}'
    return cmd


cuda_device_id = 0
def get_device_id():
    global cuda_device_id
    cuda_device_id += 1
    import torch
    number_of_devices = torch.cuda.device_count()
    return cuda_device_id % number_of_devices


def train(config: Config):
    print(config)

    for i in range(5):
        cmds = [
            # get_train_cmd(cuda_device=get_device_id(), eta=32, penalty=0,
            #               curiosity=False, name=f'ppo-icm-{i}', group='ppo-only'),
            get_train_cmd(cuda_device=get_device_id(), eta=32, penalty=0,
                          curiosity=True, name=f'ppo-icm-{i}', group='ppo-icm-32'),
            get_train_cmd(cuda_device=get_device_id(), eta=48, penalty=0,
                          curiosity=True, name=f'ppo-icm-{i}', group='ppo-icm-48'),
            # get_train_cmd(cuda_device=get_device_id(), eta=64, penalty=0,
            #               curiosity=True, name=f'ppo-icm-{i}', group='ppo-icm-64'),
            # get_train_cmd(cuda_device=get_device_id(), eta=96, penalty=0,
            #               curiosity=True, name=f'ppo-icm-{i}', group='ppo-icm-96'),
        ]

        tmux = TmuxSession(session_name=f'{config.session_name}-{i}',
                           n_of_windows=len(cmds),
                           attach=config.kill)

        for window_id, cmd in enumerate(cmds):
            window = tmux.get_window(window_id)
            if config.kill:
                # Send Ctrl+C to kill the process
                window.send_cmd('C-c')
            else:
                window.send_cmd('cd ~/workspace/drl-lecture')
                window.send_cmd(cmd)


if __name__ == '__main__':
    config = Config.from_args()
    train(config)
