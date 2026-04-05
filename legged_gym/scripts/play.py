# -*- coding: utf-8 -*-
"""
play.py — 在 Isaac Gym 中加载已训练策略并可视化 / 导出

整体流程（与 train 的对应关系）：
1. 解析命令行参数（任务名、checkpoint、设备等），与 train.py 共用 get_args()。
2. 读取该任务在 task_registry 中注册的环境配置 env_cfg 与训练配置 train_cfg。
3. 将 env_cfg 改成「测试模式」：更少并行环境、小地形、关闭噪声与域随机化等，便于稳定观看策略效果。
4. make_env 创建仿真环境；make_alg_runner 在 train_cfg.runner.resume=True 时从 logs 目录加载权重。
5. 用 get_inference_policy 得到仅前向推理的 policy（不更新梯度）。
6. 可选：将 Actor 导出为 TorchScript（.pt），供 MuJoCo / C++ 等部署使用。
7. 循环 env.step，在窗口中回放策略（非 headless 时）。

常用命令示例：
    python legged_gym/scripts/play.py --task=g1
    python legged_gym/scripts/play.py --task=h1 --load_run=-1 --checkpoint=-1
    python legged_gym/scripts/play.py --task=go2 --experiment_name=my_exp --headless

checkpoint 加载逻辑在 task_registry.make_alg_runner 中：resume_path 由 get_load_path(
    logs/<experiment_name>, load_run, checkpoint) 决定；load_run/checkpoint 为 -1 时通常表示「最近一次 run / 最新 checkpoint」。
"""

import os

# 必须先导入 isaacgym，再创建基于 Isaac Gym 的环境（官方示例惯例，保证底层库初始化顺序）
import isaacgym

from legged_gym import LEGGED_GYM_ROOT_DIR
# 导入 envs 会执行 legged_gym/envs/__init__.py，向 task_registry 注册 go2/g1/h1/h1_2 等任务
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry

# 模块级开关：直接 import 本文件时 play() 也能访问；运行脚本前可在此修改
EXPORT_POLICY = True
# 以下两项在官方 legged_gym 中可能用于录屏或相机跟随，本仓库当前 play.py 未使用，保留便于对齐或后续扩展
RECORD_FRAMES = False
MOVE_CAMERA = False


def play(args):
    """
    加载指定任务的策略并在仿真中运行若干步。

    参数:
        args: gymutil / get_args 解析得到的命名空间，至少包含 task、sim_device、rl_device、headless 等。
    """
    # 从全局注册表取出该任务默认的「环境配置」与「PPO/Runner 配置」（类实例，可直接改字段）
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # ---------- 以下为「测试 / 演示」专用覆盖，避免与训练时的大规模并行、课程学习、随机化混在一起 ----------
    # 并行环境数上限 100：训练可能上千，回放时减少 GPU 占用与画面复杂度
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    # 地形缩小为 5x5 块，加载快、视野简单
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    # 关闭地形课程：不随训练进度切换难度，固定为当前设定地形
    env_cfg.terrain.curriculum = False
    # 关闭观测噪声：看到「干净」策略效果，便于判断是否是噪声导致的发抖
    env_cfg.noise.add_noise = False
    # 关闭摩擦系数随机：测试时地面特性固定
    env_cfg.domain_rand.randomize_friction = False
    # 关闭随机推机器人：避免演示时被外力干扰
    env_cfg.domain_rand.push_robots = False

    # 环境内部可据此走测试分支（例如关闭仅训练需要的逻辑）；具体行为见各 *env.py / legged_robot.py
    env_cfg.env.test = True

    # 使用（可能已被命令行参数进一步覆盖的）env_cfg 创建 Isaac Gym 矢量环境
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # 初始观测，形状与训练时一致，供策略第一步输入
    obs = env.get_observations()

    # 要求 OnPolicyRunner 从磁盘恢复：make_alg_runner 会调用 get_load_path + runner.load(resume_path)
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    # 包装为推理模式策略（eval、不构建计算图等，具体见 rsl_rl OnPolicyRunner）
    policy = ppo_runner.get_inference_policy(device=env.device)

    # ---------- 导出 TorchScript：MLP → policy_1.pt；带 LSTM 的记忆网络 → policy_lstm_1.pt（见 helpers.export_policy_as_jit）----------
    # 读取模块级 EXPORT_POLICY；可在本文件顶部改为 False 以跳过导出、加快纯回放
    if EXPORT_POLICY:
        path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            train_cfg.runner.experiment_name,
            "exported",
            "policies",
        )
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print("Exported policy as jit script to: ", path)

    # 运行 10 倍单回合最大步数：足够长以便多次 reset 后仍能看到持续行为；具体 max_episode_length 由环境配置决定
    for i in range(10 * int(env.max_episode_length)):
        # detach：切断与旧计算图的联系，避免意外保留梯度；策略与环境各自在对应 device 上
        actions = policy(obs.detach())
        # 返回值与 rsl_rl VecEnv 约定一致：下一观测、特权观测(此处不用)、奖励、是否终止、info
        obs, _, rews, dones, infos = env.step(actions.detach())


if __name__ == "__main__":
    args = get_args()
    play(args)
