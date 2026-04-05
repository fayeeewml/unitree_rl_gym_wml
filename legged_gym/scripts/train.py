# -*- coding: utf-8 -*-
"""
train.py — 使用 Isaac Gym + rsl_rl 对 Unitree 腿式机器人任务做 on-policy 强化学习训练

流程概览：
1. get_args() 解析命令行（任务名、并行环境数、随机种子、是否 headless、是否从 checkpoint 继续训练等）。
2. task_registry.make_env() 根据注册名（go2 / g1 / h1 / h1_2）创建矢量环境：加载对应 env_cfg、应用命令行覆盖、设种子、构造仿真参数并实例化环境类。
3. task_registry.make_alg_runner() 创建 rsl_rl 的 OnPolicyRunner（通常为 PPO），并设置 TensorBoard 等日志目录；
   若配置里 runner.resume=True（常由命令行 --resume 触发），则会加载已有模型继续训。
4. ppo_runner.learn() 按 train_cfg.runner.max_iterations 执行多轮策略更新；init_at_random_ep_len=True 使每轮初始 episode 长度随机，利于探索。

常用命令示例：
    python legged_gym/scripts/train.py --task=g1
    python legged_gym/scripts/train.py --task=go2 --headless --num_envs=4096
    python legged_gym/scripts/train.py --task=h1 --resume --load_run=-1 --checkpoint=-1

日志默认在：legged_gym 包根目录下的 logs/<experiment_name>/<日期时间>_<run_name>/，
checkpoint 一般为 model_<iteration>.pt（具体以 rsl_rl 保存逻辑为准）。

与 play.py 的区别：train 使用各任务 config 里的完整训练设置（大地形、域随机化、噪声等），
play 会刻意缩小地形并关闭随机化以便稳定可视化；训练时不要求与 play 相同的覆盖逻辑。
"""

# 须先于环境创建导入，保证 Isaac Gym 与底层物理库初始化顺序正确
import isaacgym

# 执行 envs/__init__.py，向 task_registry 注册各机器人任务
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


def train(args):
    """
    创建环境并启动 PPO（OnPolicyRunner）训练循环。

    参数:
        args: 由 get_args() 解析得到，须包含 task，以及 sim_device / headless 等 Isaac Gym 所需字段。
    """
    # 使用注册表中的默认 env_cfg，并根据命令行（如 --num_envs、--seed）做 update_cfg_from_args 覆盖
    env, env_cfg = task_registry.make_env(name=args.task, args=args)

    # 创建算法运行器：默认 log_root 指向 LEGGED_GYM_ROOT_DIR/logs/<experiment_name>/...
    # 若 train_cfg.runner.resume 为 True，会在 make_alg_runner 内 load 指定 checkpoint
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)

    # num_learning_iterations：总训练迭代次数，来自各任务 *config 中 runner.max_iterations（可被命令行覆盖）
    # init_at_random_ep_len=True：每个 rollout 周期开始时随机化 episode 已进行步数，减轻各环境同步 reset 带来的相关性
    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    args = get_args()
    train(args)
