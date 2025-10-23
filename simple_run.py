# simple_run.py
import torch as th
import numpy as np
import random
import os
from types import SimpleNamespace
import yaml

def main():
    # 設定の読み込み
    with open("config/default.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    with open("config/algs/smpe_mpe.yaml", "r") as f:
        alg_config = yaml.load(f, Loader=yaml.FullLoader)
    
    with open("config/envs/gymma.yaml", "r") as f:
        env_config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 設定をマージ
    config.update(alg_config)
    config.update(env_config)
    
    # 環境パラメータを設定
    config["env_args"]["key"] = "mpe:SimpleSpread-v0"
    config["env_args"]["time_limit"] = 25
    config["seed"] = 1
    config["t_max"] = 100000
    config["use_cuda"] = False
    config["device"] = "cpu"
    
    # 結果ディレクトリ
    config["local_results_path"] = "results/simple_run"
    os.makedirs(config["local_results_path"], exist_ok=True)
    
    args = SimpleNamespace(**config)
    
    # シードの設定
    np.random.seed(args.seed)
    random.seed(args.seed)
    th.manual_seed(args.seed)
    
    print("=" * 50)
    print("Starting Simple Run")
    print("=" * 50)
    
    # モジュールのインポート
    from runners import REGISTRY as runner_REGISTRY
    from controllers import REGISTRY as mac_REGISTRY
    from learners import REGISTRY as learner_REGISTRY
    from components.episode_buffer import ReplayBuffer
    from components.transforms import OneHot
    from utils.logging import get_logger
    from utils.logging import Logger
    
    # ロガーの設定
    console_logger = get_logger()
    logger = Logger(console_logger)
    
    # Runnerのセットアップ
    print("Setting up Runner...")
    runner = runner_REGISTRY[args.runner](args=args, logger=logger)
    
    # 環境情報を取得
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    
    print(f"Environment Info:")
    print(f"  n_agents: {args.n_agents}")
    print(f"  n_actions: {args.n_actions}")
    print(f"  state_shape: {args.state_shape}")
    
    # スキームのセットアップ
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}
    
    # リプレイバッファ
    buffer = ReplayBuffer(
        scheme, groups, args.buffer_size, 
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu"
    )
    
    # MAC
    print("Setting up MAC...")
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    
    # Learner
    print("Setting up Learner...")
    learner = learner_REGISTRY[args.learner](mac, buffer.scheme, logger, args)
    
    # 学習ループ
    print("=" * 50)
    print("Starting training...")
    print("=" * 50)
    
    episode = 0
    
    while runner.t_env <= args.t_max:
        
        # エピソード実行
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)
        
        # 学習
        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]
            
            learner.train(episode_sample, runner.t_env, episode)
        
        episode += args.batch_size_run
        
        # ログ出力
        if episode % 10 == 0:
            print(f"Episode: {episode}, t_env: {runner.t_env}/{args.t_max}")
        
        # テスト
        if runner.t_env % args.test_interval == 0:
            print(f"Running test at t_env={runner.t_env}")
            for _ in range(args.test_nepisode):
                runner.run(test_mode=True)
            logger.print_recent_stats()
    
    print("=" * 50)
    print("Training completed!")
    print("=" * 50)
    
    runner.close_env()

if __name__ == "__main__":
    main()