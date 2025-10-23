import json
import os
import numpy as np
from datetime import datetime

class DetailedLogger:
    def __init__(self, log_dir="results/detailed_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.training_log = {
            'timestamp': [],
            't_env': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'reward_mean': [],
        }
        
        self.test_episodes = []
    
    def log_training_step(self, t_env, metrics):
        """学習ステップのメトリクスを記録"""
        self.training_log['timestamp'].append(datetime.now().isoformat())
        self.training_log['t_env'].append(t_env)
        
        for key, value in metrics.items():
            if key not in self.training_log:
                self.training_log[key] = []
            self.training_log[key].append(float(value))
    
    def log_test_episode(self, episode_num, episode_data):
        """テストエピソードを記録"""
        episode_info = {
            'episode_num': episode_num,
            'timestamp': datetime.now().isoformat(),
            'observations': episode_data['observations'],
            'actions': episode_data['actions'],
            'rewards': episode_data['rewards'],
            'states': episode_data['states'],
            'total_reward': sum(episode_data['rewards']),
            'episode_length': len(episode_data['rewards']),
        }
        self.test_episodes.append(episode_info)
    
    def save(self):
        """ログをファイルに保存"""
        # 学習ログの保存
        training_path = os.path.join(self.log_dir, 'training_log.json')
        with open(training_path, 'w') as f:
            json.dump(self.training_log, f, indent=2)
        
        # テストエピソードの保存
        test_path = os.path.join(self.log_dir, 'test_episodes.json')
        with open(test_path, 'w') as f:
            json.dump(self.test_episodes, f, indent=2)
        
        # CSV形式でも保存（分析用）
        import pandas as pd
        df = pd.DataFrame(self.training_log)
        df.to_csv(os.path.join(self.log_dir, 'training_log.csv'), index=False)
    
    def plot_training_curves(self):
        """学習曲線をプロット"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        metrics = ['policy_loss', 'value_loss', 'entropy', 'reward_mean']
        for ax, metric in zip(axes.flat, metrics):
            if metric in self.training_log and len(self.training_log[metric]) > 0:
                ax.plot(self.training_log['t_env'], self.training_log[metric])
                ax.set_xlabel('Environment Steps')
                ax.set_ylabel(metric)
                ax.set_title(f'{metric} over time')
                ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_curves.png'))
        plt.close()