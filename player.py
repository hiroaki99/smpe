import time
import gym
from envs.__init__ import _GymmaWrapper  # 既存のラッパを流用

def main():
    # 学習と同条件で環境を作る（seedは任意）
    env = _GymmaWrapper(key="mpe:SimpleSpread-v0", time_limit=25, pretrained_wrapper=None, seed=0)
    obs, state = env.reset()

    for t in range(25):  # 1エピソード分
        # ランダム行動（挙動確認用。学習済みポリシーを使うならここを差し替え）
        actions = [env.longest_action_space.sample() for _ in range(env.n_agents)]
        r, done, _ = env.step(actions)
        env.render()            # ここでウィンドウ表示
        time.sleep(0.03)
        if done:
            break

    env.close()

if __name__ == "__main__":
    main()
