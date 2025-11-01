from functools import partial
import pretrained
from smac.env import MultiAgentEnv , StarCraft2Env
import sys
import os
import gym
from gym import ObservationWrapper, spaces
from gym.envs import registry as gym_registry
from gym.spaces import flatdim
import numpy as np
from gym.wrappers import TimeLimit as GymTimeLimit


class ResetCompatWrapper(gym.Wrapper):
    """reset() の戻り値を常に (obs, info) に正規化する互換ラッパ"""
    def reset(self, **kwargs):
        try:
            out = self.env.reset(**kwargs)   # 新API (obs, info) 期待
        except TypeError:
            # 旧APIで kwargs(特に seed) を受け付けない場合
            out = self.env.reset()

        # 形を正規化
        if isinstance(out, tuple):
            # (obs, info) ならそのまま / 3要素以上でも先頭を obs とみなす
            obs = out[0]
            info = out[1] if len(out) > 1 and isinstance(out[1], dict) else {}
        else:
            obs, info = out, {}

        return obs, info

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
# from .traffic_junction import Traffic_JunctionEnv
# REGISTRY["traffic_junction"] = partial(env_fn, env=Traffic_JunctionEnv)

if sys.platform == "linux":
    os.environ.setdefault(
        "SC2PATH", os.path.join(os.getcwd(), "3rdparty", "StarCraftII")
    )


class TimeLimit(GymTimeLimit):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        # if self.env.spec is not None:
        #     self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    # def step(self, action):
    #     assert (
    #         self._elapsed_steps is not None
    #     ), "Cannot call env.step() before calling reset()"
    #     observation, reward, done, info = self.env.step(action)
    #     self._elapsed_steps += 1
    #     if self._elapsed_steps >= self._max_episode_steps:
    #         info["TimeLimit.truncated"] = not all(done)
    #         done = len(observation) * [True]
    #     return observation, reward, done, info

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before reset()"
        out = self.env.step(action)

        # 内側が4 or 5 どちらでも吸収
        if isinstance(out, tuple) and len(out) == 5:
            observation, reward, terminated, truncated, info = out
        else:
            observation, reward, done, info = out
            # 旧APIの done を terminated に割当、truncated は既定 False
            terminated, truncated = done, False

        self._elapsed_steps += 1

        # 既定のタイムアウト処理
        if self._elapsed_steps >= self._max_episode_steps:
            # terminated が配列（マルチエージェント）でも動くように
            def _all(x):
                try:
                    return all(x)
                except TypeError:
                    return bool(x)
            info["TimeLimit.truncated"] = not _all(terminated)
            truncated = True

        return observation, reward, terminated, truncated, info


class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation of individual agents."""

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)

        ma_spaces = []

        for sa_obs in env.observation_space:
            flatdim = spaces.flatdim(sa_obs)
            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim,),
                    dtype=np.float32,
                )
            ]

        self.observation_space = spaces.Tuple(tuple(ma_spaces))

    def observation(self, observation):
        return tuple(
            [
                spaces.flatten(obs_space, obs)
                for obs_space, obs in zip(self.env.observation_space, observation)
            ]
        )


class _GymmaWrapper(MultiAgentEnv):
    def __init__(self, key, time_limit, pretrained_wrapper, **kwargs):
        # self.episode_limit = time_limit
        # self._env = TimeLimit(gym.make(f"{key}"), max_episode_steps=time_limit)
        # self._env = FlattenObservation(self._env)

        # if pretrained_wrapper:
        #     self._env = getattr(pretrained, pretrained_wrapper)(self._env)

        # self.n_agents = self._env.n_agents
        # self._obs = None

        # self.longest_action_space = max(self._env.action_space, key=lambda x: x.n)
        # self.longest_observation_space = max(
        #     self._env.observation_space, key=lambda x: x.shape
        # )

        # self._seed = kwargs["seed"]
        # self._env.seed(self._seed)

        self.episode_limit = time_limit
        # self._env = gym.make(f"{key}", disable_env_checker=True)
        # self._env = TimeLimit(self._env, max_episode_steps=time_limit)
        # self._env = FlattenObservation(self._env)

        base_env = gym.make(f"{key}")
        base_env = ResetCompatWrapper(base_env)                  # ← 追加（最初に噛ませる）
        self._env = TimeLimit(base_env, max_episode_steps=time_limit)
        self._env = FlattenObservation(self._env)

        if pretrained_wrapper:
            self._env = getattr(pretrained, pretrained_wrapper)(self._env)

        self.n_agents = self._env.n_agents
        self._obs = None

        self.longest_action_space = max(self._env.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(
            self._env.observation_space, key=lambda x: x.shape
        )

        # self._seed = kwargs.get("seed", None)
        self._seed = kwargs["seed"]

    # def step(self, actions):
    #     """ Returns reward, terminated, info """
    #     actions = [int(a) for a in actions]
    #     self._obs, reward, done, info = self._env.step(actions)
    #     self._obs = [
    #         np.pad(
    #             o,
    #             (0, self.longest_observation_space.shape[0] - len(o)),
    #             "constant",
    #             constant_values=0,
    #         )
    #         for o in self._obs
    #     ]

        # return float(sum(reward)), all(done), {}

    def step(self, actions):
        actions = [int(a) for a in actions]
        out = self._env.step(actions)

        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
            # terminated / truncated が配列でも True/False 配列として扱う
            if not isinstance(terminated, (list, tuple, np.ndarray)):
                terminated = [bool(terminated)] * self.n_agents
            if not isinstance(truncated, (list, tuple, np.ndarray)):
                truncated = [bool(truncated)] * self.n_agents
            done = [bool(t) or bool(u) for t, u in zip(terminated, truncated)]
        else:
            obs, reward, done, info = out

        # 観測のパディング
        self._obs = [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in obs
        ]

        # 既存フレームワークに合わせて：総報酬, すべて終了?, info（空）を返す
        return float(sum(reward)), all(done), {}

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self._obs

    # def get_obs_agent(self, agent_id):
    #     """ Returns observation for agent_id """
    #     raise self._obs[agent_id]

    # def get_obs_size(self):
    #     """ Returns the shape of the observation """
    #     return flatdim(self.longest_observation_space)

    # def get_state(self):
    #     return np.concatenate(self._obs, axis=0).astype(np.float32)

    # def get_state_size(self):
    #     """ Returns the shape of the state"""
    #     return self.n_agents * flatdim(self.longest_observation_space)

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self._obs[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return flatdim(self.longest_observation_space)

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.n_agents * flatdim(self.longest_observation_space)

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        valid = flatdim(self._env.action_space[agent_id]) * [1]
        invalid = [0] * (self.longest_action_space.n - len(valid))
        return valid + invalid

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return flatdim(self.longest_action_space)

    # def reset(self):
    #     """ Returns initial observations and states"""
        
    #     # 1. 環境をリセットし、obsとinfoを安全に受け取る（1回のみ実行）
    #     reset_output = self._env.reset()
    #     if isinstance(reset_output, tuple) and len(reset_output) == 2:
    #          self._obs, info = reset_output
    #     else:
    #          self._obs = reset_output
    #          info = {} # infoを空の辞書として定義
        
    #     # 2. 観測のパディング処理を適用 (自己完結させる)
    #     self._obs = [
    #         np.pad(
    #             o,
    #             (0, self.longest_observation_space.shape[0] - len(o)),
    #             "constant",
    #             constant_values=0,
    #         )
    #         for o in self._obs
    #     ]

    #     return self.get_obs(), self.get_state()

    def reset(self):
        # Gym>=0.26: reset(seed=...), 旧Gym: seedを kwargs で受けない → 互換ラッパが吸収
        try:
            obs, _info = self._env.reset(seed=self._seed)
        except TypeError:
            # 旧APIでも ResetCompatWrapper が (obs, info) で返す
            obs, _info = self._env.reset()

        # パディング
        self._obs = [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in obs
        ]
        # episode_runner は戻り値を使わないので return 不要


    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    # def seed(self):
    #     return self._env.seed
    def seed(self, seed=None):
        self._seed = seed

    def save_replay(self):
        pass

    def get_stats(self):
        return {}


REGISTRY["gymma"] = partial(env_fn, env=_GymmaWrapper)
