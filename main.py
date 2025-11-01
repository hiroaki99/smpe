import numpy as np
import os
import random
# import collections
from collections.abc import Mapping
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver, MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger, Logger
import yaml
import pickle
import re

from run import run

from gym.envs.registration import register

os.makedirs("results/sacred", exist_ok=True)
os.makedirs("results/models", exist_ok=True)
os.makedirs("results/tb_logs", exist_ok=True)

def _safe_for_dir(s: str) -> str:
    # Windows 禁則文字 <>:"/\|?* と先頭末尾の空白/ドットを避ける
    s = re.sub(r'[<>:"/\\|?*]+', '_', str(s))
    s = s.strip().strip('.')
    return s or "unnamed"


register(
    id="Foraging-8x8-5p-1f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 5,
        "max_player_level": 3,
        "field_size": (8, 8),
        "max_food": 1,
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-11x11-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (11, 11),
        "max_food": 2,
        "sight": 11,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-15x15-3p-4f-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (15, 15),
        "max_food": 4,
        "sight": 15,
        "max_episode_steps": 50,
        "force_coop": False,
    },
)

register(
    id="Foraging-15x15-3p-4f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (15, 15),
        "max_food": 4,
        "sight": 15,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-8s-25x25-8p-5f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 8,
        "max_player_level": 3,
        "field_size": (25, 25),
        "max_food": 5,
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-5s-25x25-8p-5f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 8,
        "max_player_level": 3,
        "field_size": (25, 25),
        "max_food": 5,
        "sight": 5,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-7s-50x50-8p-5f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 8,
        "max_player_level": 3,
        "field_size": (50, 50),
        "max_food": 5,
        "sight": 7,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-30x30-7p-4f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 7,
        "max_player_level": 3,
        "field_size": (30, 30),
        "max_food": 4,
        "sight": 30,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-7s-30x30-7p-5f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 7,
        "max_player_level": 3,
        "field_size": (30, 30),
        "max_food": 5,
        "sight": 7,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-7s-30x30-7p-4f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 7,
        "max_player_level": 3,
        "field_size": (30, 30),
        "max_food": 4,
        "sight": 7,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-4s-30x30-8p-5f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 8,
        "max_player_level": 3,
        "field_size": (30, 30),
        "max_food": 5,
        "sight": 4,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-7s-15x15-5p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 5,
        "max_player_level": 3,
        "field_size": (15, 15),
        "max_food": 3,
        "sight": 7,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-2s-11x11-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (11, 11),
        "max_food": 2,
        "sight": 2,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-4s-11x11-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (11, 11),
        "max_food": 2,
        "sight": 4,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-2s-9x9-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (9, 9),
        "max_food": 2,
        "sight": 2,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-9x9-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (9, 9),
        "max_food": 2,
        "sight": 9,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-2s-8x8-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (8, 8),
        "max_food": 2,
        "sight": 2,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-8x8-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (8, 8),
        "max_food": 2,
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-6x6-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (6, 6),
        "max_food": 2,
        "sight": 6,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-15x15-3p-5f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (15, 15),
        "max_food": 5,
        "sight": 15,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-15x15-3p-5f-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (15, 15),
        "max_food": 5,
        "sight": 15,
        "max_episode_steps": 50,
        "force_coop": False,
    },
)

register(
    id="Foraging-6x6-3p-1f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (6, 6),
        "max_food": 1,
        "sight": 6,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-7x7-3p-1f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (7, 7),
        "max_food": 1,
        "sight": 7,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-7x7-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (7, 7),
        "max_food": 2,
        "sight": 7,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-2s-7x7-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (7, 7),
        "max_food": 2,
        "sight": 2,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-8x8-4p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (8, 8),
        "max_food": 2,
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-8x8-4p-2f-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (8, 8),
        "max_food": 2,
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": False,
    },
)

register(
    id="Foraging-8x8-4p-1f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (8, 8),
        "max_food": 1,
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-5x5-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (5, 5),
        "max_food": 2,
        "sight": 5,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-5x5-3p-1f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (5, 5),
        "max_food": 1,
        "sight": 5,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-8x8-6p-1f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 6,
        "max_player_level": 3,
        "field_size": (8, 8),
        "max_food": 1,
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-15x15-3p-5f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (15, 15),
        "max_food": 5,
        "sight": 15,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-15x15-3p-4f-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (15, 15),
        "max_food": 4,
        "sight": 15,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-8x8-2p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 2,
        "max_player_level": 3,
        "field_size": (8, 8),
        "max_food": 2,
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)


register(
    id="Foraging-10x10-4p-1f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (10, 10),
        "max_food": 1,
        "sight": 10,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-7x7-4p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (7, 7),
        "max_food": 3,
        "sight": 7,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)


register(
    id="Foraging-9x9-4p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (9, 9),
        "max_food": 2,
        "sight": 9,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-7s-20x20-5p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 5,
        "max_player_level": 3,
        "field_size": (20, 20),
        "max_food": 3,
        "sight": 7,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-5s-20x20-5p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 5,
        "max_player_level": 3,
        "field_size": (20, 20),
        "max_food": 3,
        "sight": 5,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-2s-11x11-4p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (11, 11),
        "max_food": 3,
        "sight": 2,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-4s-11x11-4p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (11, 11),
        "max_food": 3,
        "sight": 4,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-13x13-4p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (11, 11),
        "max_food": 3,
        "sight": 13,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-11x11-4p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (11, 11),
        "max_food": 3,
        "sight": 11,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-9x9-4p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (9, 9),
        "max_food": 3,
        "sight": 9,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-5x5-4p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (5, 5),
        "max_food": 3,
        "sight": 5,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-10x10-3p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (10, 10),
        "max_food": 3,
        "sight": 10,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-2s-12x12-2p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 2,
        "max_player_level": 3,
        "field_size": (12, 12),
        "max_food": 2,
        "sight": 2,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-6s-12x12-2p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 2,
        "max_player_level": 3,
        "field_size": (12, 12),
        "max_food": 2,
        "sight": 6,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-12x12-2p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 2,
        "max_player_level": 3,
        "field_size": (12, 12),
        "max_food": 2,
        "sight": 12,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-8x8-2p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 2,
        "max_player_level": 3,
        "field_size": (8, 8),
        "max_food": 2,
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)


register(
    id="Foraging-8x8-3p-1f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (8, 8),
        "max_food": 1,
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-2s-5x5-3p-1f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (5, 5),
        "max_food": 1,
        "sight": 2,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

SETTINGS['CAPTURE_MODE'] = "no" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")
# results_path = "/home/ubuntu/data"

@ex.main
def my_main(_run, _config, _log):
    print("-----------my_main------------")
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    # config = dict(_config)
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]

    # run the framework
    run(_run, config, _log)


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                # config_dict = yaml.load(f)
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


if __name__ == '__main__':

    import torch.multiprocessing as mp
    mp.set_start_method('spawn')

    params = deepcopy(sys.argv[1:])
    th.set_num_threads(1)

    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            # config_dict = yaml.load(f)
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
            print("----------------success loading----------------")
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc).replace(":", "_")

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    try:
        map_name = config_dict["env_args"]["map_name"]
    except:
        map_name = config_dict["env_args"]["key"]   

    try:
        map_name_raw = config_dict["env_args"]["map_name"]
    except:
        map_name_raw = config_dict["env_args"]["key"]

    map_name = _safe_for_dir(map_name_raw) 
    
    # now add all the config to sacred
    # ex.add_config(config_dict)

    def _parse_updates(argv_tokens):
        """
       argv の中から:  with k1=v1 k2=v2 ... を拾って {k: v} の dict を返す
        """
        if "with" not in argv_tokens:
            return {}
        i = argv_tokens.index("with")
        updates = {}
        for tok in argv_tokens[i+1:]:
            if tok.startswith("-"):  # 以降に別オプションが来たら打ち切り
                break
            if "=" not in tok:
                continue
            k, v = tok.split("=", 1)
            # 値の型を素朴に推定（int/float/bool）
            vv = v.strip('"').strip("'")
            if vv.lower() in ("true", "false"):
                vv = vv.lower() == "true"
            else:
                try:
                    if "." in vv:
                        vv = float(vv)
                    else:
                        vv = int(vv)
                except ValueError:
                    pass
            updates[k] = vv
        return updates

    def _deep_set(d, dotted_key, value):
        """
        'env_args.time_limit' のようなキーを dict に反映
        """
        keys = dotted_key.split(".")
        cur = d
        for k in keys[:-1]:
            if k not in cur or not isinstance(cur[k], dict):
                cur[k] = {}
            cur = cur[k]
        cur[keys[-1]] = value

    # "with" 以降の上書きを config_dict に適用
    updates = _parse_updates(params)
    for k, v in updates.items():
        _deep_set(config_dict, k, v)

    # Sacred に最終 config を登録
    ex.add_config(config_dict)
    
    for param in params:
        if param.startswith("env_args.map_name"):
            map_name = param.split("=")[1]
        elif param.startswith("env_args.key"):
            map_name = param.split("=")[1].replace(":", "_")

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, f"sacred/{config_dict['name']}/{map_name}")

    # ex.observers.append(MongoObserver(db_name="marlbench")) #url='172.31.5.187:27017'))
    ex.observers.append(FileStorageObserver.create(file_obs_path))
    # ex.observers.append(MongoObserver())
    print("--------------------------------------")
    # try:
    #     ex.run_commandline(params)
    # except Exception as e:
    #     import traceback
    #     print("!!! Sacred run failed !!!", e)
    #     traceback.print_exc()
    #     raise
    ex.run()
    print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

