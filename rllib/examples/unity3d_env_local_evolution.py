"""
Example of running an RLlib Trainer against a locally running Unity3D editor
instance (available as Unity3DEnv inside RLlib).
For a distributed cloud setup example with Unity,
see `examples/serving/unity3d_[server|client].py`

To run this script against a local Unity3D engine:
1) Install Unity3D and `pip install mlagents`.

2) Open the Unity3D Editor and load an example scene from the following
   ml-agents pip package location:
   `.../ml-agents/Project/Assets/ML-Agents/Examples/`
   This script supports the `3DBall`, `3DBallHard`, `SoccerStrikersVsGoalie`,
    `Tennis`, and `Walker` examples.
   Specify the game you chose on your command line via e.g. `--env 3DBall`.
   Feel free to add more supported examples here.

3) Then run this script (you will have to press Play in your Unity editor
   at some point to start the game and the learning process):
$ python unity3d_env_local.py --env 3DBall --stop-reward [..]
  [--framework=torch]?
"""

import argparse
import os
from pickletools import read_uint1
import re
from select import epoll
from prometheus_client import MetricsHandler

import ray
from ray import tune
from ray.rllib.env.wrappers.unity3d_env import Unity3DEnv
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.agents.ppo.ppo import PPOTrainer
import logging
from typing import List, Optional, Type, Union, Callable
from gym.spaces import Box, MultiDiscrete, Tuple as TupleSpace
from ray.rllib.policy.policy import PolicySpec

from ray.util.debug import log_once
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.agents.trainer import Trainer
from python.ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.agents.trainer_config import TrainerConfig
from ray.rllib.evaluation.metrics import (
    collect_episodes,
    collect_metrics,
    summarize_episodes,
)
from ray.rllib.execution.rollout_ops import (
    standardize_fields,
)
from ray.rllib.execution.train_ops import (
    train_one_step,
    multi_gpu_train_one_step,
)
from ray.rllib.utils.annotations import ExperimentalAPI
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.metrics.learner_info import LEARNER_INFO, LEARNER_STATS_KEY
from ray.rllib.utils.typing import TrainerConfigDict, ResultDict
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    WORKER_UPDATE_TIMER,
)
import numpy as np
import logging

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    type=str,
    default="XlandFindObject",
    choices=[
        "MazeFood",
        "XlandFindObject",
    ],
    help="The name of the Env to run in the Unity3D editor: `MazeFood | XlandFindObject `",
)
parser.add_argument(
    "--file-name",
    type=str,
    default=None,
    help="The Unity3d binary (compiled) game, e.g. "
    "'/home/ubuntu/soccer_strikers_vs_goalie_linux.x86_64'. Use `None` for "
    "a currently running Unity3D editor.",
)
parser.add_argument(
    "--from-checkpoint",
    type=str,
    default=None,
    help="Full path to a checkpoint file for restoring a previously saved "
    "Trainer state.",
)
parser.add_argument("--num-workers", type=int, default=0)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=9999, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=10000000, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward",
    type=float,
    default=9999.0,
    help="Reward at which we stop training.",
)
parser.add_argument(
    "--horizon",
    type=int,
    default=3000,
    help="The max. number of `step()`s for any episode (per agent) before "
    "it'll be reset again automatically.",
)
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="tf",
    help="The DL framework specifier.",
)


# Add WFC and gRPC support for unity3D RLLib enviroment
# TODO: finish the implementation and UnitTest
class WFCUnity3DEnv(Unity3DEnv):
    def __init__(self,
                 file_name: str = None,
                 port: Optional[int] = None,
                 seed: int = 0,
                 no_graphics: bool = False,
                 timeout_wait: int = 300,
                 episode_horizon: int = 1000):
        super().__init__(file_name, port, seed, no_graphics, timeout_wait, episode_horizon)


        self.height_map = None
        if config['map_image'] is not None and config['map_image'] != self.height_map:
            self.height_map = config['map_image']
            self.render_new_map(self.height_map)
        pass

    @staticmethod
    def get_policy_configs_for_game(game_name: str):
        obs_spaces = {
                 "MazeFood": Box(float("-inf"), float("inf"), (84, 84, 3)),
                "XlandFindObject": Box(float("-inf"), float("inf"), (84, 84, 3))
            }
        action_spaces = {
            # MazeFood Continous.
            "MazeFood": Box(
                float("-inf"), float("inf"), (2, ), dtype=np.float32),
            # XlandFindObject.
            "XlandFindObject": MultiDiscrete([9]),
        }
        policies = {
        game_name: PolicySpec(
            observation_space=obs_spaces[game_name],
            action_space=action_spaces[game_name]),
        }

        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            return game_name
        return policies, policy_mapping_fn
            


    def render_new_map(self, mapimg):
        with self.get_grpc_channel() as channel :
            unity_grpc = grpc.connect(channel)
            return unity_grpc.render_new_map(mapimg) 
    
    # conditoinal WFC mutation
    def generate_new_height_map(self, mapimg=None):
        if not mapimg:
            mapimg = self.height_map
        pass
    
    def get_height_map(self):
        pass

    def save_height_map(self):
        pass


def env_creator(config):
    rllib_env = WFCUnity3DEnv(
            file_name=config["file_name"],
            no_graphics=(args.env != "VisualHallway" and config["file_name"] is not None),
            episode_horizon=config["episode_horizon"],
        )
    return rllib_env


# TODO: Implement this Trainer and pass UnitTest
class Unity3DTrainer(PPOTrainer):
    @override(Trainer)
    def evaluate(
            self,
            episodes_left_fn=None,  # deprecated
            duration_fn: Optional[Callable[[int], int]] = None,
        ) -> dict:
        evaluation_metrics = super().evaluate(episodes_left_fn, duration_fn)
        # custom hook api
        self._after_evaluate(evaluation_metrics)
        return evaluation_metrics
    
    def _after_evaluate(self, evaluation_metrics):
        logger.debug(f"Initial Evaluation metrics: {evaluation_metrics}")
        """After-evaluation callback."""
        def should_continue(results):
            eposides_reward = np.array(results['evaluation']['hist_stats']['episode_reward'])
            nonzero_episodes = (eposides_reward > 0).sum()
            # Map is too easy since there are too many succussful episodes 
            return nonzero_episodes >= (len(eposides_reward) // 2)
        metric = evaluation_metrics
        count = 0
        config = self.config.copy()
        # is Time to evolute a new map
        while should_continue(metric):
            count += 1
            logger.info(f"Start Evolving map for the {count} time")
            new_env_map = self.workers.local_worker().env.generate_new_height_map()
            config['map_img'] = new_env_map
            logger.info(f"Evaluating the evolved map now")
            # Create a rollout worker and using it to collect experiences.
            evolute_worker = WorkerSet(
                env_creator=lambda c: env_creator(c),
                trainer_config=config,
                policy_class=self.get_default_policy_class(config),
                num_workers=0,
                local_worker=True
            )
            evolute_worker.sync_weights(
                from_worker=self.workers.local_worker()
            )
            self._sync_filters_if_needed(evolute_worker)
            metric = {"evaluation": collect_metrics(evolute_worker)}
            logger.debug(f"Map Evaluation metrics: {metric}")
            logger.info(f"{count} time Map Evaluation Done")
        logger.info("Got a New Map!")
        logger.debug(f"New Map Evaluation metrics: {metric}")
        self.config['map_img'] = new_env_map
        # Rendering the new map for next training_iteration
        def fn(env, env_context):
            env.render_new_map(env_context['map_img'])
        self.workers.foreach_env_with_context(fn)
        self.evaluation_workers.foreach_env_with_context(fn)


if __name__ == "__main__":
    ray.init()

    args = parser.parse_args()

    tune.register_env(
        "unity3d",
        lambda config: env_creator(config), 
    )

    # Get policies (different agent types; "behaviors" in MLAgents) and
    # the mappings from individual agents to Policies.
    policies, policy_mapping_fn = Unity3DEnv.get_policy_configs_for_game(args.env)

    config = {
        "env": "unity3d",
        "env_config": {
            "file_name": args.file_name,
            "episode_horizon": args.horizon,
        },
        # For running in editor, force to use just one Worker (we only have
        # one Unity running)!
        "num_workers": args.num_workers if args.file_name else 0,
        # Other settings.
        "lr": 0.0003,
        "lambda": 0.95,
        "gamma": 0.99,
        "sgd_minibatch_size": 256,
        "train_batch_size": 4000,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_sgd_iter": 20,
        "rollout_fragment_length": 200,
        "clip_param": 0.2,
        # Multi-agent setup for the particular env.
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
        },
        "model": {
            "fcnet_hiddens": [512, 512],
        },
        "framework": "tf",
        "no_done_at_end": True,
    }

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }
    trainer = PPOTrainer(config=config, stop=stop)

    # Run the experiment.
    results = tune.run(
        "PPO",
        config=config,
        stop=stop,
        verbose=1,
        checkpoint_freq=5,
        checkpoint_at_end=True,
        restore=args.from_checkpoint,
    )

    # And check the results.
    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()
