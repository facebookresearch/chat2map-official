# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

r"""
This file hosts task-specific or trainer-specific environments for trainers.
All environments here should be a (direct or indirect ) subclass of Env class
in habitat. Customized environments should be registered using
``@baseline_registry.register_env(name="myEnv")` for reusability
"""

from typing import Optional, Type
import logging

import habitat
from habitat import Config, Dataset
from chat2map.common.baseline_registry import baseline_registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions


def get_env_class(env_name: str) -> Type[habitat.RLEnv]:
    """
    Return environment class based on name.
    :param env_name: name of the environment.
    :return: Type[habitat.RLEnv]: env class.
    """

    return baseline_registry.get_env(env_name)


@baseline_registry.register_env(name="DummyHabitatEnv")
class DummyHabitatEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        """
        Create environment
        :param config: config to create environment
        :param dataset: dataset for environment
        """

        self._rl_config = config.RL
        self._config = config
        self._core_env_config = config.TASK_CONFIG

        super().__init__(self._core_env_config, dataset)

    def reset(self):
        """
        Reset environment
        :return: observation after reset
        """

        self._env_step = 0
        observation = super().reset()
        logging.debug(super().current_episode)
        return observation

    def step(self, *args, **kwargs):
        """
        Take step in enviroment
        :param args: args for taking step
        :param kwargs: keyword args for taking step
        :return: tuple containing new observation after taking step, reward from this step, done flag and episode info
                after step
        """

        observation, reward, done, info = super().step(*args, **kwargs)
        self._env_step += 1
        return observation, reward, done, info

    def get_reward_range(self):
        """
        get reward range for task
        :return: reward range
        """

        return (
            0,
            float('inf'),
        )   

    def get_reward(self, observations):
        """
        get reward for task
        :param observations: observations for computing the reward
        :return: computed reward
        """

        return 0

    def _distance_target(self):
        """
        get distance to target
        :return: distance to target
        """

        return float('inf')

    def _episode_success(self):
        """
        get flag saying if episode successful or not
        :return: flag saying if episode successful or not
        """

        return False

    def _goal_reached(self):
        """
        get flag saying if goal reached or not
        :return: flag saying if goal reached or not
        """

        return False

    def get_done(self, observations):
        """
        get flag saying if episode done or not
        :param observations: observations for deciding if episode done or not
        :return: flag saying if episode done or not
        """

        done = False
        if self._env.episode_over:
            done = True
        return done

    def get_info(self, observations):
        """
        get episode info
        :param observations: observations
        :return: episode info
        """

        return self.habitat_env.get_metrics()

    # for data collection
    def get_current_episode_id(self):
        """
        get current episode ID
        :return: current episode ID
        """

        return self.habitat_env.current_episode.episode_id
