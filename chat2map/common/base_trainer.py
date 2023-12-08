# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from typing import ClassVar, Dict, List

import torch

from habitat import Config, logger
from chat2map.common.tensorboard_utils import TensorboardWriter
from chat2map.common.utils import poll_checkpoint_folder


class BaseTrainer:
    """Generic trainer class that serves as a base template for more
    specific trainer classes like RL trainer, SLAM or imitation learner.
    Includes only the most basic functionality.
    """

    supported_tasks: ClassVar[List[str]]

    def train(self) -> None:
        raise NotImplementedError

    def eval(self) -> None:
        raise NotImplementedError

    def save_checkpoint(self, file_name) -> None:
        raise NotImplementedError

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        raise NotImplementedError


class BaseRLTrainer(BaseTrainer):
    """Base trainer class for RL trainers. Future RL-specific
    methods should be hosted here.
    """

    device: torch.device
    config: Config
    video_option: List[str]
    _flush_secs: int

    def __init__(self, config: Config):
        super().__init__()
        assert config is not None, "needs config file to initialize trainer"
        self.config = config
        self._flush_secs = 30

    @property
    def flush_secs(self):
        return self._flush_secs

    @flush_secs.setter
    def flush_secs(self, value: int):
        self._flush_secs = value

    def train(self) -> None:
        raise NotImplementedError

    def eval(self, eval_interval=1, prev_ckpt_ind=-1, eval_wo_ckpt_path=False,) -> None:
        """
        Main method of trainer evaluation. Calls _eval_checkpoint() that
        is specified in Trainer class that inherits from BaseRLTrainer
        :param eval_interval: checkpoint index interval for evaluation
        :param prev_ckpt_ind: previous checkpoint index
        :param eval_wo_ckpt_path: eval without checkpoint path
        :return: None
        """

        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if "tensorboard" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.TENSORBOARD_DIR) > 0
            ), "Must specify a tensorboard directory for video display"
        if "disk" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.VIDEO_DIR) > 0
            ), "Must specify a directory for storing videos on disk"

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            if eval_wo_ckpt_path:
                self._eval_checkpoint(None, writer)
            else:
                if os.path.isfile(self.config.EVAL_CKPT_PATH):
                    # evaluate singe checkpoint
                    self._eval_checkpoint(self.config.EVAL_CKPT_PATH, writer)
                else:
                    # evaluate multiple checkpoints in order
                    while True:
                        current_ckpt = None
                        while current_ckpt is None:
                            current_ckpt = poll_checkpoint_folder(
                                self.config.EVAL_CKPT_PATH, prev_ckpt_ind, eval_interval
                            )
                            time.sleep(2)  # sleep for 2 secs before polling again
                        logger.info(f"=======current_ckpt: {current_ckpt}=======")
                        prev_ckpt_ind += eval_interval
                        self._eval_checkpoint(
                            checkpoint_path=current_ckpt,
                            writer=writer,
                            checkpoint_index=prev_ckpt_ind
                        )

    def _setup_eval_config(self, checkpoint_config: Config) -> Config:
        """
        Sets up and returns a merged config for evaluation. Config
            object saved from checkpoint is merged into config file specified
            at evaluation time with the following overwrite priority:
                  eval_opts > ckpt_opts > eval_cfg > ckpt_cfg
            If the saved config is outdated, only the eval config is returned.
        :param checkpoint_config: saved config from checkpoint.
        :return: merged config for eval.
        """

        config = self.config.clone()

        if checkpoint_config is not None:
            ckpt_cmd_opts = checkpoint_config.CMD_TRAILING_OPTS
            
        eval_cmd_opts = config.CMD_TRAILING_OPTS

        if checkpoint_config is not None:
            try:
                config.merge_from_other_cfg(checkpoint_config)
                config.merge_from_other_cfg(self.config)
                config.merge_from_list(ckpt_cmd_opts)
                config.merge_from_list(eval_cmd_opts)
            except KeyError:
                logger.info("Saved config is outdated, using solely eval config")
                config = self.config.clone()
                config.merge_from_list(eval_cmd_opts)
        else:
                logger.info("Saved config is outdated, using solely eval config")
                config = self.config.clone()
                config.merge_from_list(eval_cmd_opts)

        if config.TASK_CONFIG.DATASET.SPLIT == "train":
            config.TASK_CONFIG.defrost()
            config.TASK_CONFIG.DATASET.SPLIT = "val"

        config.TASK_CONFIG.SIMULATOR.AGENT_0.defrost()
        config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = self.config.SENSORS
        config.freeze()

        return config

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        """
        Evaluates a single checkpoint. Trainer algorithms should
        implement this.
        :param checkpoint_path: path of checkpoint
        :param writer: tensorboard writer object for logging to tensorboard
        :param checkpoint_index: index of cur checkpoint for logging
        :return: a dictionary containing the eval results
        """

        raise NotImplementedError

    def save_checkpoint(self, file_name) -> None:
        raise NotImplementedError

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        raise NotImplementedError

    @staticmethod
    def _pause_envs(
        envs_to_pause,
        envs,
        test_recurrent_hidden_states,
        not_done_masks,
        current_episode_reward,
        batch,
        episode_steps,
        episode_idxs,
    ):
        """
        pause environments when they run out of episodes
        :param envs_to_pause: environments to pause
        :param envs: all environments
        :param test_recurrent_hidden_states: policy RNN states during test
        :param not_done_masks: episode not done masks
        :param current_episode_reward: current episode reward
        :param batch: observation batch
        :param episode_steps: object tracking episode steps
        :param episode_idxs: object tracking episode indexes
        :return: tuple containing updated objects after pausing environments
        """

        max_ep_idx = torch.max(episode_idxs).item()

        # pausing self.envs with no new episode
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)

            # indexing along the batch dimensions
            test_recurrent_hidden_states = test_recurrent_hidden_states[
                :, state_index
            ]
            not_done_masks = not_done_masks[state_index]
            current_episode_reward = current_episode_reward[state_index]

            for k, v in batch.items():
                batch[k] = v[state_index]

            episode_steps = episode_steps[state_index]
            episode_idxs = episode_idxs[state_index]

        for env_idx, done in enumerate(not_done_masks):
            if env_idx > 0:
                max_ep_idx = torch.max(episode_idxs).item()
            if done[0].item() == 0:
                episode_idxs[env_idx, 0] = max_ep_idx + 1

        return (
            envs,
            test_recurrent_hidden_states,
            not_done_masks,
            current_episode_reward,
            batch,
            episode_steps,
            episode_idxs,
        )
