# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
import random
import pickle
from typing import Dict
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from torch.utils.data import DataLoader

from habitat import logger

from chat2map.common.base_trainer import BaseRLTrainer
from chat2map.common.baseline_registry import baseline_registry
from chat2map.common.tensorboard_utils import TensorboardWriter
from chat2map.common.loss_n_evalMetrics import compute_loss_n_evalMetrics

from chat2map.mapping.passive_mapping import PassiveMappingPolicy
from chat2map.mapping.passive_mapping.passive_mapping import PassiveMapping

from chat2map.mapping.datasets.dataset import PassiveMappingDataset

from habitat_audio.utils import load_points_data


SCENE_IDX_TO_NAME = {
    "mp3d":
        {0: 'sT4fr6TAbpF', 1: 'E9uDoFAP3SH', 2: 'VzqfbhrpDEA', 3: 'kEZ7cmS4wCh', 4: '29hnd4uzFmX', 5: 'ac26ZMwG7aT',
         6: 's8pcmisQ38h', 7: 'rPc6DW4iMge', 8: 'EDJbREhghzL', 9: 'mJXqzFtmKg4', 10: 'B6ByNegPMKs', 11: 'JeFG25nYj2p',
         12: '82sE5b5pLXE', 13: 'D7N2EKCX4Sj', 14: '7y3sRwLe3Va', 15: '5LpN3gDmAk7', 16: 'gTV8FGcVJC9', 17: 'ur6pFq6Qu1A',
         18: 'qoiz87JEwZ2', 19: 'PuKPg4mmafe', 20: 'VLzqgDo317F', 21: 'aayBHfsNo7d', 22: 'JmbYfDe2QKZ', 23: 'XcA2TqTSSAj',
         24: '8WUmhLawc2A', 25: 'sKLMLpTHeUy', 26: 'r47D5H71a5s', 27: 'Uxmj2M2itWa', 28: 'Pm6F8kyY3z2', 29: 'p5wJjkQkbXX',
         30: '759xd9YjKW5', 31: 'JF19kD82Mey', 32: 'V2XKFyX4ASd', 33: '1LXtFkjw3qL', 34: '17DRP5sb8fy', 35: '5q7pvUzZiYa',
         36: 'VVfe2KiqLaN', 37: 'Vvot9Ly1tCj', 38: 'ULsKaCPVFJR', 39: 'D7G3Y4RVNrH', 40: 'uNb9QFRL6hY', 41: 'ZMojNkEp431',
         42: '2n8kARJN3HM', 43: 'vyrNrziPKCB', 44: 'e9zR4mvMWw7', 45: 'r1Q1Z4BcV1o', 46: 'PX4nDJXEHrG', 47: 'YmJkqBEsHnH',
         48: 'b8cTxDM8gDG', 49: 'GdvgFV5R1Z5', 50: 'pRbA3pwrgk9', 51: 'jh4fc5c5qoQ', 52: '1pXnuDYAj8r', 53: 'S9hNv5qa7GM',
         54: 'VFuaQ6m2Qom', 55: 'cV4RVeZvu5T', 56: 'SN83YJsR3w2', 57: '2azQ1b91cZZ', 58: '5ZKStnWn8Zo', 59: '8194nk5LbLH',
         60: 'ARNzJeq3xxb', 61: 'EU6Fwq7SyZv', 62: 'QUCTc6BB5sX', 63: 'TbHJrupSAjP', 64: 'UwV83HsGsw3', 65: 'Vt2qJdWjCF2',
         66: 'WYY7iVyf5p8', 67: 'X7HyMhZNoso', 68: 'YFuZgdQ5vWj', 69: 'Z6MFQCViBuw', 70: 'fzynW3qQPVF', 71: 'gYvKGZ5eRqb',
         72: 'gxdoqLR6rwA', 73: 'jtcxE69GiFV', 74: 'oLBMNvg9in8', 75: 'pLe4wQe7qrG', 76: 'pa4otMbVnkk', 77: 'q9vSo1VnCiC',
         78: 'rqfALeAoiTq', 79: 'wc2JMjhGNzB', 80: 'x8F5xyUWy9e', 81: 'yqstnuAEVhm', 82: 'zsNo4HB9uLZ'},
}


SCENE_SPLITS = {
    "mp3d":
        {
            "train": ['sT4fr6TAbpF', 'E9uDoFAP3SH', 'VzqfbhrpDEA', 'kEZ7cmS4wCh', '29hnd4uzFmX',
                      'ac26ZMwG7aT', 's8pcmisQ38h', 'rPc6DW4iMge', 'EDJbREhghzL', 'mJXqzFtmKg4',
                      'B6ByNegPMKs', 'JeFG25nYj2p', '82sE5b5pLXE', 'D7N2EKCX4Sj', '7y3sRwLe3Va',
                      '5LpN3gDmAk7', 'gTV8FGcVJC9', 'ur6pFq6Qu1A', 'qoiz87JEwZ2', 'PuKPg4mmafe',
                      'VLzqgDo317F', 'aayBHfsNo7d', 'JmbYfDe2QKZ', 'XcA2TqTSSAj', '8WUmhLawc2A',
                      'sKLMLpTHeUy', 'r47D5H71a5s', 'Uxmj2M2itWa', 'Pm6F8kyY3z2', 'p5wJjkQkbXX',
                      '759xd9YjKW5', 'JF19kD82Mey', 'V2XKFyX4ASd', '1LXtFkjw3qL', '17DRP5sb8fy',
                      '5q7pvUzZiYa', 'VVfe2KiqLaN', 'Vvot9Ly1tCj', 'ULsKaCPVFJR', 'D7G3Y4RVNrH',
                      'uNb9QFRL6hY', 'ZMojNkEp431', '2n8kARJN3HM', 'vyrNrziPKCB', 'e9zR4mvMWw7',
                      'r1Q1Z4BcV1o', 'PX4nDJXEHrG', 'YmJkqBEsHnH', 'b8cTxDM8gDG', 'GdvgFV5R1Z5',
                      'pRbA3pwrgk9', 'jh4fc5c5qoQ', '1pXnuDYAj8r', 'S9hNv5qa7GM', 'VFuaQ6m2Qom',
                      'cV4RVeZvu5T', 'SN83YJsR3w2', ],
            "val": ['QUCTc6BB5sX', 'EU6Fwq7SyZv', '2azQ1b91cZZ', 'Z6MFQCViBuw', 'pLe4wQe7qrG', 'oLBMNvg9in8',
                    'X7HyMhZNoso', 'zsNo4HB9uLZ', 'TbHJrupSAjP', '8194nk5LbLH', ],
            "test": ['pa4otMbVnkk', 'yqstnuAEVhm', '5ZKStnWn8Zo', 'Vt2qJdWjCF2', 'wc2JMjhGNzB', 'fzynW3qQPVF',
                     'UwV83HsGsw3', 'q9vSo1VnCiC', 'ARNzJeq3xxb', 'gYvKGZ5eRqb', 'jtcxE69GiFV', 'gxdoqLR6rwA',
                     'WYY7iVyf5p8', 'YFuZgdQ5vWj', 'rqfALeAoiTq', 'x8F5xyUWy9e',],
        },
}


@baseline_registry.register_trainer(name="chat2map_passiveMappingTrainer")
class PassiveMappingTrainer(BaseRLTrainer):
    r"""Trainer class for training chat2map passive mapping model
    """
    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self._n_available_gpus = None

    def _setup_passiveMapping_agent(self, use_data_parallel=False,) -> None:
        """Sets up chat2map passive mapping model"""
        logger.add_filehandler(self.config.LOG_FILE)

        policy_className = PassiveMappingPolicy

        self.actor_critic = policy_className(
            cfg=self.config,
        )

        self._n_available_gpus = torch.cuda.device_count()
        self._use_data_parallel = use_data_parallel

        self.actor_critic.to(self.device)

        if (self._n_available_gpus > 0) or use_data_parallel:
            print("Using", torch.cuda.device_count(), "GPUs!")

        self.actor_critic = nn.DataParallel(self.actor_critic,
                                            device_ids=list(range(self._n_available_gpus)),
                                            output_device=0)

        self.agent = PassiveMapping(
            actor_critic=self.actor_critic,
        )

    def save_checkpoint(self, file_name: str, epoch: int, optimizer) -> None:
        """
        Saves checkpoint with specified name
        :param file_name: file name for checkpoint
        :param epoch: Training epoch that's been completed
        :param optimizer: mode optimizer
        :return: None
        """

        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
        }

        checkpoint["optimizer_mapper"] = optimizer.state_dict()
        checkpoint["last_epoch"] = epoch

        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        """
        Load checkpoint of specified path as a dict.
        :param checkpoint_path: path of target checkpoint
        :param args: additional positional args
        :param kwargs: additional keyword args
        :return: dict containing checkpoint info
        """

        return torch.load(checkpoint_path, *args, **kwargs)

    def load_trainer_state_dict(self, checkpoint_path: str, just_copy_config=False, optimizer=None,):
        """

        :param checkpoint_path: checkpoint path
        :param just_copy_config: flag saying if only config needs to be copied without loading anything else
        :param optimizer: model optimizer
        :return: starting epoch and optimizer for new training if just_copy_config = False, else None
        """

        ckpt_dict = torch.load(checkpoint_path, map_location="cpu")

        if just_copy_config:
            self.config = ckpt_dict["config"]
        else:
            self.agent.load_state_dict(ckpt_dict["state_dict"])
            optimizer.load_state_dict(ckpt_dict["optimizer_mapper"])

            return ckpt_dict["last_epoch"] + 1, optimizer

    def get_dataloaders(self, eval_mode=False,):
        """
        build datasets and dataloaders
        :param eval_mode: flag saying if in eval mode or not
        :return:
            dataloaders: PyTorch dataloaders for training and validation
            dataset_sizes: sizes of train and val datasets
        """

        passive_mapping_cfg = self.config.PassiveMapping
        task_cfg = self.config.TASK_CONFIG
        sim_cfg = task_cfg.SIMULATOR
        audio_cfg = sim_cfg.AUDIO

        scene_dataset = sim_cfg.SCENE_DATASET

        scene_splits = {}
        if not eval_mode:
            scene_splits["train"] = SCENE_SPLITS[scene_dataset]["train"]
            scene_splits["val"] = SCENE_SPLITS[scene_dataset]["val"]
        scene_splits["test"] = SCENE_SPLITS[scene_dataset]["test"]

        all_scenes_lst = []
        for scenes_lst in scene_splits.values():
            all_scenes_lst += scenes_lst

        scene_observations_dir = os.path.join(sim_cfg.RENDERED_OBSERVATIONS, scene_dataset)
        assert os.path.isdir(scene_observations_dir)
        all_scenes_observations = dict()
        print("LOADING CACHED SCENE OBSERVATIONS")
        for scene in tqdm(all_scenes_lst):
            scene_observations_file_path = os.path.join(scene_observations_dir, f"{scene}.pkl")
            with open(scene_observations_file_path, "rb") as fi:
                all_scenes_observations[scene] = pickle.load(fi)

        datasets = dict()
        dataloaders = dict()
        dataset_sizes = dict()
        for split in scene_splits:
            scenes = scene_splits[split]
            all_scenes_graphs_this_split = dict()
            for scene in scenes:
                _, graph = load_points_data(
                    os.path.join(audio_cfg.META_DIR, scene),
                    audio_cfg.GRAPH_FILE,
                    transform=True,
                    scene_dataset=sim_cfg.SCENE_DATASET)
                all_scenes_graphs_this_split[scene] = graph

            datasets[split] = PassiveMappingDataset(
                split=split,
                all_scenes_graphs_this_split=all_scenes_graphs_this_split,
                cfg=self.config,
                all_scenes_observations=all_scenes_observations,
                eval_mode=eval_mode,
            )

            dataloaders[split] = DataLoader(dataset=datasets[split],
                                            batch_size=passive_mapping_cfg.batch_size,
                                            shuffle=(split=='train'),
                                            pin_memory=True,
                                            num_workers=passive_mapping_cfg.num_workers
                                            )
            dataset_sizes[split] = len(datasets[split])
            print('{} has {} samples'.format(split.upper(), dataset_sizes[split]))

        if eval_mode:
            return datasets, dataloaders, dataset_sizes
        else:
            return dataloaders, dataset_sizes

    def _optimize_loss(self, loss, optimizer):
        """optimize mapper training loss"""
        assert self.config.PassiveMapping.max_grad_norm is None

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return optimizer

    def train(self) -> None:
        """Main method for mapper training .

        Returns:
            None
        """

        num_ckpts_trained = 0
        if self.config.RESUME_AFTER_PREEMPTION:
            old_ckpt_found = False
            if os.path.isdir(self.config.CHECKPOINT_FOLDER) and (len(os.listdir(self.config.CHECKPOINT_FOLDER)) != 0):
                lst_ckpt_filenames = os.listdir(self.config.CHECKPOINT_FOLDER)

                ckpt_file_maxIdx = float('-inf')
                for ckpt_filename in lst_ckpt_filenames:
                    if int(ckpt_filename.split(".")[1]) > ckpt_file_maxIdx:
                        ckpt_file_maxIdx = int(ckpt_filename.split(".")[1])
                most_recent_ckpt_filename = f"best_ckpt_val.{ckpt_file_maxIdx}.pth"

                most_recent_ckpt_file_path = os.path.join(self.config.CHECKPOINT_FOLDER,
                                                          most_recent_ckpt_filename)
                num_ckpts_trained = int(most_recent_ckpt_filename.split(".")[1])

                old_ckpt_found = True

            old_tb_dir = os.path.join(self.config.MODEL_DIR, "tb")
            if os.path.isdir(old_tb_dir):
                for old_tb_idx in range(1, 10000):
                    if not os.path.isdir(os.path.join(self.config.MODEL_DIR, f"tb_{old_tb_idx}")):
                        new_tb_dir = os.path.join(self.config.MODEL_DIR, f"tb_{old_tb_idx}")
                        os.system(f"mv {old_tb_dir} {new_tb_dir}")
                        break

            if old_ckpt_found:
                assert os.path.isfile(most_recent_ckpt_file_path)

                self.load_trainer_state_dict(
                    most_recent_ckpt_file_path,
                    just_copy_config=True,
                )

        passive_mapping_cfg = self.config.PassiveMapping

        assert "RGB_SENSOR" in self.config.SENSORS

        logger.info(f"config: {self.config}")
        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)

        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        self._setup_passiveMapping_agent()
        assert self._n_available_gpus is not None

        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.actor_critic.parameters()),
                                     lr=passive_mapping_cfg.lr,
                                     betas=tuple(passive_mapping_cfg.betas),
                                     eps=passive_mapping_cfg.eps,
                                     weight_decay=passive_mapping_cfg.weight_decay)

        # build datasets and dataloaders
        dataloaders, dataset_sizes = self.get_dataloaders()

        all_metric_types = passive_mapping_cfg.EvalMetrics.types
        metric_type_for_ckpt_dump = passive_mapping_cfg.EvalMetrics.type_for_ckpt_dump
        assert metric_type_for_ckpt_dump in all_metric_types

        best_val_metric_for_ckpt_dump = float('-inf')

        starting_epoch = 0
        if self.config.RESUME_AFTER_PREEMPTION:
            if old_ckpt_found:
                starting_epoch, optimizer =\
                    self.load_trainer_state_dict(most_recent_ckpt_file_path,
                                                 just_copy_config=False,
                                                 optimizer=optimizer,
                                                 )
        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            for epoch in range(starting_epoch, passive_mapping_cfg.num_epochs):
                logging.info('-' * 10)
                logging.info('Epoch {}/{}'.format(epoch + 1, passive_mapping_cfg.num_epochs))

                for split in dataloaders.keys():
                    # set forward pass mode
                    if split == "train":
                        self.actor_critic.train()
                    else:
                        self.actor_critic.eval()

                    eval_metrics_epoch = {}
                    for batch_idx, data in enumerate(tqdm(dataloaders[split])):
                        context_maps = data[0].to(self.device)
                        context_rgbs = data[1].to(self.device)
                        context_views_pose = data[2].to(self.device)
                        context_views_mask = data[3].to(self.device)

                        context_selfAudio = data[4].to(self.device)
                        context_selfAudio_pose = data[5].to(self.device)
                        context_selfAudio_mask = data[6].to(self.device)

                        context_otherAudio = data[7].to(self.device)
                        context_otherAudio_pose = data[8].to(self.device)
                        context_otherAudio_mask = data[9].to(self.device)

                        query_maps_gt = data[10].to(self.device)
                        query_maps_exploredMasks = data[11].to(self.device)
                        query_views_pose = data[12].to(self.device)     # query_views_pose is same as context_views_pose
                        query_views_mask = data[13].to(self.device)     # some of the mask values are set to 0 because of repetition in pose

                        B = context_maps.size(0)

                        obs_batch = {"context_maps": context_maps,
                                     "context_rgbs": context_rgbs,
                                     "context_views_pose": context_views_pose,
                                     "context_views_mask": context_views_mask,
                                     "context_selfAudio": context_selfAudio,
                                     "context_selfAudio_pose": context_selfAudio_pose,
                                     "context_selfAudio_mask": context_selfAudio_mask,
                                     "context_otherAudio": context_otherAudio,
                                     "context_otherAudio_pose": context_otherAudio_pose,
                                     "context_otherAudio_mask": context_otherAudio_mask,
                                     "query_views_pose": query_views_pose,
                                     "query_views_mask": query_views_mask,
                                     }

                        if split == "train":
                            preds = self.actor_critic(obs_batch)
                        else:
                            with torch.no_grad():
                                preds = self.actor_critic(obs_batch)

                        if split == "train":
                            loss = compute_loss_n_evalMetrics(
                                loss_or_metric_types=passive_mapping_cfg.TrainLosses.types,
                                loss_or_metric_weights=passive_mapping_cfg.TrainLosses.weights,
                                gt_occMap=query_maps_gt.view(-1, *query_maps_gt.size()[2:]),
                                pred_occMap=preds.view(-1, *preds.size()[2:]),
                                mask=query_views_mask.view(-1),
                            )

                            optimizer = self._optimize_loss(loss, optimizer)

                        eval_loss = compute_loss_n_evalMetrics(
                            loss_or_metric_types=passive_mapping_cfg.TrainLosses.types,
                            loss_or_metric_weights=passive_mapping_cfg.TrainLosses.weights,
                            gt_occMap=query_maps_gt.view(-1, *query_maps_gt.size()[2:])[..., :1],
                            pred_occMap=preds.view(-1, *preds.size()[2:])[..., :1],
                            mask=query_views_mask.view(-1),
                        )
                        eval_metrics_batch = {passive_mapping_cfg.TrainLosses.types[0]: eval_loss}

                        for evalMetric_idx in range(len(all_metric_types)):
                            eval_metric_1 = compute_loss_n_evalMetrics(
                                loss_or_metric_types=[all_metric_types[evalMetric_idx]],
                                loss_or_metric_weights=[1.0],
                                gt_occMap=query_maps_gt.view(-1, *query_maps_gt.size()[2:])[..., :1],
                                pred_occMap=(preds.view(-1, *preds.size()[2:])[..., :1] > 0.5).float(),
                                mask=query_views_mask.view(-1),
                                exploredPart_mask=query_maps_exploredMasks.view(-1, *query_maps_exploredMasks.size()[2:])[..., :1],     # computes the metrics where the maps have already not been seen in the input
                                target_category=1.0,
                            )
                            eval_metrics_batch[passive_mapping_cfg.EvalMetrics.types[evalMetric_idx] + "_1"] = eval_metric_1

                            eval_metric_0 = compute_loss_n_evalMetrics(
                                loss_or_metric_types=[all_metric_types[evalMetric_idx]],
                                loss_or_metric_weights=[1.0],
                                gt_occMap=query_maps_gt.view(-1, *query_maps_gt.size()[2:])[..., :1],
                                pred_occMap=(preds.view(-1, *preds.size()[2:])[..., :1] > 0.5).float(),
                                mask=query_views_mask.view(-1),
                                exploredPart_mask=query_maps_exploredMasks.view(-1, *query_maps_exploredMasks.size()[2:])[..., :1],     # computes the metrics where the maps have already not been seen in the input
                                target_category=0.,
                            )
                            eval_metrics_batch[passive_mapping_cfg.EvalMetrics.types[evalMetric_idx] + "_0"] = eval_metric_0

                        for metric_type in eval_metrics_batch:
                            if metric_type not in eval_metrics_epoch:
                                eval_metrics_epoch[metric_type] = (eval_metrics_batch[metric_type].item() * B)
                            else:
                                eval_metrics_epoch[metric_type] += (eval_metrics_batch[metric_type].item() * B)

                    for metric_type in eval_metrics_epoch.keys():
                        eval_metrics_epoch[metric_type] /= dataset_sizes[split]

                        writer.add_scalar('{}/{}'.format(metric_type, split),
                                          eval_metrics_epoch[metric_type],
                                          epoch)

                        logging.info('{} -- {}: {:.4f}'.format(split.upper(),
                                                               metric_type,
                                                               eval_metrics_epoch[metric_type],
                                                               ))
                    if split == "val":
                        if 0.5 * (eval_metrics_epoch[metric_type_for_ckpt_dump + "_1"]\
                                  + eval_metrics_epoch[metric_type_for_ckpt_dump + "_0"]) >\
                                best_val_metric_for_ckpt_dump:
                            best_val_metric_for_ckpt_dump = 0.5 * (eval_metrics_epoch[metric_type_for_ckpt_dump + "_1"]\
                                                                   + eval_metrics_epoch[metric_type_for_ckpt_dump + "_0"])
                            self.save_checkpoint(f"best_ckpt_val.{num_ckpts_trained + 1}.pth", epoch, optimizer)

                    torch.cuda.empty_cache()

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0
    ) -> Dict:
        """evaluates a particular passive mapping checkpoint and dumps the eval metrics to the disk"""

        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)

        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        # setting up config
        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        logger.info(f"config: {config}")

        passive_mapping_cfg = config.PassiveMapping
        env_cfg = config.TASK_CONFIG.ENVIRONMENT

        assert "RGB_SENSOR" in config.SENSORS
        assert config.STITCH_TOP_DOWN_MAPS, "set config.STITCH_TOP_DOWN_MAPS to True"

        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self._setup_passiveMapping_agent(use_data_parallel=config.EVAL.DATA_PARALLEL_TRAINING)
        assert self._n_available_gpus is not None

        self.actor_critic = self.agent.actor_critic
        self.actor_critic.eval()
        self.agent.eval()
        self.agent.actor_critic.eval()

        # build datasets and dataloaders
        datasets, dataloaders, dataset_sizes = self.get_dataloaders(eval_mode=True)

        all_metric_types = passive_mapping_cfg.EvalMetrics.types

        for split in dataloaders.keys():
            eval_metrics = {}
            for batch_idx, data in enumerate(tqdm(dataloaders[split])):
                context_maps = data[0].to(self.device)
                context_rgbs = data[1].to(self.device)
                context_views_pose = data[2].to(self.device)
                context_views_mask = data[3].to(self.device)

                context_selfAudio = data[4].to(self.device)
                context_selfAudio_pose = data[5].to(self.device)
                context_selfAudio_mask = data[6].to(self.device)

                context_otherAudio = data[7].to(self.device)
                context_otherAudio_pose = data[8].to(self.device)
                context_otherAudio_mask = data[9].to(self.device)

                query_maps_gt = data[10].to(self.device)
                query_maps_exploredMasks = data[11].to(self.device)
                query_views_pose = data[12].to(self.device)     # query_views_pose is same as context_views_pose
                query_views_mask = data[13].to(self.device)     # some of the mask values are set to 0 because of repetition in pose
                query_sceneIdxs = data[14]
                query_rAzs = data[15]
                query_epIdxs = data[16]
                if self.config.STITCH_TOP_DOWN_MAPS:
                    stitched_egoLocalOccMaps = data[17].to(self.device)
                    ref_rAz = data[18].to(self.device)
                    stitched_globalCanOccMaps_egoCrops_gt = data[19].to(self.device)

                assert env_cfg.MAX_QUERY_LENGTH == query_maps_gt.size(1)

                obs_batch = {"context_maps": context_maps,
                             "context_rgbs": context_rgbs,
                             "context_views_pose": context_views_pose,
                             "context_views_mask": context_views_mask,
                             "context_selfAudio": context_selfAudio,
                             "context_selfAudio_pose": context_selfAudio_pose,
                             "context_selfAudio_mask": context_selfAudio_mask,
                             "context_otherAudio": context_otherAudio,
                             "context_otherAudio_pose": context_otherAudio_pose,
                             "context_otherAudio_mask": context_otherAudio_mask,
                             "query_views_pose": query_views_pose,
                             "query_views_mask": query_views_mask,
                             }

                with torch.no_grad():
                    preds = self.actor_critic(obs_batch)

                if self.config.STITCH_TOP_DOWN_MAPS:
                    query_stitchedPredGlobCanOccMapsEgoCrops_ph_lst = []

                for idx_epIdx in range(query_epIdxs.size(0)):
                    if self.config.STITCH_TOP_DOWN_MAPS:
                        scene_idx = query_sceneIdxs[idx_epIdx][0].item()
                        datapoint_scene = SCENE_IDX_TO_NAME[self.config.TASK_CONFIG.SIMULATOR.SCENE_DATASET][scene_idx]
                        ref_rAz_thisEp = ref_rAz[idx_epIdx].cpu().numpy().tolist()
                        preds_thisEp = preds[idx_epIdx].cpu().numpy()

                        query_stitchedPredGlobCanOccMapsEgoCrops_ph = np.zeros((
                            self.config.TASK_CONFIG.SIMULATOR.EGO_STITCHED_GLOBAL_CANONICAL_OCC_MAP.SIZE,
                            self.config.TASK_CONFIG.SIMULATOR.EGO_STITCHED_GLOBAL_CANONICAL_OCC_MAP.SIZE,
                            1)).astype("float32")

                        query_stitchedPredGlobCanOccMapsEgoCrops_validUpdateCount_ph = np.zeros((
                            self.config.TASK_CONFIG.SIMULATOR.EGO_STITCHED_GLOBAL_CANONICAL_OCC_MAP.SIZE,
                            self.config.TASK_CONFIG.SIMULATOR.EGO_STITCHED_GLOBAL_CANONICAL_OCC_MAP.SIZE,
                            1)).astype("float32")

                    for idx_maskVal in range(query_views_mask[idx_epIdx].size(0)):
                        mask_val = query_views_mask[idx_epIdx][idx_maskVal].item()
                        assert mask_val in [0, 1]
                        if mask_val == 1:
                            if self.config.STITCH_TOP_DOWN_MAPS:
                                query_stitchedPredGlobCanOccMapsEgoCrops_ph, query_stitchedPredGlobCanOccMapsEgoCrops_validUpdateCount_ph =\
                                    datasets[split].get_stitched_top_down_maps(stitched_map=query_stitchedPredGlobCanOccMapsEgoCrops_ph,
                                                                               stitch_component=preds_thisEp[idx_maskVal],
                                                                               ref_pose=ref_rAz_thisEp,
                                                                               target_pose=query_rAzs[idx_epIdx][idx_maskVal].numpy(),
                                                                               scene=datapoint_scene,
                                                                               is_occupancy=True,
                                                                               is_pred=True,
                                                                               is_ego_360deg_crops=True,
                                                                               stitched_map_updateCounter=query_stitchedPredGlobCanOccMapsEgoCrops_validUpdateCount_ph,)
                        elif mask_val == 0:
                            break

                    if self.config.STITCH_TOP_DOWN_MAPS:
                        # replaces 0s in the update counter with 1s to keep division in the coming lines valid
                        query_stitchedPredGlobCanOccMapsEgoCrops_validUpdateCount_ph =\
                            (query_stitchedPredGlobCanOccMapsEgoCrops_validUpdateCount_ph == 0.).astype("float32") + \
                            (query_stitchedPredGlobCanOccMapsEgoCrops_validUpdateCount_ph != 0.).astype("float32") *\
                            query_stitchedPredGlobCanOccMapsEgoCrops_validUpdateCount_ph
                        query_stitchedPredGlobCanOccMapsEgoCrops_ph = query_stitchedPredGlobCanOccMapsEgoCrops_ph\
                                                                      / (query_stitchedPredGlobCanOccMapsEgoCrops_validUpdateCount_ph\
                                                                         + 1.0e-13)
                        assert np.prod((query_stitchedPredGlobCanOccMapsEgoCrops_ph >= 0).astype("float32") *\
                                          (query_stitchedPredGlobCanOccMapsEgoCrops_ph <= 1.).astype("float32")).item() == 1,\
                            print(query_stitchedPredGlobCanOccMapsEgoCrops_ph.max(),
                                  query_stitchedPredGlobCanOccMapsEgoCrops_ph.min())
                        query_stitchedPredGlobCanOccMapsEgoCrops_ph =\
                            (query_stitchedPredGlobCanOccMapsEgoCrops_ph > 0.5).astype("float32")

                        query_stitchedPredGlobCanOccMapsEgoCrops_ph_lst.append(query_stitchedPredGlobCanOccMapsEgoCrops_ph)

                if self.config.STITCH_TOP_DOWN_MAPS:
                    gts_stitched = stitched_globalCanOccMaps_egoCrops_gt
                    inputs_stitched = stitched_egoLocalOccMaps
                    preds_stitched = torch.from_numpy(np.stack(query_stitchedPredGlobCanOccMapsEgoCrops_ph_lst)).to(self.device)

                    preds_stitched = preds_stitched * (inputs_stitched[..., 1:] != 1.0).float() + inputs_stitched[..., :1] * (inputs_stitched[..., 1:] == 1.0).float()

                    evalStitched_metrics_batch = {}
                    for evalMetric_idx in range(len(all_metric_types)):
                        evalStitched_metric_1 = compute_loss_n_evalMetrics(
                            loss_or_metric_types=[all_metric_types[evalMetric_idx]],
                            loss_or_metric_weights=[1.0],
                            gt_occMap=gts_stitched[..., :1],
                            pred_occMap=(preds_stitched[..., :1] > 0.5).float(),
                            mask=torch.ones(query_views_mask.size(0)).to(self.device),
                            target_category=1.0,
                            eval_mode=True,
                            is_stitched=True,
                        )
                        evalStitched_metrics_batch["predsStitched_" +
                                                   passive_mapping_cfg.EvalMetrics.types[evalMetric_idx] + "_1"] =\
                            evalStitched_metric_1

                        evalStitched_metric_0 = compute_loss_n_evalMetrics(
                            loss_or_metric_types=[all_metric_types[evalMetric_idx]],
                            loss_or_metric_weights=[1.0],
                            gt_occMap=gts_stitched[..., :1],
                            pred_occMap=(preds_stitched[..., :1] > 0.5).float(),
                            mask=torch.ones(query_views_mask.size(0)).to(self.device),
                            target_category=0.,
                            eval_mode=True,
                            is_stitched=True,
                        )
                        evalStitched_metrics_batch["predsStitched_" +
                                                   passive_mapping_cfg.EvalMetrics.types[evalMetric_idx] + "_0"] =\
                            evalStitched_metric_0

                        evalInputsStitched_metric_1 = compute_loss_n_evalMetrics(
                            loss_or_metric_types=[all_metric_types[evalMetric_idx]],
                            loss_or_metric_weights=[1.0],
                            gt_occMap=gts_stitched[..., :1],
                            pred_occMap=(inputs_stitched[..., :1] > 0.5).float(),
                            mask=torch.ones(query_views_mask.size(0)).to(self.device),
                            target_category=1.0,
                            eval_mode=True,
                            is_stitched=True,
                        )
                        evalStitched_metrics_batch["inputsStitched_" +
                                                   passive_mapping_cfg.EvalMetrics.types[evalMetric_idx] + "_1"] =\
                            evalInputsStitched_metric_1

                        evalInputsStitched_metric_0 = compute_loss_n_evalMetrics(
                            loss_or_metric_types=[all_metric_types[evalMetric_idx]],
                            loss_or_metric_weights=[1.0],
                            gt_occMap=gts_stitched[..., :1],
                            pred_occMap=(inputs_stitched[..., :1] > 0.5).float(),
                            mask=torch.ones(query_views_mask.size(0)).to(self.device),
                            target_category=0.,
                            eval_mode=True,
                            is_stitched=True,
                        )
                        evalStitched_metrics_batch["inputsStitched_" +
                                                   passive_mapping_cfg.EvalMetrics.types[evalMetric_idx] + "_0"] =\
                            evalInputsStitched_metric_0

                    for eval_metricName in evalStitched_metrics_batch.keys():
                        if eval_metricName not in eval_metrics:
                            eval_metrics[eval_metricName] = evalStitched_metrics_batch[eval_metricName]
                        else:
                            eval_metrics[eval_metricName] += evalStitched_metrics_batch[eval_metricName]

            for metric_type in eval_metrics:
                if isinstance(eval_metrics[metric_type], list):
                    writer.add_scalar('{}/{}/mean'.format(metric_type, split),
                                      np.mean(eval_metrics[metric_type]),
                                      0)

                    writer.add_scalar('{}/{}/std'.format(metric_type, split),
                                      np.std(eval_metrics[metric_type]),
                                      0)

                    logger.info(f"{split.upper()} -- {metric_type}: "
                                f" mean -- {np.mean(eval_metrics[metric_type]):.4f}, "
                                f"std -- {np.std(eval_metrics[metric_type]):.4f}")
                elif isinstance(eval_metrics[metric_type], dict):
                    writer.add_scalar('{}/{}/mean'.format(metric_type, split),
                                      np.mean(list(eval_metrics[metric_type].values())),
                                      0)

                    writer.add_scalar('{}/{}/std'.format(metric_type, split),
                                      np.std(list(eval_metrics[metric_type].values())),
                                      0)

                    logger.info(f"{split.upper()} -- {metric_type}: "
                                f" mean -- {np.mean(list(eval_metrics[metric_type].values())):.4f}, "
                                f"std -- {np.std(list(eval_metrics[metric_type].values())):.4f}")
                else:
                    raise ValueError

            with open(os.path.join(config.MODEL_DIR, f"{split}_{dataset_sizes[split]}datapoints_metrics.pkl"), "wb") as fo:
                pickle.dump(eval_metrics, fo, protocol=pickle.HIGHEST_PROTOCOL)
