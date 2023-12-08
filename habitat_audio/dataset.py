# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import attr
import gzip
import json
import os
import logging
from typing import List, Optional, Dict, Any

from habitat.config import Config
from habitat.core.dataset import Dataset, Episode
from habitat.core.registry import registry
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    ShortestPathPoint,
)


ALL_SCENES_MASK = "*"
CONTENT_SCENES_PATH_FIELD = "content_scenes_path"
DEFAULT_SCENE_PATH_PREFIX = "data/scene_dataset/"


@attr.s(auto_attribs=True, kw_only=True)
class NavigationEpisodeCustom(NavigationEpisode):
    """
    Class for episode specification that includes all geodesic distances.
    :param all_geodesic_distances: geodesic distances of agent to source
            and in between two sources
    :param gt_actions: ground-truth actions
    :param other_starts: starting location and orientation for other agents
    :param other_info: episode info for other agents
    """

    all_geodesic_distances: Optional[Dict[str, str]] = None
    gt_actions: Optional[Dict[str, str]] = None
    other_starts: Any = None
    other_info: Any = None


@registry.register_dataset(name="chat2map")
class Chat2mapDataset(Dataset):
    episodes: List[NavigationEpisodeCustom]
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"

    @staticmethod
    def check_config_paths_exist(config: Config, is_train=True) -> bool:
        """
        check if paths to episode datasets exist
        :param config:
        :return: flag saying if config paths exist
        """

        if is_train:
            split = config.SPLIT 
        else:
            split = config.EVAL_SPLIT

        return os.path.exists(
            config.DATA_PATH.format(version=config.VERSION, split=split)
        ) and os.path.exists(config.SCENES_DIR)


    @staticmethod
    def get_scenes_to_load(config: Config, is_train=True) -> List[str]:
        """Return list of scene ids for which dataset has separate files with episodes.
        """
        if is_train:
            split = config.SPLIT 
        else:
            split = config.EVAL_SPLIT

        assert Chat2mapDataset.check_config_paths_exist(config, is_train=is_train), \
            (config.DATA_PATH.format(version=config.VERSION, split=split),
             config.SCENES_DIR)
        dataset_dir = os.path.dirname(
            config.DATA_PATH.format(version=config.VERSION, split=split)
        )

        cfg = config.clone()
        cfg.defrost()
        cfg.CONTENT_SCENES = []
        dataset = Chat2mapDataset(cfg, is_train=is_train,)
        return Chat2mapDataset._get_scenes_from_folder(
            content_scenes_path=dataset.content_scenes_path,
            dataset_dir=dataset_dir,
        )

    @staticmethod
    def _get_scenes_from_folder(content_scenes_path, dataset_dir):
        scenes = []
        content_dir = content_scenes_path.split("{scene}")[0]
        scene_dataset_ext = content_scenes_path.split("{scene}")[1]
        content_dir = content_dir.format(data_path=dataset_dir)
        if not os.path.exists(content_dir):
            return scenes

        for filename in os.listdir(content_dir):
            if filename.endswith(scene_dataset_ext):
                scene = filename[: -len(scene_dataset_ext)]
                scenes.append(scene)
        scenes.sort()
        return scenes

    def __init__(self, config: Optional[Config] = None, is_train=False) -> None:
        """Class inherited from Dataset that loads Point Navigation dataset.
        """
        self.episodes = []
        self._config = config

        if config is None:
            return

        if is_train:
            split = config.SPLIT 
        else:
            split = config.EVAL_SPLIT

        datasetfile_path = config.DATA_PATH.format(version=config.VERSION, split=split)
        with gzip.open(datasetfile_path, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.SCENES_DIR, scene_filename=datasetfile_path)

        if is_train:
            # Read separate file for each scene
            dataset_dir = os.path.dirname(datasetfile_path)
            scenes = config.CONTENT_SCENES
            if ALL_SCENES_MASK in scenes:
                scenes = chat2mapDataset._get_scenes_from_folder(
                    content_scenes_path=self.content_scenes_path,
                    dataset_dir=dataset_dir,
                )

            last_episode_cnt = 0
            for scene in scenes:
                scene_filename = self.content_scenes_path.format(
                    data_path=dataset_dir, scene=scene
                )
                with gzip.open(scene_filename, "rt") as f:
                    self.from_json(f.read(), scenes_dir=config.SCENES_DIR, scene_filename=scene_filename)

                num_episode = len(self.episodes) - last_episode_cnt
                last_episode_cnt = len(self.episodes)
                logging.info('Sampled {} from {}'.format(num_episode, scene))

        if not is_train:
            self.episodes = self.episodes[:config.EVAL_EPISODE_COUNT]

    # filter by scenes for data collection
    def filter_by_scenes(self, scenes):
        """
        filter all episodes on the basis of scene names
        :param scenes: scenes to filter episodes with
        :return: filtered episodes
        """
        episodes_to_keep = list()
        for episode in self.episodes:
            episode_scene = episode.scene_id.split("/")[-1].split(".")[0]
            if episode_scene in scenes:
                episodes_to_keep.append(episode)
        self.episodes = episodes_to_keep

    # filter by scenes for data collection
    def filter_by_scenes_n_ids(self, scenes_n_ids):
        """
        filter all episodes on the basis of scene names and episode IDs
        :param scenes_n_ids: scene names and episode IDs to filter all episodes with
        :return: filtered episodes
        """
        episodes_to_keep = list()
        for episode in self.episodes:
            episode_scene = episode.scene_id.split("/")[-1].split(".")[0]
            episode_id = int(episode.episode_id)
            if episode_scene + "_" + str(episode_id) in scenes_n_ids:
                episodes_to_keep.append(episode)
        self.episodes = episodes_to_keep

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None, scene_filename: Optional[str] = None
    ) -> None:
        """
        loads and reads episodes from per-scene json files
        :param json_str: json file name
        :param scenes_dir: directory containing json files
        :return: None
        """
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        episode_cnt = 0
        for episode in deserialized["episodes"]:
            episode = NavigationEpisodeCustom(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX):
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            for g_index, goal in enumerate(episode.goals):
                episode.goals[g_index] = NavigationGoal(**goal)
            if episode.shortest_paths is not None:
                for path in episode.shortest_paths:
                    for p_index, point in enumerate(path):
                        path[p_index] = ShortestPathPoint(**point)
            self.episodes.append(episode)
            episode_cnt += 1
