# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
from os import path as osp
from typing import Union
import json
import gzip
import pickle
import math
import time
import attr
import numba
from tqdm import tqdm
import numpy as np
from scipy.io import wavfile
import scipy
from scipy.signal import fftconvolve
import librosa
import cv2
from skimage.measure import block_reduce

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from habitat.tasks.utils import (
    cartesian_to_polar,
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from habitat_sim.utils.common import quat_from_angle_axis, quat_from_coeffs, quat_to_angle_axis


SCENE_NAME_TO_IDX = {
    "mp3d":
        {'sT4fr6TAbpF': 0, 'E9uDoFAP3SH': 1, 'VzqfbhrpDEA': 2, 'kEZ7cmS4wCh': 3, '29hnd4uzFmX': 4, 'ac26ZMwG7aT': 5,
         's8pcmisQ38h': 6, 'rPc6DW4iMge': 7, 'EDJbREhghzL': 8, 'mJXqzFtmKg4': 9, 'B6ByNegPMKs': 10, 'JeFG25nYj2p': 11,
         '82sE5b5pLXE': 12, 'D7N2EKCX4Sj': 13, '7y3sRwLe3Va': 14, '5LpN3gDmAk7': 15, 'gTV8FGcVJC9': 16, 'ur6pFq6Qu1A': 17,
         'qoiz87JEwZ2': 18, 'PuKPg4mmafe': 19, 'VLzqgDo317F': 20, 'aayBHfsNo7d': 21, 'JmbYfDe2QKZ': 22, 'XcA2TqTSSAj': 23,
         '8WUmhLawc2A': 24, 'sKLMLpTHeUy': 25, 'r47D5H71a5s': 26, 'Uxmj2M2itWa': 27, 'Pm6F8kyY3z2': 28, 'p5wJjkQkbXX': 29,
         '759xd9YjKW5': 30, 'JF19kD82Mey': 31, 'V2XKFyX4ASd': 32, '1LXtFkjw3qL': 33, '17DRP5sb8fy': 34, '5q7pvUzZiYa': 35,
         'VVfe2KiqLaN': 36, 'Vvot9Ly1tCj': 37, 'ULsKaCPVFJR': 38, 'D7G3Y4RVNrH': 39, 'uNb9QFRL6hY': 40, 'ZMojNkEp431': 41,
         '2n8kARJN3HM': 42, 'vyrNrziPKCB': 43, 'e9zR4mvMWw7': 44, 'r1Q1Z4BcV1o': 45, 'PX4nDJXEHrG': 46, 'YmJkqBEsHnH': 47,
         'b8cTxDM8gDG': 48, 'GdvgFV5R1Z5': 49, 'pRbA3pwrgk9': 50, 'jh4fc5c5qoQ': 51, '1pXnuDYAj8r': 52, 'S9hNv5qa7GM': 53,
         'VFuaQ6m2Qom': 54, 'cV4RVeZvu5T': 55, 'SN83YJsR3w2': 56, '2azQ1b91cZZ': 57, '5ZKStnWn8Zo': 58, '8194nk5LbLH': 59,
         'ARNzJeq3xxb': 60, 'EU6Fwq7SyZv': 61, 'QUCTc6BB5sX': 62, 'TbHJrupSAjP': 63, 'UwV83HsGsw3': 64, 'Vt2qJdWjCF2': 65,
         'WYY7iVyf5p8': 66, 'X7HyMhZNoso': 67, 'YFuZgdQ5vWj': 68, 'Z6MFQCViBuw': 69, 'fzynW3qQPVF': 70, 'gYvKGZ5eRqb': 71,
         'gxdoqLR6rwA': 72, 'jtcxE69GiFV': 73, 'oLBMNvg9in8': 74, 'pLe4wQe7qrG': 75, 'pa4otMbVnkk': 76, 'q9vSo1VnCiC': 77,
         'rqfALeAoiTq': 78, 'wc2JMjhGNzB': 79, 'x8F5xyUWy9e': 80, 'yqstnuAEVhm': 81, 'zsNo4HB9uLZ': 82},
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
                              'WYY7iVyf5p8', 'YFuZgdQ5vWj', 'rqfALeAoiTq', 'x8F5xyUWy9e',]
        },
}


ALL_AZIMUTHS = [0, 90, 180, 270]


def asnumpy(v):
    if torch.is_tensor(v):
        return v.cpu().numpy()
    elif isinstance(v, np.ndarray):
        return v
    else:
        raise ValueError('Invalid input')


# Read about the noise model here: http://www.alexteichman.com/octo/clams/
# Original source code: http://redwood-data.org/indoor/data/simdepth.py
@numba.jit(nopython=True, fastmath=True)
def undistort_redwood_depth_noise(x, y, z, model):
    i2 = int((z + 1) / 2)
    i1 = int(i2 - 1)
    a = (z - (i1 * 2.0 + 1.0)) / 2.0
    x = x // 8
    y = y // 6
    f = (1.0 - a) * model[y, x, min(max(i1, 0), 4)] + a * model[y, x, min(i2, 4)]

    if f < 1e-5:
        return 0.0
    else:
        return z / f


@numba.jit(nopython=True, parallel=True, fastmath=True)
def simulate_redwood_depth_noise(gt_depth, model, noise_multiplier, rand_nums):
    noisy_depth = np.empty_like(gt_depth)

    H, W = gt_depth.shape
    ymax, xmax = H - 1.0, W - 1.0

    # Parallelize just the outer loop.  This doesn't change the speed
    # noticably but reduces CPU usage compared to two parallel loops
    for j in numba.prange(H):
        for i in range(W):
            y = int(
                min(max(j + rand_nums[j, i, 0] * 0.25 * noise_multiplier, 0.0), ymax)
                + 0.5
            )
            x = int(
                min(max(i + rand_nums[j, i, 1] * 0.25 * noise_multiplier, 0.0), xmax)
                + 0.5
            )

            # Downsample
            d = gt_depth[y - y % 2, x - x % 2]
            # If the depth is greater than 10, the sensor will just return 0
            if d >= 10.0:
                noisy_depth[j, i] = 0.0
            else:
                # Distort
                # The noise model was originally made for a 640x480 sensor,
                # so re-map our arbitrarily sized sensor to that size!
                undistorted_d = undistort_redwood_depth_noise(
                    int(x / xmax * 639.0 + 0.5), int(y / ymax * 479.0 + 0.5), d, model
                )

                if undistorted_d == 0.0:
                    noisy_depth[j, i] = 0.0
                else:
                    denom = round(
                        (
                            35.130 / undistorted_d
                            + rand_nums[j, i, 2] * 0.027778 * noise_multiplier
                        )
                        * 8.0
                    )
                    if denom <= 1e-5:
                        noisy_depth[j, i] = 0.0
                    else:
                        noisy_depth[j, i] = 35.130 * 8.0 / denom

    return noisy_depth


class EgoMap:
    """
    Estimates the top-down occupancy based on current depth-map.
    """

    def __init__(
        self, map_size=31, map_scale=0.1, position=[0, 1.25, 0], depth_sensor_hfov=90,
        height_thresh=(0.2, 1.5), depth_sensor_min_depth=0, depth_sensor_max_depth=10,
        depth_sensor_width=128, depth_sensor_height=128, depth_sensor_normalize_depth=False,
    ):
        """
        Estimates the top-down occupancy based on current depth-map.
        :param map_size: size of map
        :param map_scale: scale at which the map will be computed
        :param position: agent position
        :param depth_sensor_hfov: depth sensor horizontal FoV
        :param height_thresh: height threshold for computing occupancy
        :param depth_sensor_min_depth: depth sensor minimum height
        :param depth_sensor_max_depth: depth sensor maximum height
        :param depth_sensor_width: depth sensor width
        :param depth_sensor_height: depth sensor height
        :param depth_sensor_normalize_depth: flag saying if depth sensor output is normalized or not
        """

        # depth sensor attris
        self.depth_sensor_normalize_depth = depth_sensor_normalize_depth

        # Map statistics
        self.map_size = map_size
        self.map_scale = map_scale

        # Agent height for pointcloud transformation
        self.sensor_height = position[1]

        # Compute intrinsic matrix
        hfov = float(depth_sensor_hfov) * np.pi / 180
        vfov = 2 * np.arctan((depth_sensor_height / depth_sensor_width) * np.tan(hfov / 2.0))
        self.intrinsic_matrix = np.array([[1 / np.tan(hfov / 2.), 0., 0., 0.],
                                          [0., 1 / np.tan(vfov / 2.), 0., 0.],
                                          [0., 0.,  1, 0],
                                          [0., 0., 0, 1]])
        self.inverse_intrinsic_matrix = np.linalg.inv(self.intrinsic_matrix)

        # Height thresholds for obstacles
        self.height_thresh = height_thresh

        # Depth processing
        self.min_depth = float(depth_sensor_min_depth)
        self.max_depth = float(depth_sensor_max_depth)

        # Pre-compute a grid of locations for depth projection
        W = depth_sensor_width
        H = depth_sensor_height
        self.proj_xs, self.proj_ys = np.meshgrid(
                                          np.linspace(-1, 1, W),
                                          np.linspace(1, -1, H)
                                     )

    def convert_to_pointcloud(self, depth):
        """
        compute point cloud from a depth image
        :param depth: input depth image
        :return: computed point cloud
        """

        depth_float = depth.astype(np.float32)[..., 0]

        # =========== Convert to camera coordinates ============
        W = depth.shape[1]
        xs = np.copy(self.proj_xs).reshape(-1)
        ys = np.copy(self.proj_ys).reshape(-1)
        depth_float = depth_float.reshape(-1)
        # Filter out invalid depths
        max_forward_range = self.map_size * self.map_scale
        valid_depths = (depth_float != 0.0) & (depth_float <= max_forward_range)
        # print(valid_depths.shape, valid_depths)
        xs = xs[valid_depths]
        ys = ys[valid_depths]
        depth_float = depth_float[valid_depths]
        # Unproject
        # negate depth as the camera looks along -Z
        xys = np.vstack((xs * depth_float,
                         ys * depth_float,
                         -depth_float, np.ones(depth_float.shape)))
        inv_K = self.inverse_intrinsic_matrix
        xyz_camera = np.matmul(inv_K, xys).T # XYZ in the camera coordinate system
        xyz_camera = xyz_camera[:, :3] / xyz_camera[:, 3][:, np.newaxis]

        return xyz_camera

    def safe_assign(self, im_map, x_idx, y_idx, value):
        """safe assigning a map cell with point cloud projection"""
        try:
            im_map[x_idx, y_idx] = value
        except IndexError:
            valid_idx1 = np.logical_and(x_idx >= 0, x_idx < im_map.shape[0])
            valid_idx2 = np.logical_and(y_idx >= 0, y_idx < im_map.shape[1])
            valid_idx = np.logical_and(valid_idx1, valid_idx2)
            im_map[x_idx[valid_idx], y_idx[valid_idx]] = value

    def _get_depth_projection(self, sim_depth):
        """
        Project pixels visible in depth-map to ground-plane
        """

        if self.depth_sensor_normalize_depth:
            depth = sim_depth * (self.max_depth - self.min_depth) + self.min_depth
        else:
            depth = sim_depth

        XYZ_ego = self.convert_to_pointcloud(depth)

        # Adding agent's height to the pointcloud
        XYZ_ego[:, 1] += self.sensor_height

        # Convert to grid coordinate system
        V = self.map_size
        Vby2 = V // 2

        points = XYZ_ego

        grid_x = (points[:, 0] / self.map_scale) + Vby2
        grid_y = (points[:, 2] / self.map_scale) + V

        # Filter out invalid points
        valid_idx = (grid_x >= 0) & (grid_x <= V-1) & (grid_y >= 0) & (grid_y <= V-1)
        points = points[valid_idx, :]
        grid_x = grid_x[valid_idx].astype(int)
        grid_y = grid_y[valid_idx].astype(int)

        # Create empty maps for the two channels
        obstacle_mat = np.zeros((self.map_size, self.map_size), np.uint8)
        explore_mat = np.zeros((self.map_size, self.map_size), np.uint8)

        # Compute obstacle locations
        high_filter_idx = points[:, 1] < self.height_thresh[1]
        low_filter_idx = points[:, 1] > self.height_thresh[0]
        obstacle_idx = np.logical_and(low_filter_idx, high_filter_idx)

        self.safe_assign(obstacle_mat, grid_y[obstacle_idx], grid_x[obstacle_idx], 1)
        kernel = np.ones((3, 3), np.uint8)
        obstacle_mat = cv2.dilate(obstacle_mat, kernel, iterations=1)

        # Compute explored locations
        explored_idx = high_filter_idx
        self.safe_assign(explore_mat, grid_y[explored_idx], grid_x[explored_idx], 1)
        kernel = np.ones((3, 3), np.uint8)
        explore_mat = cv2.dilate(explore_mat, kernel, iterations=1)

        # Smoothen the maps
        kernel = np.ones((3, 3), np.uint8)

        obstacle_mat = cv2.morphologyEx(obstacle_mat, cv2.MORPH_CLOSE, kernel)
        explore_mat = cv2.morphologyEx(explore_mat, cv2.MORPH_CLOSE, kernel)

        # Ensure all expanded regions in obstacle_mat are accounted for in explored_mat
        explore_mat = np.logical_or(explore_mat, obstacle_mat)

        return np.stack([obstacle_mat, explore_mat], axis=2)

    def get_observation(
        self, depth_img,
    ) -> object:
        # convert to numpy array
        sim_depth = np.expand_dims(asnumpy(depth_img), axis=-1)
        ego_map_gt = self._get_depth_projection(sim_depth)

        return ego_map_gt


class PassiveMappingDataset(Dataset):
    """Passive mapping dataset"""

    def __init__(self, split="train", all_scenes_graphs_this_split=None, cfg=None, all_scenes_observations=None,
                 eval_mode=False,):
        """
        Creates an instance of the passive mapping dataset
        :param split: dataset split
        :param all_scenes_graphs_this_split: all scene graphs present in this split
        :param cfg: config
        :param all_scenes_observations: all scene observations
        :param eval_mode: flag saying if in eval mode or not
        """

        assert split in ["train", "val", "test"]

        self.split = split
        self.config = cfg
        task_cfg = cfg.TASK_CONFIG
        self.sim_cfg = task_cfg.SIMULATOR
        self.env_cfg = task_cfg.ENVIRONMENT
        self.task_cfg = task_cfg.TASK
        self.audio_cfg = self.sim_cfg.AUDIO
        self.passive_mapping_cfg = cfg.PassiveMapping

        self._eval_mode = eval_mode

        self.scene_dataset = self.sim_cfg.SCENE_DATASET
        self.rir_sampling_rate = self.audio_cfg.RIR_SAMPLING_RATE
        self._max_valid_impulse_length = self.audio_cfg.MAX_VALID_IMPULSE_LENGTH_AFTER_REMOVING_LEADING_ZEROS
        self._anechoic_audio_slice_length = self.audio_cfg.ANECHOIC_AUDIO_SLICE_LENGTH
        self.hop_length = self.audio_cfg.HOP_LENGTH
        self.n_fft = self.audio_cfg.N_FFT
        self.win_length = self.audio_cfg.WIN_LENGTH

        self._audio_wav_shape = self.task_cfg.AMBI_WAV_SENSOR.FEATURE_SHAPE

        print(f"LOADING ANECHOIC AUDIO FOR {split}")
        if split == "train":
            anechoic_audio_dir = self.audio_cfg.ANECHOIC_DIR
        elif split == "val":
            anechoic_audio_dir = self.audio_cfg.VAL_UNHEARD_ANECHOIC_DIR
        elif split == "test":
            anechoic_audio_dir = self.audio_cfg.TEST_UNHEARD_ANECHOIC_DIR
        else:
            raise ValueError
        assert os.path.isdir(anechoic_audio_dir)

        anechoic_audio_filenames = os.listdir(anechoic_audio_dir)

        self._anechoic_filename_2_audioData = {}
        for anechoic_audio_filename in tqdm(anechoic_audio_filenames):
            anechoic_audio_filePath = os.path.join(anechoic_audio_dir, anechoic_audio_filename)
            assert os.path.isfile(anechoic_audio_filePath)

            anechoic_audioSR, anechoic_audioData = wavfile.read(anechoic_audio_filePath)
            assert anechoic_audioSR == self.rir_sampling_rate

            assert anechoic_audio_filename.split(".")[0] not in self._anechoic_filename_2_audioData
            self._anechoic_filename_2_audioData[anechoic_audio_filename.split(".")[0]] = anechoic_audioData

        self._pose_feat_shape = [3]
        self._add_truncated_gaussian_pose_noise = self.task_cfg.CONTEXT_VIEW_POSE_SENSOR.ADD_TRUNCATED_GAUSSIAN_NOISE\
                                                  and (split in ["train", "test"])
        self._truncated_gaussian_pose_noise_cfg = self.task_cfg.CONTEXT_VIEW_POSE_SENSOR.TRUNCATED_GAUSSIAN_NOISE

        self._gaussian_pose_noise_multipliers = None
        if self._add_truncated_gaussian_pose_noise and (split in ["test"]):
            assert os.path.isfile(self._truncated_gaussian_pose_noise_cfg.GAUSSIAN_NOISE_MULTIPLIERS_PATH)
            with open(self._truncated_gaussian_pose_noise_cfg.GAUSSIAN_NOISE_MULTIPLIERS_PATH, "rb") as fi:
                self._gaussian_pose_noise_multipliers = pickle.load(fi)

        self.max_context_length = self.env_cfg.MAX_CONTEXT_LENGTH
        self.max_query_length = self.env_cfg.MAX_QUERY_LENGTH

        self.episodes = []
        self.dataset_dir = None
        if split == "train":
            self.dataset_dir = task_cfg.PASSIVE_SAMPLING_DATASET.TRAIN_DATASET_DIR
            assert os.path.isdir(self.dataset_dir)
            self.dataset_dir = os.path.join(self.dataset_dir, "content")
            assert os.path.isdir(self.dataset_dir)

            print("LOADING TRAIN EPISODES")
            for scene in tqdm(all_scenes_graphs_this_split):
                sceneEpisodes_path = os.path.join(self.dataset_dir, f"{scene}.json.gz")
                assert os.path.isfile(sceneEpisodes_path)
                self.episodes += self._get_episodes_given_jsonGz(sceneEpisodes_path)
            print("TRAIN EPISODE LOADING COMPLETE")

            self.num_datapoints_per_epoch = self.passive_mapping_cfg.num_datapoints_train
        elif split in ["val", "test"]:
            if split == "val":
                self.dataset_dir = task_cfg.PASSIVE_SAMPLING_DATASET.VAL_DATASET_DIR
            elif split == "test":
                self.dataset_dir = task_cfg.PASSIVE_SAMPLING_DATASET.TEST_DATASET_DIR

            print(f"LOADING {self.split.upper()} EPISODES")
            if self.dataset_dir[-1] == "/":
                self.dataset_dir = self.dataset_dir[:-1]
            sceneEpsiodes_path = os.path.join(self.dataset_dir, f"{self.dataset_dir.split('/')[-1]}.json.gz")
            assert os.path.isfile(sceneEpsiodes_path)
            self.episodes = self._get_episodes_given_jsonGz(sceneEpsiodes_path)
            print(f"{self.split.upper()} EPISODE LOADING COMPLETE")

            if eval_mode and (split == "test"):
                self.num_datapoints_per_epoch = 1000
            else:
                self.num_datapoints_per_epoch = len(self.episodes)

        self._all_scenes_graphs_this_split = all_scenes_graphs_this_split

        assert all_scenes_observations is not None
        self.all_scenes_observations = all_scenes_observations

        self.render_local_ego_occ_maps_from_depth_images = self.sim_cfg.RENDER_LOCAL_EGO_OCC_MAPS_FROM_DEPTH_IMAGES and\
                                                           (split in ["train", "test"])
        self.local_occMap_cfg = self.sim_cfg.LOCAL_MAP
        self.ego_mapper = None
        self.redwood_depth_noise_dist_model = None
        self.redwood_depth_noise_multiplier = None
        if self.render_local_ego_occ_maps_from_depth_images:
            assert "DEPTH_SENSOR" in self.config.SENSORS
            self.ego_mapper = EgoMap(
                map_size=self.local_occMap_cfg.SIZE,
                map_scale=self.local_occMap_cfg.SCALE,
                position=self.local_occMap_cfg.AGENT_POSITION,
                depth_sensor_hfov=self.local_occMap_cfg.HFOV_DEPTH_IMG,
                height_thresh=self.local_occMap_cfg.HEIGHT_THRESH,
                depth_sensor_min_depth=self.local_occMap_cfg.MIN_DEPTH,
                depth_sensor_max_depth=self.local_occMap_cfg.MAX_DEPTH,
                depth_sensor_width=self.local_occMap_cfg.WIDTH_DEPTH_IMG,
                depth_sensor_height=self.local_occMap_cfg.HEIGHT_DEPTH_IMG,
                depth_sensor_normalize_depth=self.local_occMap_cfg.NORMALIZE_DEPTH_IMG,
            )

            if self.sim_cfg.DEPTH_SENSOR.ADD_REDWOOD_NOISE:
                """src: https://github.com/facebookresearch/habitat-sim/blob/main/src_python/habitat_sim/sensors/noise_models/redwood_depth_noise_model.py"""
                assert os.path.isfile(self.sim_cfg.DEPTH_SENSOR.REDWOOD_DEPTH_NOISE_DIST_MODEL)

                self.redwood_depth_noise_dist_model = np.load(self.sim_cfg.DEPTH_SENSOR.REDWOOD_DEPTH_NOISE_DIST_MODEL)
                self.redwood_depth_noise_dist_model = self.redwood_depth_noise_dist_model.reshape(80, 80, 5)

                self.redwood_depth_noise_multiplier = self.sim_cfg.DEPTH_SENSOR.REDWOOD_NOISE_MULTIPLIER

                self._redwood_depth_noise_rand_nums = None
                if split == "test":
                    assert os.path.isfile(self.sim_cfg.DEPTH_SENSOR.REDWOOD_NOISE_RAND_NUMS_PATH)
                    with open(self.sim_cfg.DEPTH_SENSOR.REDWOOD_NOISE_RAND_NUMS_PATH, "rb") as fi:
                        self._redwood_depth_noise_rand_nums = pickle.load(fi)

        assert os.path.isdir(self.sim_cfg.RENDERED_LOCAL_EGO_OCC_MAPS_DIR)
        self.all_scenes_local_ego_occ_maps = dict()
        print("LOADING CACHED LOCAL EGO OCC MAPS")
        for scene in tqdm(self.all_scenes_observations.keys()):
            local_ego_occ_maps_file_path = os.path.join(self.sim_cfg.RENDERED_LOCAL_EGO_OCC_MAPS_DIR, f"{scene}.pkl")
            with open(local_ego_occ_maps_file_path, "rb") as fi:
                self.all_scenes_local_ego_occ_maps[scene] = pickle.load(fi)

        assert os.path.isdir(self.sim_cfg.RENDERED_GT_GLOBAL_CANONICAL_OCC_MAP_EGO_CROP_N_ROTS_DIR)
        self.all_scenes_gt_global_canonical_occ_ego_crops_n_rots_maps = dict()
        print("LOADING CACHED GT GLOBAL CANONICAL OCC MAPS")
        for scene in tqdm(self.all_scenes_observations.keys()):
            gt_global_can_occ_map_ego_crop_n_rots_path =\
                os.path.join(self.sim_cfg.RENDERED_GT_GLOBAL_CANONICAL_OCC_MAP_EGO_CROP_N_ROTS_DIR, f"{scene}.pkl")
            with open(gt_global_can_occ_map_ego_crop_n_rots_path, "rb") as fi:
                self.all_scenes_gt_global_canonical_occ_ego_crops_n_rots_maps[scene] = pickle.load(fi)

        self.stitch_top_down_maps = self.config.STITCH_TOP_DOWN_MAPS and eval_mode

        assert "RGB_SENSOR" in self.config.SENSORS

        self.rir_dir = self.audio_cfg.RIR_DIR
        assert os.path.isdir(self.rir_dir)

        self.scenes_in_split = list(all_scenes_graphs_this_split.keys())

    def _get_episodes_given_jsonGz(self, jsonGz_path):
        """get conversation episodes"""
        with gzip.GzipFile(jsonGz_path, "rb") as fo:
            dataset = fo.read()
        dataset = dataset.decode("utf-8")
        dataset = json.loads(dataset)
        dataset_episodes = dataset["episodes"]

        return dataset_episodes

    def __len__(self):
        return self.num_datapoints_per_epoch

    def __getitem__(self, item):
        this_datapoint = self._get_datapoint(item)

        context_rgbs_this_datapoint = torch.from_numpy(this_datapoint["context"]["rgbs"])
        context_egoLocalOccMaps_this_datapoint = torch.from_numpy(this_datapoint["context"]["egoLocalOccMaps"])
        if self.stitch_top_down_maps:
            context_stitched_egoLocalOccMaps_this_datapoint = torch.from_numpy(this_datapoint["context"]["stitched_egoLocalOccMaps"])
            context_ref_rAz_this_datapoint = torch.from_numpy(this_datapoint["context"]["ref_rAz"])
        context_views_pose_this_datapoint = torch.from_numpy(this_datapoint["context"]["views_pose"])
        context_views_mask_this_datapoint = torch.from_numpy(this_datapoint["context"]["views_mask"])

        context_selfAudio_this_datapoint = torch.from_numpy(this_datapoint["context"]["selfAudio"])
        context_selfAudio_pose_this_datapoint = torch.from_numpy(this_datapoint["context"]["selfAudio_pose"])
        context_selfAudio_mask_this_datapoint = torch.from_numpy(this_datapoint["context"]["selfAudio_mask"])

        context_otherAudio_this_datapoint = torch.from_numpy(this_datapoint["context"]["otherAudio"])
        context_otherAudio_pose_this_datapoint = torch.from_numpy(this_datapoint["context"]["otherAudio_pose"])
        context_otherAudio_mask_this_datapoint = torch.from_numpy(this_datapoint["context"]["otherAudio_mask"])

        query_globalCanOccMaps_egoCrops_gt_this_datapoint = torch.from_numpy(this_datapoint["query"]["globalCanOccMaps_egoCrops_gt"])
        query_globalCanOccMaps_egoCrops_exploredMasks_this_datapoint =\
            torch.from_numpy(this_datapoint["query"]["globalCanOccMaps_egoCrops_exploredMasks"])
        if self.stitch_top_down_maps:
            query_stitched_globalCanOccMaps_egoCrops_gt_this_datapoint =\
                torch.from_numpy(this_datapoint["query"]["stitched_globalCanOccMaps_egoCrops_gt"])
        query_views_pose_this_datapoint = torch.from_numpy(this_datapoint["query"]["views_pose"])
        query_views_mask_this_datapoint = torch.from_numpy(this_datapoint["query"]["views_mask"])

        query_sceneIdxs_this_datapoint = torch.from_numpy(this_datapoint["query"]["scene_idxs"])
        if self._eval_mode:
            query_rAzs_this_datapoint = torch.from_numpy(this_datapoint["query"]["rAzs"])
            query_epIdxs_this_datapoint = torch.from_numpy(this_datapoint["query"]["ep_idxs"])

        if self._eval_mode:
            rtrn_lst = [context_egoLocalOccMaps_this_datapoint, context_rgbs_this_datapoint, context_views_pose_this_datapoint,
                        context_views_mask_this_datapoint, context_selfAudio_this_datapoint, context_selfAudio_pose_this_datapoint,
                        context_selfAudio_mask_this_datapoint, context_otherAudio_this_datapoint, context_otherAudio_pose_this_datapoint,
                        context_otherAudio_mask_this_datapoint,  query_globalCanOccMaps_egoCrops_gt_this_datapoint,
                        query_globalCanOccMaps_egoCrops_exploredMasks_this_datapoint, query_views_pose_this_datapoint,
                        query_views_mask_this_datapoint,  query_sceneIdxs_this_datapoint, query_rAzs_this_datapoint,
                        query_epIdxs_this_datapoint]

            if self.stitch_top_down_maps:
                rtrn_lst += [context_stitched_egoLocalOccMaps_this_datapoint,
                             context_ref_rAz_this_datapoint,
                             query_stitched_globalCanOccMaps_egoCrops_gt_this_datapoint]

        else:
            rtrn_lst = [context_egoLocalOccMaps_this_datapoint, context_rgbs_this_datapoint, context_views_pose_this_datapoint,
                        context_views_mask_this_datapoint, context_selfAudio_this_datapoint, context_selfAudio_pose_this_datapoint,
                        context_selfAudio_mask_this_datapoint, context_otherAudio_this_datapoint, context_otherAudio_pose_this_datapoint,
                        context_otherAudio_mask_this_datapoint, query_globalCanOccMaps_egoCrops_gt_this_datapoint,
                        query_globalCanOccMaps_egoCrops_exploredMasks_this_datapoint, query_views_pose_this_datapoint,
                        query_views_mask_this_datapoint, query_sceneIdxs_this_datapoint]

        return rtrn_lst

    def _get_datapoint(self, item_):
        """get datapoint"""

        if self.split == "train":
            item_ = torch.randint(0, len(self.episodes), size=(1,)).item()
            episode = self.episodes[item_]
        else:
            assert item_ < len(self.episodes)
            episode = self.episodes[item_]

        datapoint_scene = episode['scene_id'].split("/")[0]
        datapoint_episodeID = episode['episode_id']

        assert len(episode['other_starts']) == 1, "more than a total of 2 agents not implemented"     
        num_agents = self.sim_cfg.ALL_AGENTS.NUM
        assert 1 <= num_agents <= len(episode['other_starts']) + 1
        total_context_length = len(episode['info']['nodes_n_azimuths'])
        assert total_context_length >= self.max_context_length

        allAgents_context_allPoses = [episode['info']['nodes_n_azimuths']]
        for otherAgent_idx in range(num_agents - 1):
            allAgents_context_allPoses.append(episode['other_info'][otherAgent_idx]['nodes_n_azimuths'])

        num_total_utteranceSwitches = 3
        if (self.split == "train") or ('utterance_switches' not in episode['info']):
            agent_utterance_allSwitches = torch.randint(num_total_utteranceSwitches,
                                                        size=(total_context_length,)).numpy().tolist()
        else:
            agent_utterance_allSwitches = episode['info']['utterance_switches']
            assert np.max(agent_utterance_allSwitches) <= num_total_utteranceSwitches - 1

        lst_anechoicAudio_filenameNstartSamplingIdx = []
        if self.split == "train":
            assert num_agents <= len(self._anechoic_filename_2_audioData)
            lst_anechoic_filename_idxs = torch.randperm(len(self._anechoic_filename_2_audioData)).tolist()[:num_agents]
            for anechoic_filename_idx in lst_anechoic_filename_idxs:
                train_anechoic_filename_thisAgent = list(self._anechoic_filename_2_audioData.keys())[anechoic_filename_idx]
                train_anechoic_fileStartSampleIdx_thisAgent =\
                    torch.randint(0, self._anechoic_filename_2_audioData[train_anechoic_filename_thisAgent].shape[0], size=(1,)).item()
                lst_anechoicAudio_filenameNstartSamplingIdx.append((train_anechoic_filename_thisAgent, train_anechoic_fileStartSampleIdx_thisAgent))
        else:
            for agent_idx in range(num_agents):
                if agent_idx == 0:
                    assert "anechoicSound_fileName" in episode["info"]
                    anechoicSound_fileName = episode["info"]["anechoicSound_fileName"]

                    assert "anechoicSound_samplingStartIdx" in episode["info"]
                    anechoicSound_samplingStartIdx = episode["info"]["anechoicSound_samplingStartIdx"]
                else:
                    assert "anechoicSound_fileName" in episode["other_info"][agent_idx - 1]
                    anechoicSound_fileName = episode["other_info"][agent_idx - 1]["anechoicSound_fileName"]

                    assert "anechoicSound_samplingStartIdx" in episode["other_info"][agent_idx - 1]
                    anechoicSound_samplingStartIdx = episode["other_info"][agent_idx - 1]["anechoicSound_samplingStartIdx"]

                lst_anechoicAudio_filenameNstartSamplingIdx.append((anechoicSound_fileName, anechoicSound_samplingStartIdx))

        self._rAz_2_noise_dlX_dlZ_dlAz = None
        if self._add_truncated_gaussian_pose_noise:
            self._rAz_2_noise_dlX_dlZ_dlAz = {}
            if self.split == "test":
                assert (str(datapoint_scene), str(datapoint_episodeID)) in self._gaussian_pose_noise_multipliers
                gaussian_pose_noise_multipliers_thsScnEpID = self._gaussian_pose_noise_multipliers[(str(datapoint_scene), str(datapoint_episodeID))]
            for agent_idx in range(len(allAgents_context_allPoses)):
                contextIdx_tstPsNs = 0
                for context_pose in allAgents_context_allPoses[agent_idx]:
                    if tuple(context_pose) not in self._rAz_2_noise_dlX_dlZ_dlAz:
                        if self.split == "test":
                            assert contextIdx_tstPsNs < len(gaussian_pose_noise_multipliers_thsScnEpID)
                            gaussian_multipler_pose_x = float(gaussian_pose_noise_multipliers_thsScnEpID[contextIdx_tstPsNs][agent_idx][0])
                        else:
                            gaussian_multipler_pose_x = torch.randn(1).item()
                        del_x = gaussian_multipler_pose_x * self._truncated_gaussian_pose_noise_cfg.TRANSLATION_STD +\
                                self._truncated_gaussian_pose_noise_cfg.TRANSLATION_MEAN
                        del_x = max(self._truncated_gaussian_pose_noise_cfg.TRANSLATION_MEAN -\
                                    abs(self._truncated_gaussian_pose_noise_cfg.NUM_STDS_FOR_TRUNCATION)\
                                    * self._truncated_gaussian_pose_noise_cfg.TRANSLATION_STD, del_x)
                        del_x = min(self._truncated_gaussian_pose_noise_cfg.TRANSLATION_MEAN +\
                                    abs(self._truncated_gaussian_pose_noise_cfg.NUM_STDS_FOR_TRUNCATION)\
                                    * self._truncated_gaussian_pose_noise_cfg.TRANSLATION_STD, del_x)

                        if self.split == "test":
                            gaussian_multipler_pose_z = float(gaussian_pose_noise_multipliers_thsScnEpID[contextIdx_tstPsNs][agent_idx][1])
                        else:
                            gaussian_multipler_pose_z = torch.randn(1).item()
                        del_z = gaussian_multipler_pose_z * self._truncated_gaussian_pose_noise_cfg.TRANSLATION_STD +\
                                self._truncated_gaussian_pose_noise_cfg.TRANSLATION_MEAN
                        del_z = max(self._truncated_gaussian_pose_noise_cfg.TRANSLATION_MEAN -\
                                    abs(self._truncated_gaussian_pose_noise_cfg.NUM_STDS_FOR_TRUNCATION)\
                                    * self._truncated_gaussian_pose_noise_cfg.TRANSLATION_STD, del_z)
                        del_z = min(self._truncated_gaussian_pose_noise_cfg.TRANSLATION_MEAN +\
                                    abs(self._truncated_gaussian_pose_noise_cfg.NUM_STDS_FOR_TRUNCATION)\
                                    * self._truncated_gaussian_pose_noise_cfg.TRANSLATION_STD, del_z)

                        if self.split == "test":
                            gaussian_multipler_pose_az = float(gaussian_pose_noise_multipliers_thsScnEpID[contextIdx_tstPsNs][agent_idx][2])
                        else:
                            gaussian_multipler_pose_az = torch.randn(1).item()
                        del_az = gaussian_multipler_pose_az * self._truncated_gaussian_pose_noise_cfg.ROTATION_DEGREES_STD +\
                                 self._truncated_gaussian_pose_noise_cfg.ROTATION_DEGREES_MEAN
                        del_az = max(self._truncated_gaussian_pose_noise_cfg.ROTATION_DEGREES_MEAN -\
                                     abs(self._truncated_gaussian_pose_noise_cfg.NUM_STDS_FOR_TRUNCATION)\
                                     * self._truncated_gaussian_pose_noise_cfg.ROTATION_DEGREES_STD, del_az)
                        del_az = min(self._truncated_gaussian_pose_noise_cfg.ROTATION_DEGREES_MEAN +\
                                     abs(self._truncated_gaussian_pose_noise_cfg.NUM_STDS_FOR_TRUNCATION)\
                                     * self._truncated_gaussian_pose_noise_cfg.ROTATION_DEGREES_STD, del_az)

                        self._rAz_2_noise_dlX_dlZ_dlAz[tuple(context_pose)] = [del_x, del_z, del_az]

                    contextIdx_tstPsNs += 1

        if self.sim_cfg.DEPTH_SENSOR.ADD_REDWOOD_NOISE:
            redwood_depth_noise_rand_nums_thsScnEpID = None
            if self.split == "test":
                assert self._redwood_depth_noise_rand_nums is not None
                assert (str(datapoint_scene), str(datapoint_episodeID)) in self._redwood_depth_noise_rand_nums
                redwood_depth_noise_rand_nums_thsScnEpID = self._redwood_depth_noise_rand_nums[(str(datapoint_scene), str(datapoint_episodeID))]

        if self.split in ["train"]:
            context_length = torch.randint(1, self.max_context_length + 1, size=(1,)).item()
        else:
            context_length = self.max_context_length

        allAgents_context_poses = []
        for agent_idx in range(len(allAgents_context_allPoses)):
            # print(agent_idx, len(allAgents_context_allPoses), len(allAgents_context_allPoses[agent_idx]))
            allAgents_context_poses.append(allAgents_context_allPoses[agent_idx][:context_length])
        agent_utterance_switches = agent_utterance_allSwitches[:context_length]

        query_allPoses = []
        for agent_idx in range(len(allAgents_context_poses)):
            for context_pose in allAgents_context_poses[agent_idx]:
                if context_pose not in query_allPoses:
                    query_allPoses.append(context_pose)

        query_length = min(self.max_query_length, len(query_allPoses))
        query_poses = query_allPoses[:query_length]

        datapoint = {}

        """views (context + query)"""
        assert sorted(self.config.SENSORS) == sorted(["RGB_SENSOR", "DEPTH_SENSOR"])

        assert self.sim_cfg.RGB_SENSOR.HEIGHT == self.sim_cfg.DEPTH_SENSOR.HEIGHT
        view_sensor_height = self.sim_cfg.RGB_SENSOR.HEIGHT

        assert self.sim_cfg.RGB_SENSOR.WIDTH == self.sim_cfg.DEPTH_SENSOR.WIDTH
        view_sensor_width = self.sim_cfg.RGB_SENSOR.WIDTH

        allAgents_context_rgbs_ph = np.zeros((num_agents,
                                              self.max_context_length,
                                              view_sensor_height,
                                              view_sensor_width,
                                              3)).astype("float32")

        allAgents_context_egoLocalOccMaps_ph = np.zeros((num_agents,
                                                         self.max_context_length,
                                                         self.sim_cfg.EGO_LOCAL_OCC_MAP.SIZE,
                                                         self.sim_cfg.EGO_LOCAL_OCC_MAP.SIZE,
                                                         self.sim_cfg.EGO_LOCAL_OCC_MAP.NUM_CHANNELS)).astype("float32")
        if self.stitch_top_down_maps:
            assert self.task_cfg.CONTEXT_EGO_LOCAL_MAP_SENSOR.NUM_CHANNELS == 2

            allAgents_context_stitchedEgoLocalOccMaps_ph = np.zeros((
                                                         self.sim_cfg.EGO_STITCHED_GLOBAL_CANONICAL_OCC_MAP.SIZE,
                                                         self.sim_cfg.EGO_STITCHED_GLOBAL_CANONICAL_OCC_MAP.SIZE,
                                                         self.task_cfg.CONTEXT_EGO_LOCAL_MAP_SENSOR.NUM_CHANNELS)).astype("float32")

        query_gt_globCanOccMaps_egoCrops_ph = np.zeros((self.max_query_length,
                                                        self.sim_cfg.GT_GLOBAL_CANONICAL_OCC_MAP_EGO_CROP.SIZE,
                                                        self.sim_cfg.GT_GLOBAL_CANONICAL_OCC_MAP_EGO_CROP.SIZE,
                                                        self.sim_cfg.GT_GLOBAL_CANONICAL_OCC_MAP_EGO_CROP.NUM_CHANNELS)).astype("float32")

        query_globCanOccMaps_egoCrop_exploredMasks_ph = np.zeros((self.max_query_length,
                                                              self.sim_cfg.GT_GLOBAL_CANONICAL_OCC_MAP_EGO_CROP.SIZE,
                                                              self.sim_cfg.GT_GLOBAL_CANONICAL_OCC_MAP_EGO_CROP.SIZE,
                                                              self.sim_cfg.GT_GLOBAL_CANONICAL_OCC_MAP_EGO_CROP.NUM_CHANNELS)).astype("float32")

        if self.stitch_top_down_maps:
            query_stitchedGtGlobCanOccMapsEgoCrops_ph = np.zeros((
                                                         self.sim_cfg.EGO_STITCHED_GLOBAL_CANONICAL_OCC_MAP.SIZE,
                                                         self.sim_cfg.EGO_STITCHED_GLOBAL_CANONICAL_OCC_MAP.SIZE,
                                                         1)).astype("float32")

        """poses (context + query)"""
        allAgents_context_onlyRRelPoses_ph = np.zeros((num_agents,
                                                       self.max_context_length,
                                                       5)).astype("float32")
        query_relPoses_ph = np.zeros((self.max_query_length,
                                      5)).astype("float32")
        allAgents_context_srRelPoses_ph = np.zeros((num_agents,
                                                    self.max_context_length,
                                                    5)).astype("float32")

        """mask (context + query)"""
        allAgents_context_viewsMask_ph = np.zeros((num_agents,
                                                   self.max_context_length)).astype("float32")
        allAgents_context_selfAudioMask_ph = np.zeros((num_agents,
                                                      self.max_context_length)).astype("float32")
        allAgents_context_otherAudioMask_ph = np.zeros((num_agents,
                                                       self.max_context_length)).astype("float32")
        query_mask_ph = np.zeros(self.max_query_length).astype("float32")

        """audio (context)"""
        allAgents_context_selfAudio_ph = np.zeros((num_agents,
                                                  self.max_context_length,
                                                  self._audio_wav_shape[0],
                                                  self._audio_wav_shape[1])).astype("float32")

        allAgents_context_otherAudio_ph = np.zeros((num_agents,
                                                   self.max_context_length,
                                                   self._audio_wav_shape[0],
                                                   self._audio_wav_shape[1])).astype("float32")

        """eval datapoint sceneIdx and rAz (query)"""
        query_sceneIdxs_ph = np.zeros(self.max_query_length, dtype="int32")
        if self._eval_mode:
            query_epIdxs_ph = np.zeros(1, dtype="int32")
            query_epIdxs_ph[0] = int(datapoint_episodeID)
            query_rAz_ph = np.zeros((self.max_query_length, 2), dtype="int32")

        assert context_length >= 1, "can't compute relative query pose if there isn't at least 1 valid entry in context"
        ref_pose_for_computing_rel_pose = allAgents_context_poses[0][0]
        if self.stitch_top_down_maps:
            ref_rAz_ph = np.zeros(2, dtype="int32")
            ref_rAz_ph[0] = ref_pose_for_computing_rel_pose[0]
            ref_rAz_ph[1] = ref_pose_for_computing_rel_pose[1]

        for context_idx in range(context_length):
            for agent_idx in range(num_agents):
                """context views"""
                context_rgb =\
                    self.all_scenes_observations[datapoint_scene][(allAgents_context_poses[agent_idx][context_idx][0],
                                                                   self._compute_rotation_from_azimuth(allAgents_context_poses[agent_idx][context_idx][1]))]["rgb"][:, :, :3]
                context_rgb = context_rgb.astype("float32")
                allAgents_context_rgbs_ph[agent_idx][context_idx] = context_rgb

                if self.render_local_ego_occ_maps_from_depth_images:
                    context_depth = self.all_scenes_observations[datapoint_scene][(allAgents_context_poses[agent_idx][context_idx][0],
                                                                                   self._compute_rotation_from_azimuth(allAgents_context_poses[agent_idx][context_idx][1]))]["depth"]
                    if self.sim_cfg.DEPTH_SENSOR.ADD_REDWOOD_NOISE and (self.split in ["train"]):
                        if self.split == "test":
                            # raise NotImplementedError
                            assert redwood_depth_noise_rand_nums_thsScnEpID is not None
                            assert context_idx < len(redwood_depth_noise_rand_nums_thsScnEpID)
                            rand_nums = redwood_depth_noise_rand_nums_thsScnEpID[context_idx][agent_idx]
                        else:
                            rand_nums =\
                                torch.randn((self.sim_cfg.DEPTH_SENSOR.HEIGHT, self.sim_cfg.DEPTH_SENSOR.WIDTH, 3)).numpy().astype(np.float32)

                        context_depth = simulate_redwood_depth_noise(
                            context_depth,
                            self.redwood_depth_noise_dist_model,
                            self.redwood_depth_noise_multiplier,
                            rand_nums,
                        )

                    context_egoLocalOccMap = self.ego_mapper.get_observation(context_depth)

                    assert not np.any(np.logical_and(context_egoLocalOccMap[..., 0] != 1, context_egoLocalOccMap[..., 0] != 0))
                    assert not np.any(np.logical_and(context_egoLocalOccMap[..., 1] != 1, context_egoLocalOccMap[..., 1] != 0))
                else:
                    context_egoLocalOccMap = self.all_scenes_local_ego_occ_maps[datapoint_scene][(allAgents_context_poses[agent_idx][context_idx][0],
                                                                                                  self._compute_rotation_from_azimuth(allAgents_context_poses[agent_idx][context_idx][1]))].astype("float32")
                if self.sim_cfg.EGO_LOCAL_OCC_MAP.NUM_CHANNELS == 1:
                    context_egoLocalOccMap = context_egoLocalOccMap[..., :1]
                elif self.sim_cfg.EGO_LOCAL_OCC_MAP.NUM_CHANNELS == 2:
                    pass
                else:
                    raise ValueError

                allAgents_context_egoLocalOccMaps_ph[agent_idx][context_idx] = context_egoLocalOccMap

                """context view mask"""
                allAgents_context_viewsMask_ph[agent_idx][context_idx] = 1
                if self.stitch_top_down_maps:
                    allAgents_context_stitchedEgoLocalOccMaps_ph =\
                        self.get_stitched_top_down_maps(stitched_map=allAgents_context_stitchedEgoLocalOccMaps_ph,
                                                        stitch_component=context_egoLocalOccMap,
                                                        ref_pose=ref_pose_for_computing_rel_pose,
                                                        target_pose=allAgents_context_poses[agent_idx][context_idx],
                                                        scene=datapoint_scene,
                                                        num_channels=self.task_cfg.CONTEXT_EGO_LOCAL_MAP_SENSOR.NUM_CHANNELS)

                anechoic_audio_slice_startSampleIdx_firstContext_thisAgent = lst_anechoicAudio_filenameNstartSamplingIdx[agent_idx][1]
                anechoic_audio_slice_startSampleIdx_thisContext_thisAgent = anechoic_audio_slice_startSampleIdx_firstContext_thisAgent +\
                                                                  int(self.rir_sampling_rate * self._anechoic_audio_slice_length * context_idx)
                anechoic_audio_thisAgent = self._anechoic_filename_2_audioData[lst_anechoicAudio_filenameNstartSamplingIdx[agent_idx][0]]

                anechoic_audio_slice_startSampleIdx_thisContext_thisAgent = anechoic_audio_slice_startSampleIdx_thisContext_thisAgent %\
                                                                  anechoic_audio_thisAgent.shape[0]

                if anechoic_audio_slice_startSampleIdx_thisContext_thisAgent +\
                        int(self.rir_sampling_rate * self._anechoic_audio_slice_length) > anechoic_audio_thisAgent.shape[0]:
                    anechoic_audio_slice_prefix = anechoic_audio_thisAgent[anechoic_audio_slice_startSampleIdx_thisContext_thisAgent:]
                    anechoic_audio_slice_suffix = anechoic_audio_thisAgent[:anechoic_audio_slice_startSampleIdx_thisContext_thisAgent\
                                                                            + int(self.rir_sampling_rate * self._anechoic_audio_slice_length)\
                                                                            - anechoic_audio_thisAgent.shape[0]]
                    anechoic_audio_slice = np.concatenate([anechoic_audio_slice_prefix, anechoic_audio_slice_suffix])
                else:
                    anechoic_audio_slice =\
                        anechoic_audio_thisAgent[anechoic_audio_slice_startSampleIdx_thisContext_thisAgent:\
                                                 anechoic_audio_slice_startSampleIdx_thisContext_thisAgent +\
                                                 int(self.rir_sampling_rate * self._anechoic_audio_slice_length)]

                """context self audio"""
                context_selfAudio = self._compute_audio(scene=datapoint_scene,
                                                        azimuth=int(allAgents_context_poses[agent_idx][context_idx][1]),
                                                        receiver_node=int(allAgents_context_poses[agent_idx][context_idx][0]),
                                                        source_node=int(allAgents_context_poses[agent_idx][context_idx][0]),
                                                        anechoic_audio_slice=anechoic_audio_slice,
                                                        )

                allAgents_context_selfAudio_ph[agent_idx][context_idx] = context_selfAudio

                """context self audio mask"""
                if agent_idx == 0:
                    if agent_utterance_switches[context_idx] in [0, 2]:
                        allAgents_context_selfAudioMask_ph[agent_idx][context_idx] = 1
                elif agent_idx == 1:
                    if agent_utterance_switches[context_idx] in [1, 2]:
                        allAgents_context_selfAudioMask_ph[agent_idx][context_idx] = 1
                else:
                    raise ValueError

                """context onlyR (view + self rir) pose"""
                assert len(allAgents_context_poses[agent_idx][context_idx]) == 2
                context_onlyRPose = [
                    allAgents_context_poses[agent_idx][context_idx][0],
                    allAgents_context_poses[agent_idx][context_idx][0],
                    allAgents_context_poses[agent_idx][context_idx][1]
                ]

                context_onlyRRelPose =\
                    np.array(self._compute_relative_pose(current_pose=context_onlyRPose,
                                                         ref_pose=ref_pose_for_computing_rel_pose,
                                                         scene_graph=self._all_scenes_graphs_this_split[datapoint_scene],
                                                         )).astype("float32")

                allAgents_context_onlyRRelPoses_ph[agent_idx][context_idx] = context_onlyRRelPose

                """context  audio from other ego"""
                if num_agents == 2:
                    if agent_idx == 0:
                        other_agent_idx = 1
                    elif agent_idx == 1:
                        other_agent_idx = 0
                    else:
                        raise ValueError
                else:
                    raise NotImplementedError

                anechoic_audio_slice_startSampleIdx_firstContext_thisAgent = lst_anechoicAudio_filenameNstartSamplingIdx[other_agent_idx][1]
                anechoic_audio_slice_startSampleIdx_thisContext_thisAgent = anechoic_audio_slice_startSampleIdx_firstContext_thisAgent +\
                                                                  int(self.rir_sampling_rate * self._anechoic_audio_slice_length * context_idx)
                anechoic_audio_thisAgent = self._anechoic_filename_2_audioData[lst_anechoicAudio_filenameNstartSamplingIdx[other_agent_idx][0]]

                anechoic_audio_slice_startSampleIdx_thisContext_thisAgent = anechoic_audio_slice_startSampleIdx_thisContext_thisAgent %\
                                                                  anechoic_audio_thisAgent.shape[0]

                if anechoic_audio_slice_startSampleIdx_thisContext_thisAgent +\
                        int(self.rir_sampling_rate * self._anechoic_audio_slice_length) > anechoic_audio_thisAgent.shape[0]:
                    anechoic_audio_slice_prefix = anechoic_audio_thisAgent[anechoic_audio_slice_startSampleIdx_thisContext_thisAgent:]
                    anechoic_audio_slice_suffix = anechoic_audio_thisAgent[:anechoic_audio_slice_startSampleIdx_thisContext_thisAgent\
                                                                            + int(self.rir_sampling_rate * self._anechoic_audio_slice_length)\
                                                                            - anechoic_audio_thisAgent.shape[0]]
                    anechoic_audio_slice = np.concatenate([anechoic_audio_slice_prefix, anechoic_audio_slice_suffix])
                else:
                    anechoic_audio_slice =\
                        anechoic_audio_thisAgent[anechoic_audio_slice_startSampleIdx_thisContext_thisAgent:\
                                                 anechoic_audio_slice_startSampleIdx_thisContext_thisAgent +\
                                                 int(self.rir_sampling_rate * self._anechoic_audio_slice_length)]

                if agent_idx == 0:
                    context_otherAudio = self._compute_audio(scene=datapoint_scene,
                                                             azimuth=int(allAgents_context_poses[agent_idx][context_idx][1]),
                                                             receiver_node=int(allAgents_context_poses[agent_idx][context_idx][0]),
                                                             source_node=int(allAgents_context_poses[1][context_idx][0]),
                                                             anechoic_audio_slice=anechoic_audio_slice,
                                                             )
                elif agent_idx == 1:
                    context_otherAudio = self._compute_audio(scene=datapoint_scene,
                                                             azimuth=int(allAgents_context_poses[agent_idx][context_idx][1]),
                                                             receiver_node=int(allAgents_context_poses[agent_idx][context_idx][0]),
                                                             source_node=int(allAgents_context_poses[0][context_idx][0]),
                                                             anechoic_audio_slice=anechoic_audio_slice,
                                                             )

                allAgents_context_otherAudio_ph[agent_idx][context_idx] = context_otherAudio

                """context audio from other ego mask"""
                if agent_idx == 0:
                    if agent_utterance_switches[context_idx] in [1, 2]:
                        allAgents_context_otherAudioMask_ph[agent_idx][context_idx] = 1
                elif agent_idx == 1:
                    if agent_utterance_switches[context_idx] in [0, 2]:
                            allAgents_context_otherAudioMask_ph[agent_idx][context_idx] = 1
                else:
                    raise ValueError

                """context sr (audio from other ego) pose"""
                context_srPose = [
                    allAgents_context_poses[agent_idx][context_idx][0],
                    None,
                    allAgents_context_poses[agent_idx][context_idx][1]
                ]
                if agent_idx == 0:
                    context_srPose[1] = allAgents_context_poses[1][context_idx][0]
                elif agent_idx == 1:
                    context_srPose[1] = allAgents_context_poses[0][context_idx][0]
                else:
                    raise ValueError

                context_srRelPose =\
                    np.array(self._compute_relative_pose(current_pose=context_srPose,
                                                         ref_pose=ref_pose_for_computing_rel_pose,
                                                         scene_graph=self._all_scenes_graphs_this_split[datapoint_scene],
                                                         )).astype("float32")

                allAgents_context_srRelPoses_ph[agent_idx][context_idx] = context_srRelPose

        datapoint["context"] = {}

        """context views + onlyRRelPoses + mask"""
        datapoint["context"]["rgbs"] = allAgents_context_rgbs_ph
        datapoint["context"]["egoLocalOccMaps"] = allAgents_context_egoLocalOccMaps_ph
        if self.stitch_top_down_maps:
            datapoint["context"]["stitched_egoLocalOccMaps"] = allAgents_context_stitchedEgoLocalOccMaps_ph
            datapoint["context"]["ref_rAz"] = ref_rAz_ph
        datapoint["context"]["views_pose"] = allAgents_context_onlyRRelPoses_ph
        datapoint["context"]["views_mask"] = allAgents_context_viewsMask_ph

        """context self audio + onlyRRelPoses + mask"""
        datapoint["context"]["selfAudio"] = allAgents_context_selfAudio_ph
        datapoint["context"]["selfAudio_pose"] = allAgents_context_onlyRRelPoses_ph
        datapoint["context"]["selfAudio_mask"] = allAgents_context_selfAudioMask_ph

        """context audio from other ego + srRelPoses + mask"""
        datapoint["context"]["otherAudio"] = allAgents_context_otherAudio_ph
        datapoint["context"]["otherAudio_pose"] = allAgents_context_srRelPoses_ph
        datapoint["context"]["otherAudio_mask"] = allAgents_context_otherAudioMask_ph

        for query_idx in range(query_length):   # self.max_query_length
            """query gt maps"""
            query_gt_globalCanOccMap_egoCrop =\
                self.all_scenes_gt_global_canonical_occ_ego_crops_n_rots_maps[datapoint_scene][(query_poses[query_idx][0],
                                                                                                self._compute_rotation_from_azimuth(query_poses[query_idx][1]))].astype("float32")

            if self.sim_cfg.GT_GLOBAL_CANONICAL_OCC_MAP_EGO_CROP.NUM_CHANNELS == 1:
                query_gt_globCanOccMaps_egoCrops_ph[query_idx] = query_gt_globalCanOccMap_egoCrop[..., :1]
            elif self.sim_cfg.GT_GLOBAL_CANONICAL_OCC_MAP_EGO_CROP.NUM_CHANNELS == 2:
                query_gt_globCanOccMaps_egoCrops_ph[query_idx] = query_gt_globalCanOccMap_egoCrop
            else:
                raise ValueError

            if self.stitch_top_down_maps:
                query_stitchedGtGlobCanOccMapsEgoCrops_ph =\
                    self.get_stitched_top_down_maps(stitched_map=query_stitchedGtGlobCanOccMapsEgoCrops_ph,
                                                    stitch_component=query_gt_globalCanOccMap_egoCrop,
                                                    ref_pose=ref_pose_for_computing_rel_pose,
                                                    target_pose=query_poses[query_idx],
                                                    scene=datapoint_scene,
                                                    is_occupancy=True,
                                                    is_ego_360deg_crops=True,)

            query_globCanOccMaps_egoCrop_exploredMask = self.get_query_globalCanOccMap_egoCrop_exploredMask(
                scene=datapoint_scene,
                node=query_poses[query_idx][0],
                az=query_poses[query_idx][1],
            )
            query_globCanOccMaps_egoCrop_exploredMasks_ph[query_idx] = query_globCanOccMaps_egoCrop_exploredMask

            """query onlyR pose"""
            assert len(query_poses[query_idx]) == 2
            query_onlyRPose = [
                query_poses[query_idx][0],
                query_poses[query_idx][0],
                query_poses[query_idx][1]
            ]

            query_onlyRRelPose =\
                np.array(self._compute_relative_pose(current_pose=query_onlyRPose,
                                                     ref_pose=ref_pose_for_computing_rel_pose,
                                                     scene_graph=self._all_scenes_graphs_this_split[datapoint_scene],
                                                     )).astype("float32")

            query_relPoses_ph[query_idx] = query_onlyRRelPose

            """query mask"""
            query_mask_ph[query_idx] = 1

            """query sceneIdx and rAz (eval_mode)"""
            assert datapoint_scene in SCENE_NAME_TO_IDX[self.scene_dataset]
            query_sceneIdxs_ph[query_idx] = SCENE_NAME_TO_IDX[self.scene_dataset][datapoint_scene]

            if self._eval_mode:
                # max_query_length x 2 ... in the order of r, Az
                # r gets assigned
                query_rAz_ph[query_idx][0] = int(query_poses[query_idx][0])
                # az gets assigned
                query_rAz_ph[query_idx][1] = int(query_poses[query_idx][1])

        datapoint["query"] = {}

        """query gt maps + onlyRRelPoses + mask"""
        datapoint["query"]["globalCanOccMaps_egoCrops_gt"] = query_gt_globCanOccMaps_egoCrops_ph
        datapoint["query"]["globalCanOccMaps_egoCrops_exploredMasks"] = query_globCanOccMaps_egoCrop_exploredMasks_ph
        if self.stitch_top_down_maps:
            datapoint["query"]["stitched_globalCanOccMaps_egoCrops_gt"] = query_stitchedGtGlobCanOccMapsEgoCrops_ph
        datapoint["query"]["views_pose"] = query_relPoses_ph
        datapoint["query"]["views_mask"] = query_mask_ph

        """query sceneIdxs and rAzs for viz"""
        datapoint["query"]["scene_idxs"] = query_sceneIdxs_ph
        if self._eval_mode:
            datapoint["query"]["ep_idxs"] = query_epIdxs_ph
            datapoint["query"]["rAzs"] = query_rAz_ph

        return datapoint

    def _compute_audio(self,
                       scene='',
                       azimuth=0,
                       receiver_node=0,
                       source_node=0,
                       anechoic_audio_slice=None,
                       ):
        """convolve IR with anechoic audio to compute spatial audio"""
        rir_file = os.path.join(self.rir_dir, scene, "irs", f"{receiver_node}_{source_node}.wav")
        assert os.path.isfile(rir_file), print(rir_file)

        try:
            fs_imp, sig_imp = wavfile.read(rir_file)
            assert fs_imp == self.rir_sampling_rate, "RIR doesn't have sampling frequency of rir_sampling_rate"
        except ValueError:
            sig_imp = np.zeros((self.rir_sampling_rate, 9)).astype("float32")
            fs_imp = self.rir_sampling_rate

        if len(sig_imp) == 0:
            sig_imp = np.zeros((self.rir_sampling_rate, 9)).astype("float32")
            fs_imp = self.rir_sampling_rate

        imp_full_length = np.zeros((self._max_valid_impulse_length, 9))
        imp_full_length[:min(imp_full_length.shape[0], sig_imp.shape[0])] =\
            sig_imp[:min(imp_full_length.shape[0], sig_imp.shape[0]), :]
        imp_full_length = self._rotate_ambisonics(imp_full_length, int(azimuth))

        sig_imp = imp_full_length
        assert fs_imp == self.rir_sampling_rate

        assert anechoic_audio_slice.dtype == np.int16
        avg_power = np.power((np.mean(np.power(anechoic_audio_slice.astype("float32"), 2))), 0.5)
        anechoic_audio_slice = anechoic_audio_slice.astype("float32") * self.audio_cfg.ANECHOIC_AUDIO_TARGET_RMS\
                               / (avg_power + 1.0e-13)
        anechoic_audio_slice = anechoic_audio_slice.astype("int16")
        anechoic_audio_slice = anechoic_audio_slice.astype("float32") / 32768

        convolved = []
        for ch_i in range(sig_imp.shape[-1]):
            """using scipy.signal.convolve in place of fftconvolve chooses the fastest algo for convolution when mode='auto'"""
            convolved.append(scipy.signal.convolve(anechoic_audio_slice,
                                                   sig_imp[:, ch_i],
                                                   mode="full").astype("float32"))
        convolved = np.array(convolved).T

        convolved = convolved[:sig_imp.shape[0]]
        sig_imp = convolved

        return sig_imp

    def _compute_relative_pose(self,
                               current_pose=[],
                               ref_pose=[],
                               scene_graph=None,
                               ):
        """compute relative pose with respect to a reference pose"""

        assert isinstance(current_pose, list)
        assert isinstance(ref_pose, list)
        assert len(ref_pose) == 2

        assert len(current_pose) == 3

        ref_position_xyz = np.array(list(scene_graph.nodes[ref_pose[0]]["point"]), dtype=np.float32)
        if self._add_truncated_gaussian_pose_noise:
            assert (ref_pose[0], ref_pose[-1]) in self._rAz_2_noise_dlX_dlZ_dlAz
            ref_rAz_noise_dlX_dlZ_dlAz = self._rAz_2_noise_dlX_dlZ_dlAz[(ref_pose[0], ref_pose[-1])]

            ref_position_xyz[0] += ref_rAz_noise_dlX_dlZ_dlAz[0]
            ref_position_xyz[2] += ref_rAz_noise_dlX_dlZ_dlAz[1]

        ref_az = ref_pose[-1]
        if self._add_truncated_gaussian_pose_noise:
            ref_az += ref_rAz_noise_dlX_dlZ_dlAz[-1]
        rotation_world_ref = quat_from_angle_axis(np.deg2rad(self._compute_rotation_from_azimuth(ref_az)),
                                                  np.array([0, 1, 0]))

        agent_position_xyz = np.array(list(scene_graph.nodes[current_pose[0]]["point"]), dtype=np.float32)
        if self._add_truncated_gaussian_pose_noise:
            assert (current_pose[0], current_pose[-1]) in  self._rAz_2_noise_dlX_dlZ_dlAz
            agent_rAz_noise_dlX_dlZ_dlAz = self._rAz_2_noise_dlX_dlZ_dlAz[(current_pose[0], current_pose[-1])]

            agent_position_xyz[0] += agent_rAz_noise_dlX_dlZ_dlAz[0]
            agent_position_xyz[2] += agent_rAz_noise_dlX_dlZ_dlAz[1]

        agent_az = current_pose[2]

        if self._add_truncated_gaussian_pose_noise:
            agent_az += agent_rAz_noise_dlX_dlZ_dlAz[-1]
        rotation_world_agent = quat_from_angle_axis(np.deg2rad(self._compute_rotation_from_azimuth(agent_az)),
                                                    np.array([0, 1, 0]))

        audio_source_position_xyz = np.array(list(scene_graph.nodes[current_pose[1]]["point"]), dtype=np.float32)
        if self._add_truncated_gaussian_pose_noise:
            audio_source_rAz_noise_dlX_dlZ_dlAz = None
            for tmp_az in ALL_AZIMUTHS:
                if (current_pose[1], tmp_az) in self._rAz_2_noise_dlX_dlZ_dlAz:
                    audio_source_rAz_noise_dlX_dlZ_dlAz = self._rAz_2_noise_dlX_dlZ_dlAz[(current_pose[1], tmp_az)]
                    break
            assert audio_source_rAz_noise_dlX_dlZ_dlAz is not None
            audio_source_position_xyz[0] += audio_source_rAz_noise_dlX_dlZ_dlAz[0]
            audio_source_position_xyz[2] += audio_source_rAz_noise_dlX_dlZ_dlAz[1]

        # next 2 lines compute relative rotation in the counter-clockwise direction, i.e. -z to -x
        # rotation_world_agent.inverse() * rotation_world_ref = rotation_world_agent - rotation_world_ref
        heading_vector = quaternion_rotate_vector(rotation_world_agent.inverse() * rotation_world_ref,
                                                  np.array([0, 0, -1]))
        agent_heading = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]

        agent_rel_pose = [-agent_position_xyz[2], agent_position_xyz[0], -audio_source_position_xyz[2],
                          audio_source_position_xyz[0], agent_heading]

        return agent_rel_pose

    def get_query_globalCanOccMap_egoCrop_exploredMask(self, scene=None, node=None, az=None):
        """compute a mask that reveals the parts of a target topdown map that have been explored in the input maps"""

        query_globalCanOccMap_egoCrop_exploredMask = np.ones((self.sim_cfg.GT_GLOBAL_CANONICAL_OCC_MAP_EGO_CROP.SIZE,
                                                          self.sim_cfg.GT_GLOBAL_CANONICAL_OCC_MAP_EGO_CROP.SIZE,
                                                          self.sim_cfg.GT_GLOBAL_CANONICAL_OCC_MAP_EGO_CROP.NUM_CHANNELS)).astype("float32")

        rowOrCol_center = (self.sim_cfg.GT_GLOBAL_CANONICAL_OCC_MAP_EGO_CROP.SIZE // 2) - 1

        bot_row = rowOrCol_center
        top_row = max(rowOrCol_center - (self.sim_cfg.EGO_LOCAL_OCC_MAP.SIZE - 1), 0)
        left_col = max(rowOrCol_center - ((self.sim_cfg.EGO_LOCAL_OCC_MAP.SIZE - 1) // 2), 0)
        right_col = left_col + (self.sim_cfg.EGO_LOCAL_OCC_MAP.SIZE - 1)

        assert 0 <= top_row < query_globalCanOccMap_egoCrop_exploredMask.shape[0]
        assert 0 <= bot_row < query_globalCanOccMap_egoCrop_exploredMask.shape[0]
        assert 0 <= left_col < query_globalCanOccMap_egoCrop_exploredMask.shape[1]
        assert 0 <= right_col < query_globalCanOccMap_egoCrop_exploredMask.shape[1]

        local_ego_explored_map = self.all_scenes_local_ego_occ_maps[scene][(node,
                                                                            self._compute_rotation_from_azimuth(az))][..., 1:].astype("float32")
        local_ego_inv_explored_map = 1. - local_ego_explored_map
        query_globalCanOccMap_egoCrop_exploredMask[top_row: bot_row + 1, left_col: right_col + 1] = local_ego_inv_explored_map
        return query_globalCanOccMap_egoCrop_exploredMask

    def get_stitched_top_down_maps(self,
                                   stitched_map=None,
                                   stitch_component=None,
                                   ref_pose=None,
                                   target_pose=None,
                                   scene=None,
                                   is_occupancy=False,
                                   is_pred=False,
                                   is_ego_360deg_crops=False,
                                   stitched_map_updateCounter=None,
                                   num_channels=1,):
        """
        get stitched (registered onto a shared map) version of individual crops (predicted or gt) of a map
        :param stitched_map: shared map in which the stitching will take place
        :param stitch_component: individual crop (predicted or gt) to stitch
        :param ref_pose: refernce pose for stitching
        :param target_pose: target pose of individual crop (predicted or gt)
        :param graph: scene graph
        :param is_occupancy: flag saying if map just has the occupied channel (number of channels = 1)
        :param is_pred: flag sayinf if map is a prediction
        :param is_ego_360deg_crops: flag saying if map is a 360 degree FoV egocentric crop
        :param stitched_map_updateCounter: update counter tracking the number of updates occurring for every cell in the
                                           stitched (shared) map
        :param num_channels: number of channels in the maps (useful if is_occupancy is False)
        :return: stitched map
        """
        center = (stitched_map.shape[0] // 2, stitched_map.shape[1] // 2)

        ref_posn = self._all_scenes_graphs_this_split[scene].nodes()[ref_pose[0]]['point']
        ref_x = ref_posn[0]
        ref_z = ref_posn[2]

        ref_az = ref_pose[1]

        target_posn = self._all_scenes_graphs_this_split[scene].nodes()[target_pose[0]]['point']
        x = target_posn[0]
        z = target_posn[2]

        az = target_pose[1]

        map_scale = self.sim_cfg.GT_GLOBAL_CANONICAL_OCC_MAP_EGO_CROP.SCALE
        if ref_az == 0:
            del_row = int((z - ref_z) / map_scale)
            del_col = int((x - ref_x) / map_scale)
        elif ref_az == 90:
            del_row = int((ref_x - x) / map_scale)
            del_col = int((z - ref_z) / map_scale)
        elif ref_az == 180:
            del_row = int((ref_z - z) / map_scale)
            del_col = int((ref_x - x) / map_scale)
        elif ref_az == 270:
            del_row = int((x - ref_x) / map_scale)
            del_col = int((ref_z - z) / map_scale)

        del_az = (360 + ref_az - az) % 360

        if is_occupancy:
            stitch_component_rotated = np.rot90(stitch_component[..., :1].copy(), del_az // 90)
        else:
            stitch_component_rotated = np.rot90(stitch_component[..., :num_channels].copy(), del_az // 90)

        center_stitchComponent = (center[0] + del_row, center[1] + del_col)

        map_size = stitch_component_rotated.shape[0]
        if is_ego_360deg_crops:
            map_numRowsToTop_or_numColsToLeft = map_size // 2
            top_row_stitchedMap = center_stitchComponent[0] - (map_numRowsToTop_or_numColsToLeft - 1)
            left_col_stitchedMap = center_stitchComponent[1] - (map_numRowsToTop_or_numColsToLeft - 1)
        else:
            if del_az == 0:
                top_row_stitchedMap = center_stitchComponent[0] - (map_size - 1)
                left_col_stitchedMap = center_stitchComponent[1] - ((map_size - 1) // 2)
            elif del_az == 90:
                top_row_stitchedMap = center_stitchComponent[0] - ((map_size - 1) // 2)
                left_col_stitchedMap = center_stitchComponent[1] - (map_size - 1)
            elif del_az == 180:
                top_row_stitchedMap = center_stitchComponent[0]
                left_col_stitchedMap = center_stitchComponent[1] - ((map_size - 1) // 2)
            elif del_az == 270:
                top_row_stitchedMap = center_stitchComponent[0] - ((map_size - 1) // 2)
                left_col_stitchedMap = center_stitchComponent[1]

        bot_row_stitchedMap = top_row_stitchedMap + (map_size - 1)
        right_col_stitchedMap = left_col_stitchedMap + (map_size - 1)

        top_row_stitchComponent = 0
        bot_row_stitchComponent = map_size - 1
        left_col_stitchComponent = 0
        right_col_stitchComponent = map_size - 1

        if top_row_stitchedMap < 0:
            top_row_stitchComponent = abs(top_row_stitchedMap)
            top_row_stitchedMap = 0
        assert 0 <= top_row_stitchedMap < stitched_map.shape[0]

        if bot_row_stitchedMap >= stitched_map.shape[0]:
            bot_row_stitchComponent -= ((bot_row_stitchedMap - stitched_map.shape[0]) + 1)
            bot_row_stitchedMap = stitched_map.shape[0] - 1
        assert 0 <= bot_row_stitchedMap < stitched_map.shape[0]

        assert top_row_stitchedMap <= bot_row_stitchedMap
        assert top_row_stitchComponent <= bot_row_stitchComponent
        assert bot_row_stitchComponent - top_row_stitchComponent == bot_row_stitchedMap - top_row_stitchedMap

        if left_col_stitchedMap < 0:
            left_col_stitchComponent = abs(left_col_stitchedMap)
            left_col_stitchedMap = 0
        assert 0 <= left_col_stitchedMap < stitched_map.shape[1]

        if right_col_stitchedMap >= stitched_map.shape[1]:
            right_col_stitchComponent -= ((right_col_stitchedMap - stitched_map.shape[1]) + 1)
            right_col_stitchedMap = stitched_map.shape[1] - 1
        assert 0 <= right_col_stitchedMap < stitched_map.shape[1]

        assert left_col_stitchedMap <= right_col_stitchedMap
        assert left_col_stitchComponent <= right_col_stitchComponent
        assert right_col_stitchComponent - left_col_stitchComponent == right_col_stitchedMap - left_col_stitchedMap

        if is_pred:
            stitched_map[top_row_stitchedMap: bot_row_stitchedMap + 1,
                        left_col_stitchedMap: right_col_stitchedMap + 1, :] =\
                (stitched_map[top_row_stitchedMap: bot_row_stitchedMap + 1,
                            left_col_stitchedMap: right_col_stitchedMap + 1, :] +\
                 stitch_component_rotated[top_row_stitchComponent: bot_row_stitchComponent + 1,
                                          left_col_stitchComponent: right_col_stitchComponent + 1]).astype("float32")

            stitched_map_updateCounter[top_row_stitchedMap: bot_row_stitchedMap + 1,
                                        left_col_stitchedMap: right_col_stitchedMap + 1, :] += 1

            return stitched_map, stitched_map_updateCounter
        else:
            stitched_map[top_row_stitchedMap: bot_row_stitchedMap + 1,
                        left_col_stitchedMap: right_col_stitchedMap + 1, :] =\
                    np.logical_or(stitched_map[top_row_stitchedMap: bot_row_stitchedMap + 1,
                                              left_col_stitchedMap: right_col_stitchedMap + 1, :],
                                  stitch_component_rotated[top_row_stitchComponent: bot_row_stitchComponent + 1,
                                                            left_col_stitchComponent: right_col_stitchComponent + 1]).astype("float32")

            return stitched_map

    def _rotate_ambisonics(self, signal, rotation_angle=0):
        """rotate an ambisonic IR to make it egocentric"""
        rotation_angle = (rotation_angle + 360) % 360
        assert rotation_angle in [0, 90, 180, 270]
        out_signal = signal
        if rotation_angle == 90:
            out_signal = out_signal[:, [0, 3, 2, 1, 4, 7, 6, 5, 8]]
            out_signal[:, 3] *= -1
            out_signal[:, 4] *= -1
            out_signal[:, 7] *= -1
            out_signal[:, 8] *= -1
        elif rotation_angle == 180:
            out_signal[:, 1] *= -1
            out_signal[:, 3] *= -1
            out_signal[:, 5] *= -1
            out_signal[:, 7] *= -1
        elif rotation_angle == 270:
            out_signal = out_signal[:, [0, 3, 2, 1, 4, 7, 6, 5, 8]]
            out_signal[:, 1] *= -1
            out_signal[:, 4] *= -1
            out_signal[:, 5] *= -1
            out_signal[:, 8] *= -1

        return out_signal

    def _compute_rotation_from_azimuth(self, azimuth):
        # rotation is calculated in the habitat coordinate frame counter-clocwise so -Z is 0 and -X is -90
        return -(azimuth + 0) % 360
