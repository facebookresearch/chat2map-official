# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List
from collections import defaultdict
import logging
import pickle
import os
import cv2
import torch

import librosa
import scipy
import numba
import numpy as np
import networkx as nx
from scipy.io import wavfile
from scipy.signal import fftconvolve

from habitat.core.registry import registry
import habitat_sim
from habitat.tasks.utils import (
    cartesian_to_polar,
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from habitat_sim.utils.common import quat_from_angle_axis, quat_from_coeffs, quat_to_angle_axis
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.core.simulator import (Config, AgentState, ShortestPathPoint)
from habitat_audio.utils import load_points_data, _to_tensor


EPS = 1e-8

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

    # rand_nums = np.random.randn(H, W, 3).astype(np.float32)

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
    r"""Estimates the top-down occupancy based on current depth-map.
    Args:
        sim: reference to the simulator for calculating task observations.
        config: contains the MAP_SCALE, MAP_SIZE, HEIGHT_THRESH fields to
                decide grid-size, extents of the projection, and the thresholds
                for determining obstacles and explored space.
    """

    def __init__(
        self, map_size=31, map_scale=0.1, position=[0, 1.25, 0], depth_sensor_hfov=90,
        height_thresh=(0.2, 1.5), depth_sensor_min_depth=0, depth_sensor_max_depth=10,
        depth_sensor_width=128, depth_sensor_height=128, depth_sensor_normalize_depth=False,
    ):
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
        Inputs:
            depth = (H, W, 1) numpy array
        Returns:
            xyz_camera = (N, 3) numpy array for (X, Y, Z) in egocentric world coordinates
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


class DummySimulatorMultiAgent:
    def __init__(self, num_agents=2):
        self.num_agents = num_agents
        self.positions = [None] * num_agents
        self.rotations = [None] * num_agents
        self._sim_obs = None

        self.position = None
        self.rotation = None

    def seed(self, seed):
        pass

    def set_agent_state(self, positions=[], rotations=[]):
        for i in range(len(positions)):
            self.positions[i] = np.array(positions[i], dtype=np.float32)
            self.rotations[i] = rotations[i]

        self.position = np.array(positions[0], dtype=np.float32)
        self.rotation = rotations[0]

    def get_agent_state(self):
        class State:
            def __init__(self, positions=[], rotations=[]):
                self.positions = []
                self.rotations = []
                for i in range(len(positions)):
                    self.positions.append(positions[i])
                    self.rotations.append(rotations[i])

                self.position = positions[0]
                self.rotation = rotations[0]

        return State(self.positions, self.rotations)

    def set_sensor_observations(self, sim_obs):
        self._sim_obs = sim_obs

    def get_sensor_observations(self):
        return self._sim_obs

    def close(self):
        pass
        

@registry.register_simulator()
class HabitatSimAudioEnabledMultiAgentActiveMapping(HabitatSim):
    def action_space_shortest_path(self, source: AgentState, targets: List[AgentState], agent_id: int = 0) -> List[
            ShortestPathPoint]:
        pass

    def __init__(self, config: Config) -> None:
        """Changes made to simulator wrapper over habitat-sim
    
        This simulator allows two agents to have a conversation episode between them as per the Chat2Map task
        Args:
            config: configuration for initializing the simulator.
        """
        super().__init__(config)

        self.env_cfg = self.config.SIM_ENV
        self.task_cfg = self.config.SIM_TASK
        self.audio_cfg = self.config.AUDIO
        self.passive_mapping_cfg = self.config.SIM_TRAINER

        self.scene_dataset = self.config.SCENE_DATASET
        self.rir_sampling_rate = self.audio_cfg.RIR_SAMPLING_RATE
        self._max_valid_impulse_length = self.audio_cfg.MAX_VALID_IMPULSE_LENGTH_AFTER_REMOVING_LEADING_ZEROS

        self.hop_length = self.audio_cfg.HOP_LENGTH
        self.n_fft = self.audio_cfg.N_FFT
        self.win_length = self.audio_cfg.WIN_LENGTH
        self._anechoic_audio_slice_length = self.audio_cfg.ANECHOIC_AUDIO_SLICE_LENGTH

        self._audio_wav_shape = self.task_cfg.CONTEXT_SELF_AUDIO_SENSOR.FEATURE_SHAPE

        print(f"LOADING ANECHOIC AUDIO FOR train")
        anechoic_audio_dir = self.audio_cfg.ANECHOIC_DIR
        assert os.path.isdir(anechoic_audio_dir)

        anechoic_audio_filenames = os.listdir(anechoic_audio_dir)

        self._anechoic_filename_2_audioData = {}
        for anechoic_audio_filename in anechoic_audio_filenames:
            anechoic_audio_filePath = os.path.join(anechoic_audio_dir, anechoic_audio_filename)
            assert os.path.isfile(anechoic_audio_filePath)

            anechoic_audioSR, anechoic_audioData = wavfile.read(anechoic_audio_filePath)
            assert anechoic_audioSR == self.rir_sampling_rate

            assert anechoic_audio_filename.split(".")[0] not in self._anechoic_filename_2_audioData
            self._anechoic_filename_2_audioData[anechoic_audio_filename.split(".")[0]] = anechoic_audioData

        assert "CONTEXT_VIEW_POSE_SENSOR" in self.task_cfg.SENSORS
        self._pose_feat_shape = self.task_cfg.CONTEXT_VIEW_POSE_SENSOR.FEATURE_SHAPE
        self._add_truncated_gaussian_pose_noise = self.task_cfg.CONTEXT_VIEW_POSE_SENSOR.ADD_TRUNCATED_GAUSSIAN_NOISE
        self._truncated_gaussian_pose_noise_cfg = self.task_cfg.CONTEXT_VIEW_POSE_SENSOR.TRUNCATED_GAUSSIAN_NOISE

        # self._truncated_gaussian_pose_noise_random_multipliers = None
        self._gaussian_pose_noise_multipliers = None
        if self._add_truncated_gaussian_pose_noise:
            assert os.path.isfile(self._truncated_gaussian_pose_noise_cfg.GAUSSIAN_NOISE_MULTIPLIERS_PATH)
            with open(self._truncated_gaussian_pose_noise_cfg.GAUSSIAN_NOISE_MULTIPLIERS_PATH, "rb") as fi:
                self._gaussian_pose_noise_multipliers = pickle.load(fi)

        self.max_context_length = self.env_cfg.MAX_CONTEXT_LENGTH
        self.visual_budget = self.env_cfg.VISUAL_BUDGET
        self.max_query_length = self.env_cfg.MAX_QUERY_LENGTH
        assert self.max_query_length == (self.config.ALL_AGENTS.NUM * self.max_context_length)

        self.render_local_ego_occ_maps_from_depth_images = self.config.RENDER_LOCAL_EGO_OCC_MAPS_FROM_DEPTH_IMAGES
        self.local_occMap_cfg = self.config.LOCAL_MAP
        self.ego_mapper = None
        self.redwood_depth_noise_dist_model = None
        self.redwood_depth_noise_multiplier = None
        if self.render_local_ego_occ_maps_from_depth_images:
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

            if self.config.DEPTH_SENSOR.ADD_REDWOOD_NOISE:
                """src: https://github.com/facebookresearch/habitat-sim/blob/main/src_python/habitat_sim/sensors/noise_models/redwood_depth_noise_model.py"""
                assert os.path.isfile(self.config.DEPTH_SENSOR.REDWOOD_DEPTH_NOISE_DIST_MODEL)

                self.redwood_depth_noise_dist_model = np.load(self.config.DEPTH_SENSOR.REDWOOD_DEPTH_NOISE_DIST_MODEL)
                self.redwood_depth_noise_dist_model = self.redwood_depth_noise_dist_model.reshape(80, 80, 5)

                self.redwood_depth_noise_multiplier = self.config.DEPTH_SENSOR.REDWOOD_NOISE_MULTIPLIER

                assert os.path.isfile(self.config.DEPTH_SENSOR.REDWOOD_NOISE_RAND_NUMS_PATH)
                with open(self.config.DEPTH_SENSOR.REDWOOD_NOISE_RAND_NUMS_PATH, "rb") as fi:
                    self._redwood_depth_noise_rand_nums = pickle.load(fi)

        self.stitch_top_down_maps = self.config.STITCH_TOP_DOWN_MAPS

        self.rir_dir = self.audio_cfg.RIR_DIR
        assert os.path.isdir(self.rir_dir)

        self.num_agents = self.config.ALL_AGENTS.NUM
        assert self.num_agents == 2

        self.total_context_length = None
        self.agent_utterance_allSwitches = None
        self.lst_anechoicAudio_filenameNstartSamplingIdx = None
        self.used_query_nodsNrots = None
        self._current_context_rgb = None
        self._current_context_ego_local_map = None 
        self._current_context_view_pose = None
        self._current_context_view_rAz = None
        self._previous_context_view_mask = None
        self._current_context_selfAudio = None 
        self._current_context_otherAudio = None 
        self._current_context_otherAudio_pose = None
        self._current_context_audio_mask = None
        self._all_context_audio_mask = None
        self._current_query_globCanMapEgoCrop_gt = None
        self._current_query_globCanMapEgoCrop_gt_exploredPartMask = None 
        self._current_query_mask = None
        self._all_query_mask = None
        if self.stitch_top_down_maps:
            self._current_stitched_query_globCanMapEgoCrop_gt = None

        assert self.config.SCENE_DATASET in ["mp3d"],\
            "SCENE_DATASET needs to be in ['mp3d']"
        self._previous_receiver_position_indexs = [None] * self.num_agents
        self._current_receiver_position_indexs = [None] * self.num_agents
        self._previous_rotation_angles = [None] * self.num_agents
        self._current_rotation_angles = [None] * self.num_agents
        self._frame_cache = defaultdict(dict)
        self._episode_count = 0
        self._step_count = 0
        self._view_count = self.num_agents
        self._action = 1
        self._is_episode_active = None
        self._previous_step_collideds = [None] * self.num_agents
        self._nodes_n_azimuths_lists = [None] * self.num_agents
        self._position_to_index_mapping = dict()
        self.points, self.graph = load_points_data(self.meta_dir, self.config.AUDIO.GRAPH_FILE,
                                                   scene_dataset=self.config.SCENE_DATASET)
        for node in self.graph.nodes():
            if 'point' in self.graph.nodes()[node]:
                self._position_to_index_mapping[self.position_encoding(self.graph.nodes()[node]['point'])] = node

        logging.info('Current scene: {}'.format(self.current_scene_name,))

        logging.info('Loaded the rendered observations for all scenes')
        with open(self.current_scene_observation_file, 'rb') as fo:
            self._frame_cache = pickle.load(fo)

        assert os.path.isdir(self.config.RENDERED_LOCAL_EGO_OCC_MAPS_DIR)
        self.all_scenes_local_ego_occ_maps = dict()
        logging.info('Loaded the rendered local ego occ maps for this scene')
        with open(self.current_local_ego_occ_maps_file, "rb") as fi:
            self.all_scenes_local_ego_occ_maps[self.current_scene_name] = pickle.load(fi)

        assert os.path.isdir(self.config.RENDERED_GT_GLOBAL_CANONICAL_OCC_MAP_EGO_CROP_N_ROTS_DIR)
        self.all_scenes_gt_global_canonical_occ_ego_crops_n_rots_maps = dict()
        logging.info("Loaded the rendered gt global canonical occ map cropsNrots for this scene")
        with open(self.current_gt_glob_can_occ_map_egoCropsNrots_file, "rb") as fi:
            self.all_scenes_gt_global_canonical_occ_ego_crops_n_rots_maps[self.current_scene_name] = pickle.load(fi)

        if self.config.USE_RENDERED_OBSERVATIONS:
            self._sim.close()
            del self._sim
            self._sim = DummySimulatorMultiAgent()
        else:
            self._dummy_sim = DummySimulatorMultiAgent()

    def get_agent_state(self, agent_id: int = 0) -> habitat_sim.AgentState:
        """
        get current agent state
        :param agent_id: agent ID
        :return: agent state
        """
        if not self.config.USE_RENDERED_OBSERVATIONS:
            agent_state = super().get_agent_state(agent_id)
        else:
            agent_state = self._sim.get_agent_state()
        return agent_state

    def get_agent_state_dummy_sim(self, agent_id: int = 0) -> habitat_sim.AgentState:
        """
        get current agent state from the dummy simulator
        :param agent_id: agent ID
        :return: agent state
        """
        if not self.config.USE_RENDERED_OBSERVATIONS:
            agent_state = self._dummy_sim.get_agent_state()
        else:
            agent_state = self._sim.get_agent_state()
        return agent_state

    def set_agent_state(
        self,
        position: List[float],
        rotation: List[float],
        agent_id: int = 0,
        reset_sensors: bool = True,
    ) -> bool:
        """
        set agent's state when not using pre-rendered observations
        :param position: 3D position of the agent
        :param rotation: rotation angle of the agent
        :param agent_id: agent ID
        :param reset_sensors: reset sensors or not
        :return: None
        """
        if not self.config.USE_RENDERED_OBSERVATIONS:
            super().set_agent_state(position, rotation, agent_id=agent_id, reset_sensors=reset_sensors)
        else:
            pass

    @property
    def current_scene_observation_file(self):
        """
        get path to pre-rendered observations for the current scene
        :return: path to pre-rendered observations for the current scene
        """
        return os.path.join(self.config.RENDERED_OBSERVATIONS, self.config.SCENE_DATASET,
                            self.current_scene_name + '.pkl')

    @property
    def current_local_ego_occ_maps_file(self):
        """
        get path to 90 degree FoV local egocentric occupancy maps file
        :return: path to 90 degree FoV local egocentric occupancy maps file
        """
        return os.path.join(self.config.RENDERED_LOCAL_EGO_OCC_MAPS_DIR,
                            self.current_scene_name + '.pkl')

    @property
    def current_gt_glob_can_occ_map_egoCropsNrots_file(self):
        """
        get path to 360 degree FoV local egocentric target occupancy maps file
        :return: path to 360 degree FoV local egocentric target occupancy maps file
        """
        return os.path.join(self.config.RENDERED_GT_GLOBAL_CANONICAL_OCC_MAP_EGO_CROP_N_ROTS_DIR,
                            self.current_scene_name + '.pkl')
    
    @property
    def meta_dir(self):
        """
        get path to meta-dir containing data about location of navigation nodes and their connectivity
        :return: path to meta-dir containing data about location of navigation nodes and their connectivity
        """
        return os.path.join(self.config.AUDIO.META_DIR, self.current_scene_name)

    @property
    def current_scene_name(self):
        """
        get current scene name
        :return: current scene name
        """
        if self.config.SCENE_DATASET == "mp3d":
            return self._current_scene.split('/')[-2]
        else:
            raise ValueError

    def reconfigure(self, config: Config) -> None:
        """
        reconfigure for new episode
        :param config: config for reconfiguration
        :return: None
        """
        self.config = config
        is_same_scene = config.SCENE == self._current_scene
        if not is_same_scene:
            self._current_scene = config.SCENE
            logging.debug('Current scene: {}'.format(self.current_scene_name))

            if not self.config.USE_RENDERED_OBSERVATIONS:
                self._sim.close()
                del self._sim
                self.sim_config = self.create_sim_config(self._sensor_suite)
                self._sim = habitat_sim.Simulator(self.sim_config)
                self._update_agents_state()

            with open(self.current_scene_observation_file, 'rb') as fo:
                self._frame_cache = pickle.load(fo)

            with open(self.current_local_ego_occ_maps_file, "rb") as fi:
                self.all_scenes_local_ego_occ_maps[self.current_scene_name] = pickle.load(fi)

            with open(self.current_gt_glob_can_occ_map_egoCropsNrots_file, "rb") as fi:
                self.all_scenes_gt_global_canonical_occ_ego_crops_n_rots_maps[self.current_scene_name] = pickle.load(fi)

            logging.info('Loaded scene {}'.format(self.current_scene_name))

            self.points, self.graph = load_points_data(self.meta_dir, self.config.AUDIO.GRAPH_FILE,
                                                       scene_dataset=self.config.SCENE_DATASET)
            for node in self.graph.nodes():
                if "point" in self.graph.nodes()[node]:
                    self._position_to_index_mapping[self.position_encoding(self.graph.nodes()[node]['point'])] = node

        # set agent positions
        self._previous_receiver_position_indexs[0] = self.config.AGENT_0.NODES_N_AZIMUTHS[0][0]
        for i in range(1, len(self._previous_receiver_position_indexs)):
            self._previous_receiver_position_indexs[i] = self.config.AGENT_0.OTHER_NODES_N_AZIMUTHS[i-1][0][0]

        self._current_receiver_position_indexs[0] = self.config.AGENT_0.NODES_N_AZIMUTHS[1][0]
        for i in range(1, len(self._current_receiver_position_indexs)):
            self._current_receiver_position_indexs[i] = self.config.AGENT_0.OTHER_NODES_N_AZIMUTHS[i-1][1][0]

        # the agent rotates about +Y starting from -Z counterclockwise,
        # so rotation angle 90 means the agent rotate about +Y 90 degrees
        self._previous_rotation_angles[0] =\
            int(self.get_rotation_angle_from_azimuth(self.config.AGENT_0.NODES_N_AZIMUTHS[0][1])) % 360
        for i in range(1, len(self._previous_rotation_angles)):
            self._previous_rotation_angles[i] =\
                int(self.get_rotation_angle_from_azimuth(self.config.AGENT_0.OTHER_NODES_N_AZIMUTHS[i-1][0][1])) % 360

        self._current_rotation_angles[0] =\
            self.get_rotation_angle_from_azimuth(self.config.AGENT_0.NODES_N_AZIMUTHS[1][1])
        for i in range(1, len(self._current_rotation_angles)):
            self._current_rotation_angles[i] =\
                self.get_rotation_angle_from_azimuth(self.config.AGENT_0.OTHER_NODES_N_AZIMUTHS[i-1][1][1])

        self._nodes_n_azimuths_lists[0] = self.config.AGENT_0.NODES_N_AZIMUTHS
        for i in range(1, len(self._nodes_n_azimuths_lists)):
            self._nodes_n_azimuths_lists[i] = self.config.AGENT_0.OTHER_NODES_N_AZIMUTHS[i-1]            

        self.total_context_length = len(self._nodes_n_azimuths_lists[0])
        assert self.total_context_length >= self.max_context_length

        if len(self.config.AGENT_0.UTTERANCE_SWITCHES) == 0:
            num_total_utteranceSwitches = 3
            self.agent_utterance_allSwitches = np.random.choice(num_total_utteranceSwitches,
                                                                self.total_context_length,
                                                                replace=True).tolist()
        else:
            self.agent_utterance_allSwitches = self.config.AGENT_0.UTTERANCE_SWITCHES

        if len(self.config.AGENT_0.ANECHOIC_AUDIO_FILENAME_N_START_SAMPLING_IDX) == 0:
            self.lst_anechoicAudio_filenameNstartSamplingIdx = []
            lst_anechoic_filename_idxs = np.random.choice(len(self._anechoic_filename_2_audioData), self.num_agents,
                                                          replace=False).tolist()
            for anechoic_filename_idx in lst_anechoic_filename_idxs:
                train_anechoic_filename_thisAgent = list(self._anechoic_filename_2_audioData.keys())[anechoic_filename_idx]
                train_anechoic_fileStartSampleIdx_thisAgent =\
                    np.random.choice(self._anechoic_filename_2_audioData[train_anechoic_filename_thisAgent].shape[0], 1)[0]
                self.lst_anechoicAudio_filenameNstartSamplingIdx.append((train_anechoic_filename_thisAgent,
                                                                         train_anechoic_fileStartSampleIdx_thisAgent))
        else:
            self.lst_anechoicAudio_filenameNstartSamplingIdx = self.config.AGENT_0.ANECHOIC_AUDIO_FILENAME_N_START_SAMPLING_IDX

        self._rAz_2_noise_dlX_dlZ_dlAz = None
        if self._add_truncated_gaussian_pose_noise:
            assert self._gaussian_pose_noise_multipliers is not None
            assert (str(self.current_scene_name), str(self.config.AGENT_0.EPISODE_ID)) in self._gaussian_pose_noise_multipliers
            gaussian_pose_noise_multipliers_thsScnEpID =\
                self._gaussian_pose_noise_multipliers[(str(self.current_scene_name), str(self.config.AGENT_0.EPISODE_ID))]

            self._rAz_2_noise_dlX_dlZ_dlAz = {}
            for agent_idx in range(len(self._nodes_n_azimuths_lists)):
                contextIdx_tstPsNs = 0
                for context_pose in self._nodes_n_azimuths_lists[agent_idx][:self.max_context_length + 1]:
                    if tuple(context_pose) not in self._rAz_2_noise_dlX_dlZ_dlAz:
                        assert contextIdx_tstPsNs < len(gaussian_pose_noise_multipliers_thsScnEpID)
                        del_x_random_multiplier = float(gaussian_pose_noise_multipliers_thsScnEpID[contextIdx_tstPsNs][agent_idx][0])

                        del_x = del_x_random_multiplier * self._truncated_gaussian_pose_noise_cfg.TRANSLATION_STD +\
                                self._truncated_gaussian_pose_noise_cfg.TRANSLATION_MEAN
                        del_x = max(self._truncated_gaussian_pose_noise_cfg.TRANSLATION_MEAN -\
                                    abs(self._truncated_gaussian_pose_noise_cfg.NUM_STDS_FOR_TRUNCATION)\
                                    * self._truncated_gaussian_pose_noise_cfg.TRANSLATION_STD, del_x)
                        del_x = min(self._truncated_gaussian_pose_noise_cfg.TRANSLATION_MEAN +\
                                    abs(self._truncated_gaussian_pose_noise_cfg.NUM_STDS_FOR_TRUNCATION)\
                                    * self._truncated_gaussian_pose_noise_cfg.TRANSLATION_STD, del_x)

                        assert contextIdx_tstPsNs < len(gaussian_pose_noise_multipliers_thsScnEpID)
                        del_z_random_multiplier = float(gaussian_pose_noise_multipliers_thsScnEpID[contextIdx_tstPsNs][agent_idx][1])

                        del_z = del_z_random_multiplier * self._truncated_gaussian_pose_noise_cfg.TRANSLATION_STD +\
                                self._truncated_gaussian_pose_noise_cfg.TRANSLATION_MEAN
                        del_z = max(self._truncated_gaussian_pose_noise_cfg.TRANSLATION_MEAN -\
                                    abs(self._truncated_gaussian_pose_noise_cfg.NUM_STDS_FOR_TRUNCATION)\
                                    * self._truncated_gaussian_pose_noise_cfg.TRANSLATION_STD, del_z)
                        del_z = min(self._truncated_gaussian_pose_noise_cfg.TRANSLATION_MEAN +\
                                    abs(self._truncated_gaussian_pose_noise_cfg.NUM_STDS_FOR_TRUNCATION)\
                                    * self._truncated_gaussian_pose_noise_cfg.TRANSLATION_STD, del_z)

                        assert contextIdx_tstPsNs < len(gaussian_pose_noise_multipliers_thsScnEpID)
                        del_az_random_multiplier = float(gaussian_pose_noise_multipliers_thsScnEpID[contextIdx_tstPsNs][agent_idx][2])

                        del_az = del_az_random_multiplier * self._truncated_gaussian_pose_noise_cfg.ROTATION_DEGREES_STD +\
                                 self._truncated_gaussian_pose_noise_cfg.ROTATION_DEGREES_MEAN
                        del_az = max(self._truncated_gaussian_pose_noise_cfg.ROTATION_DEGREES_MEAN -\
                                     abs(self._truncated_gaussian_pose_noise_cfg.NUM_STDS_FOR_TRUNCATION)\
                                     * self._truncated_gaussian_pose_noise_cfg.ROTATION_DEGREES_STD, del_az)
                        del_az = min(self._truncated_gaussian_pose_noise_cfg.ROTATION_DEGREES_MEAN +\
                                     abs(self._truncated_gaussian_pose_noise_cfg.NUM_STDS_FOR_TRUNCATION)\
                                     * self._truncated_gaussian_pose_noise_cfg.ROTATION_DEGREES_STD, del_az)

                        self._rAz_2_noise_dlX_dlZ_dlAz[tuple(context_pose)] = [del_x, del_z, del_az]

                    contextIdx_tstPsNs += 1

        self.used_query_nodsNrots = []
        self._current_context_rgb = np.zeros(self.task_cfg.CONTEXT_RGB_SENSOR.FEATURE_SHAPE, dtype="float32")
        self._current_context_ego_local_map = np.zeros(self.task_cfg.CONTEXT_EGO_LOCAL_MAP_SENSOR.FEATURE_SHAPE, dtype="float32")
        self._current_context_view_pose = np.zeros(self.task_cfg.CONTEXT_VIEW_POSE_SENSOR.FEATURE_SHAPE, dtype="float32")
        self._current_context_view_rAz = np.zeros(self.task_cfg.CONTEXT_VIEW_R_N_AZ_SENSOR.FEATURE_SHAPE, dtype="int16")
        self._previous_context_view_mask = np.zeros(self.task_cfg.PREV_CONTEXT_VIEW_MASK_SENSOR.FEATURE_SHAPE, dtype="float32")
        self._current_context_selfAudio = np.zeros(self.task_cfg.CONTEXT_SELF_AUDIO_SENSOR.FEATURE_SHAPE, dtype="float32")
        self._current_context_otherAudio = np.zeros(self.task_cfg.CONTEXT_OTHER_AUDIO_SENSOR.FEATURE_SHAPE, dtype="float32")
        self._current_context_otherAudio_pose = np.zeros(self.task_cfg.CONTEXT_OTHER_AUDIO_POSE_SENSOR.FEATURE_SHAPE, dtype="float32")
        self._current_context_audio_mask = np.zeros(self.task_cfg.CONTEXT_AUDIO_MASK_SENSOR.FEATURE_SHAPE, dtype="float32")
        self._all_context_audio_mask = np.zeros(self.task_cfg.ALL_CONTEXT_AUDIO_MASK_SENSOR.FEATURE_SHAPE, dtype="float32")
        self._current_query_globCanMapEgoCrop_gt = np.zeros(self.task_cfg.QUERY_GT_GLOB_CAN_MAP_EGO_CROP_SENSOR.FEATURE_SHAPE, dtype="float32")
        self._current_query_globCanMapEgoCrop_gt_exploredPartMask = np.zeros(self.task_cfg.QUERY_GT_GLOB_CAN_MAP_EGO_CROP_EXPLORED_PART_MASK_SENSOR.FEATURE_SHAPE, dtype="float32")
        self._current_query_mask = np.zeros(self.task_cfg.QUERY_MASK_SENSOR.FEATURE_SHAPE, dtype="float32")
        self._all_query_mask = np.zeros(self.task_cfg.QUERY_MASK_SENSOR.FEATURE_SHAPE, dtype="float32")
        if self.stitch_top_down_maps:
            self._current_stitched_query_globCanMapEgoCrop_gt = np.zeros(self.task_cfg.QUERY_STITCHED_GT_GLOB_CAN_MAP_EGO_CROP_SENSOR.FEATURE_SHAPE, dtype="float32")

        for i in range(self._all_context_audio_mask.shape[-1]):
            if self.agent_utterance_allSwitches[i] in [0, 2]:
                self._all_context_audio_mask[0, i] = 1

            if self.agent_utterance_allSwitches[i] in [1, 2]:
                self._all_context_audio_mask[1, i] = 1

        if not self.config.USE_RENDERED_OBSERVATIONS:
            self.set_agent_state(list(self.graph.nodes[self._previous_receiver_position_indexs[0]]['point']),
                                 quat_from_angle_axis(np.deg2rad(self.get_rotation_angle_from_azimuth(self.config.AGENT_0.NODES_N_AZIMUTHS[0][1])),
                                                      np.array([0, 1, 0]))
                                 )

        positions = []
        rotations = []
        for i in range(len(self._previous_receiver_position_indexs)):
            positions.append(
                list(self.graph.nodes[self._previous_receiver_position_indexs[i]]['point'])
                )
            rotations.append(
                quat_from_angle_axis(np.deg2rad(self._previous_rotation_angles[i]), np.array([0, 1, 0]))
                )

        if not self.config.USE_RENDERED_OBSERVATIONS:
            self._dummy_sim.set_agent_state(positions, rotations)
        else:
            self._sim.set_agent_state(positions, rotations)

        for i in range(len(self._previous_receiver_position_indexs)):
            logging.debug("Agent {} at {}, orientation: {}".
                          format(i + 1, self._previous_receiver_position_indexs[i],
                                self.get_orientation(self._previous_rotation_angles[i])))

    @staticmethod
    def position_encoding(position):
        return '{:.2f}_{:.2f}_{:.2f}'.format(*position)

    def _position_to_index(self, position):
        if self.position_encoding(position) in self._position_to_index_mapping:
            return self._position_to_index_mapping[self.position_encoding(position)]
        else:
            raise ValueError("Position misalignment.")

    def _get_sim_observation(self):
        """
        get current observation from simulator
        :return: current observation
        """
        joint_index = (self._previous_receiver_position_indexs[0],
                       self._previous_rotation_angles[0])
        if joint_index in self._frame_cache:
            return self._frame_cache[joint_index]
        else:
            sim_obs = self._sim.get_sensor_observations()
            self._frame_cache[joint_index] = sim_obs
            return sim_obs

    def reset(self):
        """
        reset simulator for new episode
        :return: None
        """
        logging.debug('Reset simulation')

        self._episode_count += 1
        self._step_count = 0
        self._view_count = self.num_agents
        self._action = 1

        if not self.config.USE_RENDERED_OBSERVATIONS:
            sim_obs = self._sim.reset()
            if self._update_agents_state():
                sim_obs = self._get_sim_observation()
        else:
            sim_obs = self._get_sim_observation()
            self._sim.set_sensor_observations(sim_obs)

        self._is_episode_active = True
        self._previous_step_collideds = [False] * self.num_agents
        self._prev_sim_obs = sim_obs
        # Encapsule data under Observations class
        observations = self._sensor_suite.get_observations(sim_obs)

        return observations

    def step(self, action, only_allowed=True):
        """
        All angle calculations in this function is w.r.t habitat coordinate frame, on X-Z plane
        where +Y is upward, -Z is forward and +X is rightward.
        Angle 0 corresponds to +X, angle 90 corresponds to +y and 290 corresponds to 270.

        :param action: action to be taken
        :param only_allowed: if true, then can't step anywhere except allowed locations
        :return: Dict of observations
        """
        assert self._is_episode_active, (
            "episode is not active, environment not RESET or "
            "STOP action called previously"
        )
        self._num_views_to_add = 0
        self._action = action
        if action in [2, 3]:
            if self._view_count + 1 <= self.visual_budget:
                self._view_count += 1
                self._num_views_to_add = 1
        elif action == 4:
            if self._view_count + 2 <= self.visual_budget:
                self._view_count += 2
                self._num_views_to_add = 2
            elif self._view_count + 1 <= self.visual_budget:
                self._view_count += 1
                self._num_views_to_add = 1
            
        if self._view_count >= self.visual_budget:
            self._is_episode_active = False

        self._previous_step_collideds = [False] * self.num_agents

        for i in range(self.num_agents):
            self._previous_receiver_position_indexs[i] = self._current_receiver_position_indexs[i]
            self._previous_rotation_angles[i] = self._current_rotation_angles[i]

            self._current_receiver_position_indexs[i] =\
                self._nodes_n_azimuths_lists[i][(self._step_count + 2) % len(self._nodes_n_azimuths_lists[i])][0]
            self._current_rotation_angles[i] =\
                self.get_rotation_angle_from_azimuth(self._nodes_n_azimuths_lists[i][(self._step_count + 2)\
                                                                                     % len(self._nodes_n_azimuths_lists[i])][1])

        if not self.config.USE_RENDERED_OBSERVATIONS:
            self.set_agent_state(list(self.graph.nodes[self._previous_receiver_position_indexs[0]]['point']),
                                 quat_from_angle_axis(np.deg2rad(self._previous_rotation_angles[0]), np.array([0, 1, 0])))

        positions = []
        rotations = []
        for i in range(len(self._previous_receiver_position_indexs)):
            positions.append(
                list(self.graph.nodes[self._previous_receiver_position_indexs[i]]['point'])
                )
            rotations.append(
                quat_from_angle_axis(np.deg2rad(self._previous_rotation_angles[i]), np.array([0, 1, 0]))
                )

        if not self.config.USE_RENDERED_OBSERVATIONS:
            self._dummy_sim.set_agent_state(positions, rotations)
        else:
            self._sim.set_agent_state(positions, rotations)

        self._step_count += 1

        sim_obs = self._get_sim_observation()

        if self.config.USE_RENDERED_OBSERVATIONS:
            self._sim.set_sensor_observations(sim_obs)
        self._prev_sim_obs = sim_obs
        observations = self._sensor_suite.get_observations(sim_obs)

        return observations

    def get_orientation(self, rotation_angle):
        """
        get current orientation of the agent
        :return: current orientation of the agent
        """
        _base_orientation = 270
        return (_base_orientation - rotation_angle) % 360

    def get_episode_count(self):
        """
        get total episode count during eval (possibly redundant)
        :return: total episode count during eval
        """
        return self._episode_count

    def write_info_to_obs(self, observations):
        observations["agent node and location"] = (self._previous_receiver_position_indexs[0],
                                                   self.graph.nodes[self._previous_receiver_position_indexs[0]]["point"])
        observations["scene name"] = self.current_scene_name
        observations["orientation"] = self._previous_rotation_angles[0]

    def azimuth_angle(self, rotation_angle):
        """
        get current azimuth of the agent
        :return: current azimuth of the agent
        """
        # this is the angle used to index the binaural audio files
        # in mesh coordinate systems, +Y forward, +X rightward, +Z upward
        # azimuth is calculated clockwise so +Y is 0 and +X is 90
        return -(rotation_angle + 0) % 360

    def get_rotation_angle_from_azimuth(self, azimuth_angle):
        """
        get current rotation of the agent
        :return: current rotation of the agent
        """
        # this is the angle used to index the cachec RGBD/scene-observation files
        return -(azimuth_angle + 0) % 360

    def geodesic_distance(self, position_a, position_b):
        """
        get geodesic distance between 2 nodes
        :param position_a: position of 1st node
        :param position_b: position of 2nd node
        :return: geodesic distance between 2 nodes
        """
        index_a = self._position_to_index(position_a)
        index_b = self._position_to_index(position_b)
        assert index_a is not None and index_b is not None
        steps = nx.shortest_path_length(self.graph, index_a, index_b) * self.config.GRID_SIZE
        return steps

    def euclidean_distance(self, position_a, position_b):
        """
        get euclidean distance between 2 nodes
        :param position_a: position of 1st node
        :param position_b: position of 2nd node
        :return: euclidean distance between 2 nodes
        """
        assert len(position_a) == len(position_b) == 3
        assert position_a[1] == position_b[1], "height should be same for node a and b"
        return np.power(np.power(position_a[0] - position_b[0],  2) + np.power(position_a[2] - position_b[2], 2), 0.5)

    @property
    def previous_step_collided(self):
        return self._previous_step_collideds[0]

    def get_current_context_rgb(self):
        """get current RGB"""
        if self._step_count == 0:
            for step_idx in range(self.max_context_length):
                lst_rgb = []
                for agent_idx in range(len(self._nodes_n_azimuths_lists)):
                    joint_index = (self._nodes_n_azimuths_lists[agent_idx][step_idx][0],
                                   self.get_rotation_angle_from_azimuth(self._nodes_n_azimuths_lists[agent_idx][step_idx][1]))
                    assert joint_index in self._frame_cache
                    rgb = self._frame_cache[joint_index]["rgb"][..., :3]
                    lst_rgb.append(rgb.astype("float32"))

                self._current_context_rgb[:, step_idx, ...] = np.stack(lst_rgb, axis=0)

        return self._current_context_rgb

    def get_current_context_ego_local_map(self):
        """get current 90 degree FoV egocentric local occupancy map"""
        if self._step_count == 0:
            for step_idx in range(self.max_context_length):
                lst_ego_local_map = []
                for agent_idx in range(len(self._nodes_n_azimuths_lists)):
                    joint_idx = (self._nodes_n_azimuths_lists[agent_idx][step_idx][0],
                                   self.get_rotation_angle_from_azimuth(self._nodes_n_azimuths_lists[agent_idx][step_idx][1]))
                    if self.render_local_ego_occ_maps_from_depth_images:
                        context_depth_img = self._frame_cache[joint_idx]["depth"]
                        if self.config.DEPTH_SENSOR.ADD_REDWOOD_NOISE:
                            assert self._redwood_depth_noise_rand_nums is not None
                            assert (str(self.current_scene_name), str(self.config.AGENT_0.EPISODE_ID)) in\
                                   self._redwood_depth_noise_rand_nums
                            redwood_depth_noise_rand_nums_thsScnEpID =\
                                self._redwood_depth_noise_rand_nums[(str(self.current_scene_name), str(self.config.AGENT_0.EPISODE_ID))]
                            rand_nums = redwood_depth_noise_rand_nums_thsScnEpID[step_idx][agent_idx]

                            context_depth_img = simulate_redwood_depth_noise(
                                context_depth_img,
                                self.redwood_depth_noise_dist_model,
                                self.redwood_depth_noise_multiplier,
                                rand_nums,
                            )

                        context_egoLocalOccMap = self.ego_mapper.get_observation(context_depth_img).astype("float32")

                        assert not np.any(np.logical_and(context_egoLocalOccMap[..., 0] != 1, context_egoLocalOccMap[..., 0] != 0))
                        assert not np.any(np.logical_and(context_egoLocalOccMap[..., 1] != 1, context_egoLocalOccMap[..., 1] != 0))
                    else:
                        context_egoLocalOccMap = self.all_scenes_local_ego_occ_maps[self.current_scene_name][joint_idx].astype("float32")

                    if self.config.EGO_LOCAL_OCC_MAP.NUM_CHANNELS == 1:
                        context_egoLocalOccMap = context_egoLocalOccMap[..., :1]
                    elif self.config.EGO_LOCAL_OCC_MAP.NUM_CHANNELS == 2:
                        pass
                    else:
                        raise ValueError

                    lst_ego_local_map.append(context_egoLocalOccMap)

                self._current_context_ego_local_map[:, step_idx, ...] = np.stack(lst_ego_local_map, axis=0)

        return self._current_context_ego_local_map

    def get_current_context_view_pose(self):
        """get current view pose"""
        ref_pose_for_computing_rel_pose = self._nodes_n_azimuths_lists[0][0]

        if self._step_count == 0:
            for step_idx in range(self.max_context_length):
                lst_pose = []
                for agent_idx in range(len(self._nodes_n_azimuths_lists)):
                    joint_idx = (self._nodes_n_azimuths_lists[agent_idx][step_idx][0],
                                self._nodes_n_azimuths_lists[agent_idx][step_idx][1])

                    context_onlyRPose = [
                        joint_idx[0],
                        joint_idx[0],
                        joint_idx[1]
                    ]

                    context_onlyRRelPose =\
                        np.array(self._compute_relative_pose(current_pose=context_onlyRPose,
                                                             ref_pose=ref_pose_for_computing_rel_pose,
                                                             scene_graph=self.graph,
                                                             )).astype("float32")

                    lst_pose.append(context_onlyRRelPose)

                self._current_context_view_pose[:, step_idx, ...] = np.stack(lst_pose, axis=0)

        return self._current_context_view_pose

    def get_current_context_view_rAz(self):
        """get current view receiver node and azimuth"""
        if self._step_count == 0:
            for step_idx in range(self.max_context_length):
                lst_rAz = []
                for agent_idx in range(len(self._nodes_n_azimuths_lists)):
                    joint_idx = (self._nodes_n_azimuths_lists[agent_idx][step_idx][0],
                                self._nodes_n_azimuths_lists[agent_idx][step_idx][1])

                    lst_rAz.append(joint_idx)

                self._current_context_view_rAz[:, step_idx, ...] = np.stack(lst_rAz, axis=0)

        return self._current_context_view_rAz

    def get_previous_context_view_mask(self):
        """get previous view mask (1 if frame sampled by policy, 0 otherwise)"""
        if self._step_count == 0:
            self._previous_context_view_mask[:, self._step_count] = 1
        else:
            assert self._step_count < self.max_context_length
            if self._action == 1:
                pass
            elif self._action == 2:
                assert self._num_views_to_add in [0, 1]
                if self._num_views_to_add == 1:
                    self._previous_context_view_mask[0, self._step_count] = 1
            elif self._action == 3:
                assert self._num_views_to_add in [0, 1]
                if self._num_views_to_add == 1:
                    self._previous_context_view_mask[1, self._step_count] = 1
            elif self._action == 4:
                assert self._num_views_to_add in [0, 1, 2]
                if self._num_views_to_add == 1:
                    self._previous_context_view_mask[0, self._step_count] = 1
                elif self._num_views_to_add == 2:
                    self._previous_context_view_mask[:, self._step_count] = 1
            else:
                raise ValueError

        return self._previous_context_view_mask

    def get_current_context_selfAudio(self):
        """get current spatial audio from self"""
        if self._step_count == 0:
            for step_idx in range(self.max_context_length):
                lst_wavs = []
                for agent_idx in range(len(self._nodes_n_azimuths_lists)):
                    joint_idx = (self._nodes_n_azimuths_lists[agent_idx][step_idx][0],
                                self._nodes_n_azimuths_lists[agent_idx][step_idx][1])

                    anechoic_audio_slice_startSampleIdx_firstContext_thisAgent =\
                        self.lst_anechoicAudio_filenameNstartSamplingIdx[agent_idx][1]
                    anechoic_audio_slice_startSampleIdx_thisContext_thisAgent =\
                        anechoic_audio_slice_startSampleIdx_firstContext_thisAgent +\
                        int(self.rir_sampling_rate * self._anechoic_audio_slice_length * step_idx)
                    anechoic_audio_thisAgent =\
                        self._anechoic_filename_2_audioData[self.lst_anechoicAudio_filenameNstartSamplingIdx[agent_idx][0]]

                    anechoic_audio_slice_startSampleIdx_thisContext_thisAgent =\
                        anechoic_audio_slice_startSampleIdx_thisContext_thisAgent % anechoic_audio_thisAgent.shape[0]

                    if anechoic_audio_slice_startSampleIdx_thisContext_thisAgent +\
                            int(self.rir_sampling_rate * self._anechoic_audio_slice_length) >\
                            anechoic_audio_thisAgent.shape[0]:
                        anechoic_audio_slice_prefix =\
                            anechoic_audio_thisAgent[anechoic_audio_slice_startSampleIdx_thisContext_thisAgent:]
                        anechoic_audio_slice_suffix =\
                            anechoic_audio_thisAgent[:anechoic_audio_slice_startSampleIdx_thisContext_thisAgent\
                                                      + int(self.rir_sampling_rate * self._anechoic_audio_slice_length)\
                                                      - anechoic_audio_thisAgent.shape[0]]
                        anechoic_audio_slice = np.concatenate([anechoic_audio_slice_prefix, anechoic_audio_slice_suffix])
                    else:
                        anechoic_audio_slice =\
                            anechoic_audio_thisAgent[anechoic_audio_slice_startSampleIdx_thisContext_thisAgent:\
                                                     anechoic_audio_slice_startSampleIdx_thisContext_thisAgent +\
                                                     int(self.rir_sampling_rate * self._anechoic_audio_slice_length)]

                    """context self audio"""
                    context_selfAudio = self._compute_audio(scene=self.current_scene_name,
                                                            azimuth=int(joint_idx[1]),
                                                            receiver_node=int(joint_idx[0]),
                                                            source_node=int(joint_idx[0]),
                                                            anechoic_audio_slice=anechoic_audio_slice,
                                                            )

                    lst_wavs.append(context_selfAudio)

                self._current_context_selfAudio[:, step_idx, ...] = np.stack(lst_wavs, axis=0)

        return self._current_context_selfAudio

    def get_current_context_otherAudio(self):
        """get current spatial audio from the other ego"""
        if self._step_count == 0:
            for step_idx in range(self.max_context_length):
                lst_wavs = []
                for agent_idx in range(len(self._nodes_n_azimuths_lists)):
                    if agent_idx == 0:
                        other_agent_idx = 1
                    else:
                        other_agent_idx = 0

                    joint_idx = (self._nodes_n_azimuths_lists[agent_idx][step_idx][0],
                                 self._nodes_n_azimuths_lists[other_agent_idx][step_idx][0],
                                 self._nodes_n_azimuths_lists[agent_idx][step_idx][1])

                    anechoic_audio_slice_startSampleIdx_firstContext_thisAgent =\
                        self.lst_anechoicAudio_filenameNstartSamplingIdx[other_agent_idx][1]
                    anechoic_audio_slice_startSampleIdx_thisContext_thisAgent =\
                        anechoic_audio_slice_startSampleIdx_firstContext_thisAgent +\
                        int(self.rir_sampling_rate * self._anechoic_audio_slice_length * step_idx)
                    anechoic_audio_thisAgent =\
                        self._anechoic_filename_2_audioData[self.lst_anechoicAudio_filenameNstartSamplingIdx[other_agent_idx][0]]

                    anechoic_audio_slice_startSampleIdx_thisContext_thisAgent =\
                        anechoic_audio_slice_startSampleIdx_thisContext_thisAgent % anechoic_audio_thisAgent.shape[0]

                    if anechoic_audio_slice_startSampleIdx_thisContext_thisAgent +\
                            int(self.rir_sampling_rate * self._anechoic_audio_slice_length) >\
                            anechoic_audio_thisAgent.shape[0]:
                        anechoic_audio_slice_prefix =\
                            anechoic_audio_thisAgent[anechoic_audio_slice_startSampleIdx_thisContext_thisAgent:]
                        anechoic_audio_slice_suffix =\
                            anechoic_audio_thisAgent[:anechoic_audio_slice_startSampleIdx_thisContext_thisAgent\
                                                      + int(self.rir_sampling_rate * self._anechoic_audio_slice_length)\
                                                      - anechoic_audio_thisAgent.shape[0]]
                        anechoic_audio_slice = np.concatenate([anechoic_audio_slice_prefix, anechoic_audio_slice_suffix])
                    else:
                        anechoic_audio_slice =\
                            anechoic_audio_thisAgent[anechoic_audio_slice_startSampleIdx_thisContext_thisAgent:\
                                                     anechoic_audio_slice_startSampleIdx_thisContext_thisAgent +\
                                                     int(self.rir_sampling_rate * self._anechoic_audio_slice_length)]

                    """context audio from other ego"""
                    context_otherAudio = self._compute_audio(scene=self.current_scene_name,
                                                             azimuth=int(joint_idx[2]),
                                                             receiver_node=int(joint_idx[0]),
                                                             source_node=int(joint_idx[1]),
                                                             anechoic_audio_slice=anechoic_audio_slice,
                                                             )

                    lst_wavs.append(context_otherAudio)

                self._current_context_otherAudio[:, step_idx, ...] = np.stack(lst_wavs, axis=0)

        return self._current_context_otherAudio

    def get_current_context_otherAudio_pose(self):
        """get current pose of the other ego relative to this ego"""
        ref_pose_for_computing_rel_pose = self._nodes_n_azimuths_lists[0][0]

        if self._step_count == 0:
            for step_idx in range(self.max_context_length):
                lst_pose = []
                for agent_idx in range(len(self._nodes_n_azimuths_lists)):
                    if agent_idx == 0:
                        other_agent_idx = 1
                    else:
                        other_agent_idx = 0

                    """context sr (audio from other ego) pose"""
                    context_srPose = [self._nodes_n_azimuths_lists[agent_idx][step_idx][0],
                                      self._nodes_n_azimuths_lists[other_agent_idx][step_idx][0],
                                      self._nodes_n_azimuths_lists[agent_idx][step_idx][1]]

                    context_srRelPose =\
                        np.array(self._compute_relative_pose(current_pose=context_srPose,
                                                             ref_pose=ref_pose_for_computing_rel_pose,
                                                             scene_graph=self.graph,
                                                             )).astype("float32")

                    lst_pose.append(context_srRelPose)

                self._current_context_otherAudio_pose[:, step_idx, ...] = np.stack(lst_pose, axis=0)

        return self._current_context_otherAudio_pose

    def get_current_context_audio_mask(self):
        """get current audio mask (depends on which agent(s) is/are currently speaking)"""
        if self._step_count == 0:
            if self.agent_utterance_allSwitches[self._step_count] in [0, 2]:
                self._current_context_audio_mask[0, self._step_count] = 1
            if self.agent_utterance_allSwitches[self._step_count] in [1, 2]:
                self._current_context_audio_mask[1, self._step_count] = 1

        if self._step_count + 1 < self.max_context_length:
            if self.agent_utterance_allSwitches[self._step_count + 1] in [0, 2]:
                self._current_context_audio_mask[0, self._step_count + 1] = 1
            if self.agent_utterance_allSwitches[self._step_count + 1] in [1, 2]:
                self._current_context_audio_mask[1, self._step_count + 1] = 1

        return self._current_context_audio_mask

    def get_all_context_audio_mask(self):
        """get the audio mask for all steps"""
        return self._all_context_audio_mask

    def get_currentNprev_query_globCanMapEgoCrop_gt(self):
        """get the 360 FoV egocentric local target occupancy maps from all steps"""
        if self._step_count == 0:
            for step_idx in range(self.max_context_length):
                lst_query_gt_map = []
                for agent_idx in range(len(self._nodes_n_azimuths_lists)):
                    joint_idx = (self._nodes_n_azimuths_lists[agent_idx][step_idx][0],
                                self.get_rotation_angle_from_azimuth(self._nodes_n_azimuths_lists[agent_idx][step_idx][1]))

                    query_gt_globalCanOccMap_egoCrop =\
                        self.all_scenes_gt_global_canonical_occ_ego_crops_n_rots_maps[self.current_scene_name][joint_idx].astype("float32")

                    if self.config.GT_GLOBAL_CANONICAL_OCC_MAP_EGO_CROP.NUM_CHANNELS == 1:
                        query_gt_globalCanOccMap_egoCrop = query_gt_globalCanOccMap_egoCrop[..., :1]
                    elif self.config.GT_GLOBAL_CANONICAL_OCC_MAP_EGO_CROP.NUM_CHANNELS == 2:
                        query_gt_globalCanOccMap_egoCrop = query_gt_globalCanOccMap_egoCrop
                    else:
                        raise ValueError

                    lst_query_gt_map.append(query_gt_globalCanOccMap_egoCrop)

                self._current_query_globCanMapEgoCrop_gt[:, step_idx] = np.stack(lst_query_gt_map, axis=0)

        return self._current_query_globCanMapEgoCrop_gt

    def get_current_stitched_query_globCanMapEgoCrop_gt(self):
        """get the stitched version of all 360 FoV egocentric local target occupancy maps from all steps"""
        ref_pose_for_computing_rel_pose = self._nodes_n_azimuths_lists[0][0]

        query_stitchedGtGlobCanOccMapsEgoCrops_ph = np.zeros(
            (self.task_cfg.QUERY_STITCHED_GT_GLOB_CAN_MAP_EGO_CROP_SENSOR.SIZE,
             self.task_cfg.QUERY_STITCHED_GT_GLOB_CAN_MAP_EGO_CROP_SENSOR.SIZE,
             self.task_cfg.QUERY_STITCHED_GT_GLOB_CAN_MAP_EGO_CROP_SENSOR.NUM_CHANNELS,
             ),
            dtype="float32",
        )

        used_joint_idxs = []
        for step_idx in range(self.max_context_length - 1):
            if step_idx == 0:
                for agent_idx in range(len(self._nodes_n_azimuths_lists)):
                    joint_idx = (self._nodes_n_azimuths_lists[agent_idx][step_idx][0],
                                self.get_rotation_angle_from_azimuth(self._nodes_n_azimuths_lists[agent_idx][step_idx][1]))
                    if joint_idx not in used_joint_idxs:
                        query_gt_globalCanOccMap_egoCrop =\
                            self.all_scenes_gt_global_canonical_occ_ego_crops_n_rots_maps[self.current_scene_name][joint_idx].astype("float32")

                        if self.task_cfg.QUERY_STITCHED_GT_GLOB_CAN_MAP_EGO_CROP_SENSOR.NUM_CHANNELS == 1:
                            query_gt_globalCanOccMap_egoCrop = query_gt_globalCanOccMap_egoCrop[..., :1]
                        elif self.task_cfg.QUERY_STITCHED_GT_GLOB_CAN_MAP_EGO_CROP_SENSOR.NUM_CHANNELS == 2:
                            query_gt_globalCanOccMap_egoCrop = query_gt_globalCanOccMap_egoCrop
                        else:
                            raise ValueError

                        target_pose = (
                            self._nodes_n_azimuths_lists[agent_idx][step_idx][0],
                            self._nodes_n_azimuths_lists[agent_idx][step_idx][1]
                        )

                        query_stitchedGtGlobCanOccMapsEgoCrops_ph =\
                            self.get_stitched_top_down_maps(stitched_map=query_stitchedGtGlobCanOccMapsEgoCrops_ph,
                                                            stitch_component=query_gt_globalCanOccMap_egoCrop,
                                                            ref_pose=ref_pose_for_computing_rel_pose,
                                                            target_pose=target_pose,
                                                            graph=self.graph,
                                                            is_occupancy=True,
                                                            is_ego_360deg_crops=True,
                                                            map_scale=self.config.GT_GLOBAL_CANONICAL_OCC_MAP_EGO_CROP.SCALE,)

                        used_joint_idxs.append(joint_idx)

            for agent_idx in range(len(self._nodes_n_azimuths_lists)):
                joint_idx = (self._nodes_n_azimuths_lists[agent_idx][step_idx + 1][0],
                            self.get_rotation_angle_from_azimuth(self._nodes_n_azimuths_lists[agent_idx][step_idx + 1][1]))
                if joint_idx not in used_joint_idxs:
                    query_gt_globalCanOccMap_egoCrop = self.all_scenes_gt_global_canonical_occ_ego_crops_n_rots_maps[self.current_scene_name][joint_idx].astype("float32")

                    if self.task_cfg.QUERY_STITCHED_GT_GLOB_CAN_MAP_EGO_CROP_SENSOR.NUM_CHANNELS == 1:
                        query_gt_globalCanOccMap_egoCrop = query_gt_globalCanOccMap_egoCrop[..., :1]
                    elif self.task_cfg.QUERY_STITCHED_GT_GLOB_CAN_MAP_EGO_CROP_SENSOR.NUM_CHANNELS == 2:
                        query_gt_globalCanOccMap_egoCrop = query_gt_globalCanOccMap_egoCrop
                    else:
                        raise ValueError

                    target_pose = (
                        self._nodes_n_azimuths_lists[agent_idx][step_idx + 1][0],
                        self._nodes_n_azimuths_lists[agent_idx][step_idx + 1][1]
                    )

                    query_stitchedGtGlobCanOccMapsEgoCrops_ph =\
                        self.get_stitched_top_down_maps(stitched_map=query_stitchedGtGlobCanOccMapsEgoCrops_ph,
                                                        stitch_component=query_gt_globalCanOccMap_egoCrop,
                                                        ref_pose=ref_pose_for_computing_rel_pose,
                                                        target_pose=target_pose,
                                                        graph=self.graph,
                                                        is_occupancy=True,
                                                        is_ego_360deg_crops=True,
                                                        map_scale=self.config.GT_GLOBAL_CANONICAL_OCC_MAP_EGO_CROP.SCALE,)

                    used_joint_idxs.append(joint_idx)

            self._current_stitched_query_globCanMapEgoCrop_gt[step_idx] = query_stitchedGtGlobCanOccMapsEgoCrops_ph

        return self._current_stitched_query_globCanMapEgoCrop_gt

    def get_currentNprev_query_globCanMapEgoCrop_gt_exploredPartMask(self):
        """get the mask that reveals the explored parts of the  360 FoV egocentric local target occupancy maps from all steps"""
        if self._step_count == 0:
            for step_idx in range(self.max_context_length):
                lst_query_gt_map_exploredPartMask = []
                for agent_idx in range(len(self._nodes_n_azimuths_lists)):
                    joint_idx = (self._nodes_n_azimuths_lists[agent_idx][step_idx][0],
                                self._nodes_n_azimuths_lists[agent_idx][step_idx][1])

                    query_globCanOccMaps_egoCrop_exploredPartMask = self.get_query_globalCanOccMap_egoCrop_exploredPartMask(
                        scene=self.current_scene_name,
                        node=joint_idx[0],
                        az=joint_idx[1],
                    )

                    lst_query_gt_map_exploredPartMask.append(query_globCanOccMaps_egoCrop_exploredPartMask)

                self._current_query_globCanMapEgoCrop_gt_exploredPartMask[:, step_idx] =\
                    np.stack(lst_query_gt_map_exploredPartMask, axis=0)

        return self._current_query_globCanMapEgoCrop_gt_exploredPartMask

    def get_currentNprev_query_mask(self):
        """get the query mask till this episode step (some of the entries could be 0 if there are repetitions in
            agents' pose)"""
        if self._step_count == 0:
            lst_query_mask = []
            for agent_idx in range(len(self._nodes_n_azimuths_lists)):
                joint_idx = (self._nodes_n_azimuths_lists[agent_idx][self._step_count][0],
                             self.get_rotation_angle_from_azimuth(self._nodes_n_azimuths_lists[agent_idx][self._step_count][1]))

                if joint_idx not in self.used_query_nodsNrots:
                    self.used_query_nodsNrots.append(joint_idx)
                    lst_query_mask.append(1.)
                else:
                    lst_query_mask.append(0.)

            self._current_query_mask[:, self._step_count] = np.stack(lst_query_mask, axis=0)

        if self._step_count + 1 < self.max_context_length:
            lst_query_mask = []
            for agent_idx in range(len(self._nodes_n_azimuths_lists)):
                joint_idx = (self._nodes_n_azimuths_lists[agent_idx][self._step_count + 1][0],
                             self.get_rotation_angle_from_azimuth(self._nodes_n_azimuths_lists[agent_idx][self._step_count + 1][1]))

                if joint_idx not in self.used_query_nodsNrots:
                    self.used_query_nodsNrots.append(joint_idx)
                    lst_query_mask.append(1.)
                else:
                    lst_query_mask.append(0.)

            self._current_query_mask[:, self._step_count + 1] = np.stack(lst_query_mask, axis=0)

        return self._current_query_mask

    def get_current_context_idx(self):
        """get current index for indexing the context"""
        return np.array([self._step_count + 1]).astype("float32")

    def get_all_query_mask(self):
        """get the query mask for all episode steps"""
        if self._step_count == 0:
            used_allQuery_nodsNrots = []
            for step_idx in range(self.max_context_length):
                lst_query_mask = []
                for agent_idx in range(len(self._nodes_n_azimuths_lists)):
                    joint_idx = (self._nodes_n_azimuths_lists[agent_idx][step_idx][0],
                                self.get_rotation_angle_from_azimuth(self._nodes_n_azimuths_lists[agent_idx][step_idx][1]))

                    if joint_idx not in used_allQuery_nodsNrots:
                        used_allQuery_nodsNrots.append(joint_idx)
                        lst_query_mask.append(1.)
                    else:
                        lst_query_mask.append(0.)

                self._all_query_mask[:, step_idx] = np.stack(lst_query_mask, axis=0)

        return self._all_query_mask

    def get_episode_scene_idx(self):
        """get episode scene index"""
        episode_scene_idx = SCENE_NAME_TO_IDX[self.scene_dataset][self.current_scene_name]
        return np.array([episode_scene_idx]).astype("float32")

    def get_episode_ref_rAz(self):
        """get the reference receiver node and azimuth for the episode"""
        ref_rAz = self._nodes_n_azimuths_lists[0][0]
        return np.array(ref_rAz).astype("int16")

    def _compute_audio(self,
                       scene='',
                       azimuth=0,
                       receiver_node=0,
                       source_node=0,
                       anechoic_audio_slice=None,
                       ):
        """compute spatial audio by convolving an anechoic audio slice with the appropriate IR"""
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

        imp_full_length = np.zeros((self._max_valid_impulse_length, 9)).astype("float32")
        imp_full_length[:min(imp_full_length.shape[0], sig_imp.shape[0])] =\
            sig_imp[:min(imp_full_length.shape[0], sig_imp.shape[0]), :]

        imp_full_length = self._rotate_ambisonics(imp_full_length, int(azimuth))

        sig_imp = imp_full_length
        assert fs_imp == self.rir_sampling_rate

        assert anechoic_audio_slice.dtype == np.int16
        avg_power = np.power((np.mean(np.power(anechoic_audio_slice.astype("float32"), 2))), 0.5)
        anechoic_audio_slice = anechoic_audio_slice.astype("float32") * self.audio_cfg.ANECHOIC_AUDIO_TARGET_RMS /\
                               (avg_power + 1.0e-13)
        anechoic_audio_slice = anechoic_audio_slice.astype("int16")
        anechoic_audio_slice = anechoic_audio_slice.astype("float32") / 32768

        convolved = []
        for ch_i in range(sig_imp.shape[-1]):
            """using scipy.signal.convolve in place of fftconvolve chooses the fastest algo for convolution when method='auto'"""
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
        """compute the pose of an agent relative to a refernce pose"""
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
        rotation_world_ref = quat_from_angle_axis(np.deg2rad(self.get_rotation_angle_from_azimuth(ref_az)),
                                                  np.array([0, 1, 0]))

        agent_position_xyz = np.array(list(scene_graph.nodes[current_pose[0]]["point"]), dtype=np.float32)
        if self._add_truncated_gaussian_pose_noise:
            assert (current_pose[0], current_pose[-1]) in self._rAz_2_noise_dlX_dlZ_dlAz
            agent_rAz_noise_dlX_dlZ_dlAz = self._rAz_2_noise_dlX_dlZ_dlAz[(current_pose[0], current_pose[-1])]

            agent_position_xyz[0] += agent_rAz_noise_dlX_dlZ_dlAz[0]
            agent_position_xyz[2] += agent_rAz_noise_dlX_dlZ_dlAz[1]

        agent_az = current_pose[2]

        if self._add_truncated_gaussian_pose_noise:
            agent_az += agent_rAz_noise_dlX_dlZ_dlAz[-1]
        rotation_world_agent = quat_from_angle_axis(np.deg2rad(self.get_rotation_angle_from_azimuth(agent_az)),
                                                    np.array([0, 1, 0]))

        agent_position_xyz = quaternion_rotate_vector(
            rotation_world_ref.inverse(), agent_position_xyz - ref_position_xyz
        )

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

        audio_source_position_xyz = audio_source_position_xyz - ref_position_xyz

        # next 2 lines compute relative rotation in the counter-clockwise direction, i.e. -z to -x
        # rotation_world_agent.inverse() * rotation_world_ref = rotation_world_agent - rotation_world_ref
        heading_vector = quaternion_rotate_vector(rotation_world_agent.inverse() * rotation_world_ref,
                                                  np.array([0, 0, -1]))
        agent_heading = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]

        agent_rel_pose = [-agent_position_xyz[2], agent_position_xyz[0], -audio_source_position_xyz[2],
                           audio_source_position_xyz[0], agent_heading]

        return agent_rel_pose

    def get_query_globalCanOccMap_egoCrop_exploredPartMask(self, scene=None, node=None, az=None):
        """compute a mask that reveals the parts of a target topdown map that have been explored in the input maps"""
        query_globalCanOccMap_egoCrop_exploredPartMask =\
            np.ones((self.config.GT_GLOBAL_CANONICAL_OCC_MAP_EGO_CROP.SIZE,
                     self.config.GT_GLOBAL_CANONICAL_OCC_MAP_EGO_CROP.SIZE,
                     self.config.GT_GLOBAL_CANONICAL_OCC_MAP_EGO_CROP.NUM_CHANNELS)).astype("float32")

        rowOrCol_center = (self.config.GT_GLOBAL_CANONICAL_OCC_MAP_EGO_CROP.SIZE // 2) - 1

        bot_row = rowOrCol_center
        top_row = max(rowOrCol_center - (self.config.EGO_LOCAL_OCC_MAP.SIZE - 1), 0)
        left_col = max(rowOrCol_center - ((self.config.EGO_LOCAL_OCC_MAP.SIZE - 1) // 2), 0)
        right_col = left_col + (self.config.EGO_LOCAL_OCC_MAP.SIZE - 1)

        assert 0 <= top_row < query_globalCanOccMap_egoCrop_exploredPartMask.shape[0]
        assert 0 <= bot_row < query_globalCanOccMap_egoCrop_exploredPartMask.shape[0]
        assert 0 <= left_col < query_globalCanOccMap_egoCrop_exploredPartMask.shape[1]
        assert 0 <= right_col < query_globalCanOccMap_egoCrop_exploredPartMask.shape[1]

        local_ego_explored_map =\
            self.all_scenes_local_ego_occ_maps[scene][(node,
                                                       self.get_rotation_angle_from_azimuth(az))][..., 1:].astype("float32")
        local_ego_inv_explored_map = 1. - local_ego_explored_map
        query_globalCanOccMap_egoCrop_exploredPartMask[top_row: bot_row + 1, left_col: right_col + 1] = local_ego_inv_explored_map
        return query_globalCanOccMap_egoCrop_exploredPartMask

    def get_stitched_top_down_maps(self,
                                   stitched_map=None,
                                   stitch_component=None,
                                   ref_pose=None,
                                   target_pose=None,
                                   graph=None,
                                   is_occupancy=False,
                                   is_pred=False,
                                   is_ego_360deg_crops=False,
                                   stitched_map_updateCounter=None,
                                   num_channels=1,
                                   map_scale=0.1):
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
        :param map_scale: scale at which the map is computed
        :return: stitched map
        """
        center = (stitched_map.shape[0] // 2, stitched_map.shape[1] // 2)

        ref_posn = graph.nodes()[ref_pose[0]]['point']
        ref_x = ref_posn[0]
        ref_z = ref_posn[2]

        ref_az = ref_pose[1]

        target_posn = graph.nodes()[target_pose[0]]['point']
        x = target_posn[0]
        z = target_posn[2]

        az = target_pose[1]

        # map_scale = self.config.GT_GLOBAL_CANONICAL_OCC_MAP_EGO_CROP.SCALE
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
