# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    """Flatten any tensor into a 1D tensor with batch size as the leading dimension"""

    def forward(self, x):
        return x.view(x.size(0), -1)


class CustomFixedCategorical(torch.distributions.Categorical):
    """
    custom categorical net
    """

    def sample(self, sample_shape=torch.Size()):
        return super().sample(sample_shape).unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

    def get_probs(self):
        return self.probs

    def get_log_probs(self):
        return torch.log(self.probs + 1e-7)


class CategoricalNet(nn.Module):
    """
    categorical net
    """

    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.linear = nn.Linear(num_inputs, num_outputs)

        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        x = self.linear(x)
        return CustomFixedCategorical(logits=x)


def linear_decay(epoch: int, total_num_updates: int) -> float:
    """
    Returns a multiplicative factor for linear value decay
    :param epoch: current epoch number
    :param total_num_updates: total number of epochs
    :return: multiplicative factor that decreases param value linearly
    """

    return 1 - (epoch / float(total_num_updates))


def to_tensor(v, device=None):
    if torch.is_tensor(v):
        return v.to(device=device, dtype=torch.float)
    elif isinstance(v, np.ndarray):
        return (torch.from_numpy(v)).to(device=device, dtype=torch.float)
    else:
        return torch.tensor(v, dtype=torch.float, device=device)


def batch_obs(
    observations: List[Dict], device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    Transpose a batch of observation dicts to a dict of batched observations.
    :param observations: list of dicts of observations.
    :param device: The torch.device to put the resulting tensors on. Will not move the tensors if None
    :return: transposed dict of lists of observations.
    """

    batch: DefaultDict[str, List] = defaultdict(list)
    for obs in observations:
        for sensor in obs:
            batch[sensor].append(to_tensor(obs[sensor], device=device))

    for sensor in batch:
        batch[sensor] = torch.stack(batch[sensor], dim=0)

    return batch


def poll_checkpoint_folder(
    checkpoint_folder: str, previous_ckpt_ind: int, eval_interval: int
) -> Optional[str]:
    """
    Return (previous_ckpt_ind + 1)th checkpoint in checkpoint folder
    (sorted by time of last modification).
    :param checkpoint_folder: directory to look for checkpoints.
    :param previous_ckpt_ind: index of checkpoint last returned.
    :param eval_interval: number of checkpoints between two evaluation
    :return: checkpoint path if (previous_ckpt_ind + 1)th checkpoint is found else return None.
    """

    assert os.path.isdir(checkpoint_folder), (
        f"invalid checkpoint folder " f"path {checkpoint_folder}"
    )
    models_paths = list(
        filter(os.path.isfile, glob.glob(checkpoint_folder + "/*"))
    )
    models_paths.sort(key=os.path.getmtime)
    ind = previous_ckpt_ind + eval_interval
    if ind < len(models_paths):
        return models_paths[ind]
    return None


def load_points(points_file: str, transform=True, scene_dataset="replica"):
    """
    Helper method to load points data from files stored on disk and transform if necessary
    :param points_file: path to files containing points data
    :param transform: transform coordinate systems of loaded points for use in Habitat or not
    :param scene_dataset: name of scenes dataset ("replica", "mp3d", etc.)
    :return: points in transformed coordinate system for use with Habitat
    """
    points_data = np.loadtxt(points_file, delimiter="\t")
    if transform:
        if scene_dataset == "replica":
            points = list(zip(
                points_data[:, 1],
                points_data[:, 3] - 1.5528907,
                -points_data[:, 2])
            )
        elif scene_dataset == "mp3d":
            points = list(zip(
                points_data[:, 1],
                points_data[:, 3] - 1.5,
                -points_data[:, 2])
            )
        else:
            raise NotImplementedError
    else:
        points = list(zip(
            points_data[:, 1],
            points_data[:, 2],
            points_data[:, 3])
        )
    points_index = points_data[:, 0].astype(int)
    points_dict = dict(zip(points_index, points))
    assert list(points_index) == list(range(len(points)))
    return points_dict, points


def load_points_data(parent_folder, graph_file, transform=True, scene_dataset="replica"):
    """
    Main method to load points data from files stored on disk and transform if necessary
    :param parent_folder: parent folder containing files with points data
    :param graph_file: files containing connectivity of points per scene
    :param transform: transform coordinate systems of loaded points for use in Habitat or not
    :param scene_dataset: name of scenes dataset ("replica", "mp3d", etc.)
    :return: 1. points in transformed coordinate system for use with Habitat
             2. graph object containing information about the connectivity of points in a scene
    """
    points_file = os.path.join(parent_folder, 'points.txt')
    graph_file = os.path.join(parent_folder, graph_file)

    points = None
    if os.path.isfile(points_file): 
        _, points = load_points(points_file, transform=transform, scene_dataset=scene_dataset)
    if not os.path.exists(graph_file):
        raise FileExistsError(graph_file + ' does not exist!')
    else:
        with open(graph_file, 'rb') as fo:
            graph = pickle.load(fo)

    return points, graph


def _to_tensor(v):
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)


def get_stitched_top_down_maps(stitched_map=None,
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
    :param map_scale: scale at which the map has been computed
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
