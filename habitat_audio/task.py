# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Type, Union, Tuple

import math
import numpy as np
from gym import spaces

from habitat.config import Config
from habitat.core.dataset import Episode

from habitat.utils.visualizations import maps
from habitat.tasks.nav.nav import NavigationTask, Measure, EmbodiedTask, SimulatorTaskAction
from habitat.core.registry import registry
from habitat.core.simulator import (
    Sensor,
    SensorTypes,
    Simulator,
)

from habitat.core.utils import not_none_validator, try_cv2_import
from habitat.utils.visualizations import fog_of_war, maps

from habitat.sims.habitat_simulator.actions import HabitatSimActions

from habitat.tasks.utils import (
    cartesian_to_polar,
    quaternion_rotate_vector
)

cv2 = try_cv2_import()


def merge_chat2map_episode_simConfig(
    sim_config: Config, episode: Type[Episode]
) -> Any:
    """merge episode config with entries in the sim config"""
    sim_config.defrost()
    # here's where the scene update happens, extract the scene name out of the path
    sim_config.SCENE = episode.scene_id
    sim_config.freeze()
    if (
        episode.start_position is not None
        and episode.start_rotation is not None
    ):
        agent_name = sim_config.AGENTS[sim_config.DEFAULT_AGENT_ID]
        agent_cfg = getattr(sim_config, agent_name)
        agent_cfg.defrost()
        agent_cfg.EPISODE_ID = episode.episode_id
        agent_cfg.NODES_N_AZIMUTHS = episode.info["nodes_n_azimuths"]

        agent_cfg.UTTERANCE_SWITCHES = []
        if 'utterance_switches' in episode.info:
            agent_cfg.UTTERANCE_SWITCHES = episode.info['utterance_switches']
            
        agent_cfg.ANECHOIC_AUDIO_FILENAME_N_START_SAMPLING_IDX = []
        if 'anechoicSound_fileName' in episode.info:

            agent_cfg.ANECHOIC_AUDIO_FILENAME_N_START_SAMPLING_IDX = [(
                    episode.info['anechoicSound_fileName'],
                    episode.info['anechoicSound_samplingStartIdx']
                )]

        agent_cfg.OTHER_ACTIONS = [] 
        agent_cfg.OTHER_NODES_N_AZIMUTHS = []
        for startInfo_idx in range(len(episode.other_info)):
            agent_cfg.OTHER_NODES_N_AZIMUTHS.append(episode.other_info[startInfo_idx]['nodes_n_azimuths'])

            if 'anechoicSound_fileName' in episode.info:
                agent_cfg.ANECHOIC_AUDIO_FILENAME_N_START_SAMPLING_IDX.append(
                    (
                        episode.other_info[startInfo_idx]['anechoicSound_fileName'],
                        episode.other_info[startInfo_idx]['anechoicSound_samplingStartIdx']
                    )
                )

        agent_cfg.IS_SET_START_STATE = True
        agent_cfg.freeze()
    return sim_config


@registry.register_task(name="chat2map")
class Chat2MapTask(NavigationTask):
    def overwrite_sim_config(
        self, sim_config: Any, episode: Type[Episode]
    ) -> Any:
        return merge_chat2map_episode_simConfig(sim_config, episode)

    def _check_episode_is_active(self, *args: Any, **kwargs: Any) -> bool:
        return self._sim._is_episode_active


@registry.register_sensor
class ContextRGBSensor(Sensor):
    """RGB per agent to form part of context view
    """

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "current_context_rgb"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        assert hasattr(self.config, 'FEATURE_SHAPE')
        sensor_shape = self.config.FEATURE_SHAPE

        return spaces.Box(
            low=0,
            high=255,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, observations, episode: Episode, **kwargs: Any):
        return self._sim.get_current_context_rgb()


@registry.register_sensor
class ContextEgoLocalMapSensor(Sensor):
    """Ego local occ map per agent to form part of context view
    """

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "current_context_map"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        assert hasattr(self.config, 'FEATURE_SHAPE')
        sensor_shape = self.config.FEATURE_SHAPE

        return spaces.Box(
            low=0,
            high=1,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, observations, episode: Episode, **kwargs: Any):
        return self._sim.get_current_context_ego_local_map()


@registry.register_sensor
class ContextViewPoseSensor(Sensor):
    """Context view pose per agent
    """

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "current_context_pose"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        assert hasattr(self.config, 'FEATURE_SHAPE')
        sensor_shape = self.config.FEATURE_SHAPE

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, observations, episode: Episode, **kwargs: Any):
        return self._sim.get_current_context_view_pose()


@registry.register_sensor
class ContextViewRAzSensor(Sensor):
    """Context view receiver node and azimuth angle per agent
    """

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "current_context_rAz"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        assert hasattr(self.config, 'FEATURE_SHAPE')
        sensor_shape = self.config.FEATURE_SHAPE

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, observations, episode: Episode, **kwargs: Any):
        return self._sim.get_current_context_view_rAz()


@registry.register_sensor
class PreviousContextViewMaskSensor(Sensor):
    """Previous context view mask per agent (depends on if previous frame was sampled by policy or not)
    """

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "previous_context_view_mask"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        assert hasattr(self.config, 'FEATURE_SHAPE')
        sensor_shape = self.config.FEATURE_SHAPE

        return spaces.Box(
            low=0,
            high=1,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, observations, episode: Episode, **kwargs: Any):
        return self._sim.get_previous_context_view_mask()


@registry.register_sensor
class ContextSelfAudioSensor(Sensor):
    """Context self audio waveforms per agent
    """

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "current_context_selfAudio"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        assert hasattr(self.config, 'FEATURE_SHAPE')
        sensor_shape = self.config.FEATURE_SHAPE

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, observations, episode: Episode, **kwargs: Any):
        return self._sim.get_current_context_selfAudio()


@registry.register_sensor
class ContextOtherAudioSensor(Sensor):
    """Context audio waveforms from the other ego per agent
    """

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "current_context_otherAudio"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        assert hasattr(self.config, 'FEATURE_SHAPE')
        sensor_shape = self.config.FEATURE_SHAPE

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, observations, episode: Episode, **kwargs: Any):
        return self._sim.get_current_context_otherAudio()


@registry.register_sensor
class ContextOtherAudioPoseSensor(Sensor):
    """Context pose of other ego relative to this ego per agent
    """

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "current_context_otherAudio_pose"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        assert hasattr(self.config, 'FEATURE_SHAPE')
        sensor_shape = self.config.FEATURE_SHAPE

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, observations, episode: Episode, **kwargs: Any):
        return self._sim.get_current_context_otherAudio_pose()


@registry.register_sensor
class ContextAudioMaskSensor(Sensor):
    """Current audio mask per agent
    """

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "current_context_audio_mask"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        assert hasattr(self.config, 'FEATURE_SHAPE')
        sensor_shape = self.config.FEATURE_SHAPE

        return spaces.Box(
            low=0,
            high=1,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, observations, episode: Episode, **kwargs: Any):
        return self._sim.get_current_context_audio_mask()


@registry.register_sensor
class AllContextAudioMaskSensor(Sensor):
    """Context audio mask for all steps per agent
    """

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "all_context_audio_mask"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        assert hasattr(self.config, 'FEATURE_SHAPE')
        sensor_shape = self.config.FEATURE_SHAPE

        return spaces.Box(
            low=0,
            high=1,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, observations, episode: Episode, **kwargs: Any):
        return self._sim.get_all_context_audio_mask()


@registry.register_sensor
class ContextIdxSensor(Sensor):
    """current context index
    """

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "current_context_idx"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        assert hasattr(self.config, 'FEATURE_SHAPE')
        sensor_shape = self.config.FEATURE_SHAPE

        return spaces.Box(
            low=0,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, observations, episode: Episode, **kwargs: Any):
        return self._sim.get_current_context_idx()


@registry.register_sensor
class QueryGtGlobCanMapEgoCropSensor(Sensor):
    """360 degree FoV ground-truth global canonical map ego crop per agent
    """

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "current_query_globCanMapEgoCrop_gt"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        assert hasattr(self.config, 'FEATURE_SHAPE')
        sensor_shape = self.config.FEATURE_SHAPE

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, observations, episode: Episode, **kwargs: Any):
        return self._sim.get_currentNprev_query_globCanMapEgoCrop_gt()


@registry.register_sensor
class QueryStitchedGtGlobCanMapEgoCropSensor(Sensor):
    """Stitched 360 degree FoV ground-truth global canonical map ego crop for all agents
    """

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "current_sitched_query_globCanMapEgoCrop_gt"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        assert hasattr(self.config, 'FEATURE_SHAPE')
        sensor_shape = self.config.FEATURE_SHAPE

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, observations, episode: Episode, **kwargs: Any):
        return self._sim.get_current_stitched_query_globCanMapEgoCrop_gt()


@registry.register_sensor
class QueryGtGlobCanMapEgoCropExploredPartMaskSensor(Sensor):
    """Masks showing explored parts for 360 degree FoV ground-truth global canonical map ego crop per agent
    """

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "current_query_globCanMapEgoCrop_gt_exploredPartMask"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        assert hasattr(self.config, 'FEATURE_SHAPE')
        sensor_shape = self.config.FEATURE_SHAPE

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, observations, episode: Episode, **kwargs: Any):
        return self._sim.get_currentNprev_query_globCanMapEgoCrop_gt_exploredPartMask()


@registry.register_sensor
class QueryMaskSensor(Sensor):
    """Query mask per agent till current episode step
    """

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "current_query_mask"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        assert hasattr(self.config, 'FEATURE_SHAPE')
        sensor_shape = self.config.FEATURE_SHAPE

        return spaces.Box(
            low=0,
            high=1,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, observations, episode: Episode, **kwargs: Any):
        return self._sim.get_currentNprev_query_mask()


@registry.register_sensor
class AllQueryMaskSensor(Sensor):
    """All query mask per agent
    """

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "all_query_mask"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        assert hasattr(self.config, 'FEATURE_SHAPE')
        sensor_shape = self.config.FEATURE_SHAPE

        return spaces.Box(
            low=0,
            high=1,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, observations, episode: Episode, **kwargs: Any):
        return self._sim.get_all_query_mask()


@registry.register_sensor
class EpisodeSceneIdxSensor(Sensor):
    """Scene idx for this episode
    """

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "episode_scene_idx"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        assert hasattr(self.config, 'FEATURE_SHAPE')
        sensor_shape = self.config.FEATURE_SHAPE

        return spaces.Box(
            low=0,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, observations, episode: Episode, **kwargs: Any):
        return self._sim.get_episode_scene_idx()


@registry.register_sensor
class EpisodeRefRAzSensor(Sensor):
    """Reference receiver node and azimuth angle for this episode
    """

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "episode_ref_rAz"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        assert hasattr(self.config, 'FEATURE_SHAPE')
        sensor_shape = self.config.FEATURE_SHAPE

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, observations, episode: Episode, **kwargs: Any):
        return self._sim.get_episode_ref_rAz()
