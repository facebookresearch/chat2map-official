# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union
import os
import shutil

from habitat import get_config as get_task_config
from habitat.config import Config as CN
from habitat.config.default import SIMULATOR_SENSOR
import habitat

DEFAULT_CONFIG_DIR = "configs/"
CONFIG_FILE_SEPARATOR = ","
# -----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# -----------------------------------------------------------------------------
_C = CN()
_C.SEED = 0
_C.BASE_TASK_CONFIG_PATH = "configs/tasks/pointnav.yaml"
_C.TASK_CONFIG = CN()  # task_config will be stored as a config node
_C.CMD_TRAILING_OPTS = []  # store command line options as list of strings
_C.TRAINER_NAME = "ppo"
_C.ENV_NAME = "DummyHabitatEnv"
_C.SIMULATOR_GPU_ID = 0
_C.TORCH_GPU_ID = 0
_C.PARALLEL_GPU_IDS = []
_C.MODEL_DIR = ''
_C.TENSORBOARD_DIR = "tb"
_C.VIDEO_OPTION = []
_C.OUTPUT_DUMP_OPTION = []
_C.DUMP_CONTEXT_INPUTS = False
_C.DUMP_NN_INPUT = False
_C.STITCH_TOP_DOWN_MAPS = False
_C.STITCH_TOP_DOWN_MAPS_ACTIVE_MAPPING = False
_C.VIDEO_DIR = ''		# video_dir
_C.AUDIO_DIR = ''		# audio_dir
_C.VISUALIZATION_OPTION = []		# top_down_map
_C.EVAL_CKPT_PATH = "data/checkpoints"		# path to ckpt or path to ckpts dir
_C.NUM_PROCESSES = 16
_C.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
_C.CHECKPOINT_FOLDER = "data/checkpoints"
_C.NUM_UPDATES = 10000
_C.LOG_INTERVAL = 10
_C.LOG_FILE = "train.log"
_C.CHECKPOINT_INTERVAL = 50
_C.USE_VECENV = True
_C.USE_SYNC_VECENV = False
_C.EXTRA_RGB = False
_C.EXTRA_DEPTH = False
_C.DEBUG = False
_C.EPS_SCENES = []
_C.EPS_SCENES_N_IDS = []
_C.JOB_ID = 1
_C.RESUME_FROM_STATE_DICT = False
_C.STATE_DICT_PATH = None
_C.RESUME_AFTER_PREEMPTION = False
_C.VAL_WHILE_TRAINING = False
# -----------------------------------------------------------------------------
# EVAL CONFIG
# -----------------------------------------------------------------------------
_C.EVAL = CN()
_C.EVAL.SPLIT = None		# the split to evaluate on
_C.EVAL.USE_CKPT_CONFIG = False
_C.EVAL.DATA_PARALLEL_TRAINING = False
_C.EVAL.EPISODE_COUNT = 500
# -----------------------------------------------------------------------------
# REINFORCEMENT LEARNING (RL) ENVIRONMENT CONFIG
# -----------------------------------------------------------------------------
_C.RL = CN()
# -----------------------------------------------------------------------------
# ACTIVE MAPPING WITH PROXIMAL POLICY OPTIMIZATION (PPO)
# -----------------------------------------------------------------------------
_C.RL.PPO = CN()
_C.RL.PPO.clip_param = 0.2
_C.RL.PPO.ppo_epoch = 4
_C.RL.PPO.num_mini_batch = 16
_C.RL.PPO.value_loss_coef = 0.5
_C.RL.PPO.entropy_coef = 0.01
_C.RL.PPO.lr = 1e-3
_C.RL.PPO.eps = 1e-5
_C.RL.PPO.max_grad_norm = 0.5
_C.RL.PPO.num_steps = 5
_C.RL.PPO.hidden_size = 512
_C.RL.PPO.rnn_num_layers = 1
_C.RL.PPO.use_gae = True
_C.RL.PPO.use_linear_lr_decay = False
_C.RL.PPO.use_linear_clip_decay = False
_C.RL.PPO.gamma = 0.99
_C.RL.PPO.tau = 0.95
_C.RL.PPO.policy_type = None 
_C.RL.PPO.reward_type = None 
_C.RL.PPO.reward_window_size = 50
_C.RL.PPO.repeatPose_penalty_weight = 0.1
_C.RL.PPO.deterministic_eval = False
_C.RL.PPO.use_ddppo = False
_C.RL.PPO.ddppo_distrib_backend = "NCCL"
_C.RL.PPO.short_rollout_threshold = 0.25
_C.RL.PPO.sync_frac = 0.6
_C.RL.PPO.master_port = 8738
_C.RL.PPO.master_addr = "127.0.0.1"
_C.RL.PPO.agent_type = "ours" 		# ours, random
_C.RL.PPO.freeze_mapper = False
_C.RL.PPO.pretrained_ckpt_path = None
_C.RL.PPO.PoseEnc = CN()
_C.RL.PPO.PoseEnc.output_size = 32 # 16, 32
_C.RL.PPO.AudioEnc = CN()
_C.RL.PPO.AudioEnc.output_size = 512
_C.RL.PPO.VisualEnc = CN()
_C.RL.PPO.VisualEnc.output_size = 512
_C.RL.PPO.ActionEnc = CN()
_C.RL.PPO.ActionEnc.output_size = 512
_C.RL.PPO.FuseNet = CN()
_C.RL.PPO.FuseNet.output_size = 512
# -----------------------------------------------------------------------------
# PASSIVE MAPPING
# -----------------------------------------------------------------------------
_C.PassiveMapping = CN()
_C.PassiveMapping.lr = 5.0e-4
_C.PassiveMapping.eps = 1.0e-8
_C.PassiveMapping.weight_decay = 0.0
_C.PassiveMapping.max_grad_norm = None
_C.PassiveMapping.betas = [0.9, 0.999]
_C.PassiveMapping.num_epochs = 1000
_C.PassiveMapping.num_mini_batch = 16
_C.PassiveMapping.batch_size = 64 
_C.PassiveMapping.num_workers = 64 
_C.PassiveMapping.num_datapoints_train = 12000
_C.PassiveMapping.TrainLosses = CN()
_C.PassiveMapping.TrainLosses.types = ["bce_loss"]
_C.PassiveMapping.TrainLosses.weights = [1.0]
_C.PassiveMapping.EvalMetrics = CN()
_C.PassiveMapping.EvalMetrics.types = ["f1_score", "iou"]
_C.PassiveMapping.EvalMetrics.type_for_ckpt_dump = "f1_score"
_C.PassiveMapping.MemoryNet = CN()
_C.PassiveMapping.MemoryNet.type = "transformer"
_C.PassiveMapping.MemoryNet.Transformer = CN()
_C.PassiveMapping.MemoryNet.Transformer.input_size = 1024
_C.PassiveMapping.MemoryNet.Transformer.hidden_size = 2048
_C.PassiveMapping.MemoryNet.Transformer.num_encoder_layers = 6
_C.PassiveMapping.MemoryNet.Transformer.num_decoder_layers = 6
_C.PassiveMapping.MemoryNet.Transformer.decoder_out_size = 1024
_C.PassiveMapping.MemoryNet.Transformer.nhead = 8
_C.PassiveMapping.MemoryNet.Transformer.dropout = 0.0
_C.PassiveMapping.MemoryNet.Transformer.activation = "relu"
_C.PassiveMapping.modality_tag_type_encoding_size = 8
_C.PassiveMapping.VisualEnc = CN()
_C.PassiveMapping.VisualEnc.num_out_channels = 64
_C.PassiveMapping.AudioEnc = CN()
_C.PassiveMapping.AudioEnc.num_input_channels = 9
_C.PassiveMapping.PositionalNet = CN()
_C.PassiveMapping.PositionalNet.num_freqs_for_sinusoidal = 8
_C.PassiveMapping.PositionalNet.patch_hwCh = [4, 4, 1024]
# -----------------------------------------------------------------------------
# TASK CONFIG
# -----------------------------------------------------------------------------
_TC = habitat.get_config()
_TC.defrost()
########## SENSORS ###########
# -----------------------------------------------------------------------------
# AMBISONIC WAVEFORM SENSOR
# -----------------------------------------------------------------------------
_TC.TASK.AMBI_WAV_SENSOR = CN()
_TC.TASK.AMBI_WAV_SENSOR.FEATURE_SHAPE = [16000, 9] 
# -----------------------------------------------------------------------------
# CONTEXT_SELF_AUDIO_SENSOR
# -----------------------------------------------------------------------------
_TC.TASK.CONTEXT_SELF_AUDIO_SENSOR = CN()
_TC.TASK.CONTEXT_SELF_AUDIO_SENSOR.TYPE = "ContextSelfAudioSensor"
_TC.TASK.CONTEXT_SELF_AUDIO_SENSOR.FEATURE_SHAPE = [2, 8, 16000, 9]
# -----------------------------------------------------------------------------
# CONTEXT_OTHER_AUDIO_SENSOR
# -----------------------------------------------------------------------------
_TC.TASK.CONTEXT_OTHER_AUDIO_SENSOR = _TC.TASK.CONTEXT_SELF_AUDIO_SENSOR.clone()
_TC.TASK.CONTEXT_OTHER_AUDIO_SENSOR.TYPE = "ContextOtherAudioSensor"
# -----------------------------------------------------------------------------
# QUERY_MASK_SENSOR
# -----------------------------------------------------------------------------
_TC.TASK.QUERY_MASK_SENSOR = CN()
_TC.TASK.QUERY_MASK_SENSOR.TYPE = "QueryMaskSensor"
_TC.TASK.QUERY_MASK_SENSOR.FEATURE_SHAPE = [2, 8]
# -----------------------------------------------------------------------------
# ALL_QUERY_MASK_SENSOR
# -----------------------------------------------------------------------------
_TC.TASK.ALL_QUERY_MASK_SENSOR = CN()
_TC.TASK.ALL_QUERY_MASK_SENSOR.TYPE = "AllQueryMaskSensor"
_TC.TASK.ALL_QUERY_MASK_SENSOR.FEATURE_SHAPE = [2, 8]
# -----------------------------------------------------------------------------
# CONTEXT_RGB_SENSOR
# -----------------------------------------------------------------------------
_TC.TASK.CONTEXT_RGB_SENSOR = _TC.SIMULATOR.RGB_SENSOR.clone()
_TC.TASK.CONTEXT_RGB_SENSOR.TYPE = "ContextRGBSensor"
_TC.TASK.CONTEXT_RGB_SENSOR.FEATURE_SHAPE = [2, 8, 128, 128, 3]		# NUM_AGENTS x MAX_CONTEXT_LENGTH x H x W x C
# -----------------------------------------------------------------------------
# CONTEXT_VIEW_POSE_SENSOR (agent pose sensor)
# -----------------------------------------------------------------------------
_TC.TASK.CONTEXT_VIEW_POSE_SENSOR = CN()
_TC.TASK.CONTEXT_VIEW_POSE_SENSOR.TYPE = "ContextViewPoseSensor"
_TC.TASK.CONTEXT_VIEW_POSE_SENSOR.FEATURE_SHAPE = [2, 8, 5]		# NUM_AGENTS x MAX_CONTEXT_LENGTH x NUM_POSE_ATTRIBUTES
_TC.TASK.CONTEXT_VIEW_POSE_SENSOR.ADD_TRUNCATED_GAUSSIAN_NOISE = False
_TC.TASK.CONTEXT_VIEW_POSE_SENSOR.TRUNCATED_GAUSSIAN_NOISE = CN()
_TC.TASK.CONTEXT_VIEW_POSE_SENSOR.TRUNCATED_GAUSSIAN_NOISE.GAUSSIAN_NOISE_MULTIPLIERS_PATH = "data/noise/pose_noise/seed0.pkl"
_TC.TASK.CONTEXT_VIEW_POSE_SENSOR.TRUNCATED_GAUSSIAN_NOISE.TRANSLATION_MEAN = 0.025
_TC.TASK.CONTEXT_VIEW_POSE_SENSOR.TRUNCATED_GAUSSIAN_NOISE.TRANSLATION_STD = 0.001
_TC.TASK.CONTEXT_VIEW_POSE_SENSOR.TRUNCATED_GAUSSIAN_NOISE.ROTATION_DEGREES_MEAN = 0.9
_TC.TASK.CONTEXT_VIEW_POSE_SENSOR.TRUNCATED_GAUSSIAN_NOISE.ROTATION_DEGREES_STD = 0.057
_TC.TASK.CONTEXT_VIEW_POSE_SENSOR.TRUNCATED_GAUSSIAN_NOISE.NUM_STDS_FOR_TRUNCATION = 2
# -----------------------------------------------------------------------------
# CONTEXT_VIEW_R_N_AZ_SENSOR (context view receiver node and azimuth sensor)
# -----------------------------------------------------------------------------
_TC.TASK.CONTEXT_VIEW_R_N_AZ_SENSOR = CN()
_TC.TASK.CONTEXT_VIEW_R_N_AZ_SENSOR.TYPE = "ContextViewRAzSensor"
_TC.TASK.CONTEXT_VIEW_R_N_AZ_SENSOR.FEATURE_SHAPE = [2, 8, 2]		# NUM_AGENTS x MAX_CONTEXT_LENGTH x 2
# -----------------------------------------------------------------------------
# CONTEXT_OTHER_AUDIO_POSE_SENSOR (pose of other agent relative to a certain agent sensor)
# -----------------------------------------------------------------------------
_TC.TASK.CONTEXT_OTHER_AUDIO_POSE_SENSOR = _TC.TASK.CONTEXT_VIEW_POSE_SENSOR.clone()
_TC.TASK.CONTEXT_OTHER_AUDIO_POSE_SENSOR.TYPE = "ContextOtherAudioPoseSensor"
# -----------------------------------------------------------------------------
# PREV_CONTEXT_VIEW_MASK_SENSOR (mask for previous view in context (input to policy); 1 if sampled, 0 if not)
# -----------------------------------------------------------------------------
_TC.TASK.PREV_CONTEXT_VIEW_MASK_SENSOR = CN()
_TC.TASK.PREV_CONTEXT_VIEW_MASK_SENSOR.TYPE = "PreviousContextViewMaskSensor"
_TC.TASK.PREV_CONTEXT_VIEW_MASK_SENSOR.FEATURE_SHAPE = [2, 8]
# -----------------------------------------------------------------------------
# CONTEXT_EGO_LOCAL_MAP_SENSOR (input ego local map sensor)
# -----------------------------------------------------------------------------
_TC.TASK.CONTEXT_EGO_LOCAL_MAP_SENSOR = CN()
_TC.TASK.CONTEXT_EGO_LOCAL_MAP_SENSOR.TYPE = "ContextEgoLocalMapSensor"
_TC.TASK.CONTEXT_EGO_LOCAL_MAP_SENSOR.SIZE = 31
_TC.TASK.CONTEXT_EGO_LOCAL_MAP_SENSOR.NUM_CHANNELS = 2
_TC.TASK.CONTEXT_EGO_LOCAL_MAP_SENSOR.FEATURE_SHAPE = [2, 8, 31, 31, 2]		# NUM_AGENTS x MAX_CONTEXT_LENGTH x MAP_SIZE x MAP_SIZE x NUM_CHANNELS
# -----------------------------------------------------------------------------
# CONTEXT_STITCHED_EGO_LOCAL_MAP_SENSOR (placeholder sensor config to set correct sizes)
# -----------------------------------------------------------------------------
_TC.TASK.CONTEXT_STITCHED_EGO_LOCAL_MAP_SENSOR = CN()
_TC.TASK.CONTEXT_STITCHED_EGO_LOCAL_MAP_SENSOR.TYPE = "ContextStitchedEgoLocalMapSensor"
_TC.TASK.CONTEXT_STITCHED_EGO_LOCAL_MAP_SENSOR.SIZE = 1284
_TC.TASK.CONTEXT_STITCHED_EGO_LOCAL_MAP_SENSOR.NUM_CHANNELS = 2
_TC.TASK.CONTEXT_STITCHED_EGO_LOCAL_MAP_SENSOR.FEATURE_SHAPE = [7, 1284, 1284, 2]  
# -----------------------------------------------------------------------------
# CONTEXT_IDX_SENSOR (context index sensor)
# -----------------------------------------------------------------------------
_TC.TASK.CONTEXT_IDX_SENSOR = CN()
_TC.TASK.CONTEXT_IDX_SENSOR.TYPE = "ContextIdxSensor"
_TC.TASK.CONTEXT_IDX_SENSOR.FEATURE_SHAPE = [1]
# -----------------------------------------------------------------------------
# QUERY_GT_GLOB_CAN_MAP_EGO_CROP_SENSOR (query gt global canonical map ego crop sensor)
# -----------------------------------------------------------------------------
_TC.TASK.QUERY_GT_GLOB_CAN_MAP_EGO_CROP_SENSOR = CN()
_TC.TASK.QUERY_GT_GLOB_CAN_MAP_EGO_CROP_SENSOR.TYPE = "QueryGtGlobCanMapEgoCropSensor"
_TC.TASK.QUERY_GT_GLOB_CAN_MAP_EGO_CROP_SENSOR.FEATURE_SHAPE = [2, 8, 64, 64, 2]		# NUM_AGENTS, MAX_CONTEXT_LENGTH, MAP_SIZE, MAP_SIZE, NUM_CHANNELS
_TC.TASK.QUERY_GT_GLOB_CAN_MAP_EGO_CROP_SENSOR.SIZE = 64
_TC.TASK.QUERY_GT_GLOB_CAN_MAP_EGO_CROP_SENSOR.NUM_CHANNELS = 2
_TC.TASK.QUERY_GT_GLOB_CAN_MAP_EGO_CROP_SENSOR.SCALE = 0.1
# -----------------------------------------------------------------------------
# QUERY_GT_GLOB_CAN_MAP_EGO_CROP_EXPLORED_PART_MASK_SENSOR (mask for query gt global canonical map ego crop  sensor)
# -----------------------------------------------------------------------------
_TC.TASK.QUERY_GT_GLOB_CAN_MAP_EGO_CROP_EXPLORED_PART_MASK_SENSOR = CN()
_TC.TASK.QUERY_GT_GLOB_CAN_MAP_EGO_CROP_EXPLORED_PART_MASK_SENSOR.TYPE = "QueryGtGlobCanMapEgoCropExploredPartMaskSensor"
_TC.TASK.QUERY_GT_GLOB_CAN_MAP_EGO_CROP_EXPLORED_PART_MASK_SENSOR.FEATURE_SHAPE = [2, 8, 64, 64, 2]		# NUM_AGENTS, MAX_CONTEXT_LENGTH, MAP_SIZE, MAP_SIZE, NUM_CHANNELS
_TC.TASK.QUERY_GT_GLOB_CAN_MAP_EGO_CROP_EXPLORED_PART_MASK_SENSOR.SIZE = 64
_TC.TASK.QUERY_GT_GLOB_CAN_MAP_EGO_CROP_EXPLORED_PART_MASK_SENSOR.NUM_CHANNELS = 2
_TC.TASK.QUERY_GT_GLOB_CAN_MAP_EGO_CROP_EXPLORED_PART_MASK_SENSOR.SCALE = 0.1
# -----------------------------------------------------------------------------
# QUERY_STITCHED_GT_GLOB_CAN_MAP_EGO_CROP_SENSOR (ground-truth stitched global canonical map ego crops for query sensor) 
# -----------------------------------------------------------------------------
_TC.TASK.QUERY_STITCHED_GT_GLOB_CAN_MAP_EGO_CROP_SENSOR = CN()
_TC.TASK.QUERY_STITCHED_GT_GLOB_CAN_MAP_EGO_CROP_SENSOR.TYPE = "QueryStitchedGtGlobCanMapEgoCropSensor"
_TC.TASK.QUERY_STITCHED_GT_GLOB_CAN_MAP_EGO_CROP_SENSOR.FEATURE_SHAPE = [7, 1284, 1284, 2]		# MAX_CONTEXT_LENGTH - 1, STITCHED MAP SIZE, STITCHED MAP SIZE, NUM_CHANNELS
_TC.TASK.QUERY_STITCHED_GT_GLOB_CAN_MAP_EGO_CROP_SENSOR.SIZE = 1284
_TC.TASK.QUERY_STITCHED_GT_GLOB_CAN_MAP_EGO_CROP_SENSOR.NUM_CHANNELS = 2
# -----------------------------------------------------------------------------
# CONTEXT_AUDIO_MASK_SENSOR (current context audio mask sensor)
# -----------------------------------------------------------------------------
_TC.TASK.CONTEXT_AUDIO_MASK_SENSOR = CN()
_TC.TASK.CONTEXT_AUDIO_MASK_SENSOR.TYPE = "ContextAudioMaskSensor"
_TC.TASK.CONTEXT_AUDIO_MASK_SENSOR.FEATURE_SHAPE = [2, 8] 		# NUM_AGENTS, MAX_CONTEXT_LENGTH
# -----------------------------------------------------------------------------
# ALL_CONTEXT_AUDIO_MASK_SENSOR (all context audio mask sensor)
# -----------------------------------------------------------------------------
_TC.TASK.ALL_CONTEXT_AUDIO_MASK_SENSOR = CN()
_TC.TASK.ALL_CONTEXT_AUDIO_MASK_SENSOR.TYPE = "AllContextAudioMaskSensor"
_TC.TASK.ALL_CONTEXT_AUDIO_MASK_SENSOR.FEATURE_SHAPE = [2, 8]		# NUM_AGENTS, MAX_CONTEXT_LENGTH
# -----------------------------------------------------------------------------
# EPISODE_SCENE_IDX_SENSOR (scene index to get scene name from a dictionary defined in the file where the sensor output is being used)
# -----------------------------------------------------------------------------
_TC.TASK.EPISODE_SCENE_IDX_SENSOR = CN()
_TC.TASK.EPISODE_SCENE_IDX_SENSOR.TYPE = "EpisodeSceneIdxSensor"
_TC.TASK.EPISODE_SCENE_IDX_SENSOR.FEATURE_SHAPE = [1]
# -----------------------------------------------------------------------------
# EPISODE_REF_RECEIVER_AZIMUTH_SENSOR (reference receiver node and azimuth sensor for stiching map)
# -----------------------------------------------------------------------------
_TC.TASK.EPISODE_REF_RECEIVER_AZIMUTH_SENSOR = CN()
_TC.TASK.EPISODE_REF_RECEIVER_AZIMUTH_SENSOR.TYPE = "EpisodeRefRAzSensor"
_TC.TASK.EPISODE_REF_RECEIVER_AZIMUTH_SENSOR.FEATURE_SHAPE = [2]
# -----------------------------------------------------------------------------
# environment config
# -----------------------------------------------------------------------------
_TC.ENVIRONMENT.MAX_EPISODE_STEPS = 7
_TC.ENVIRONMENT.MAX_CONTEXT_LENGTH = 8
_TC.ENVIRONMENT.VISUAL_BUDGET = 6 		# VISUAL_BUDGET = NUM_AGENTS + ACTUAL_VISUAL_BUDGET
_TC.ENVIRONMENT.VISUAL_BUDGET_COMPARE_AGAINST_LAST_TOP_DOWN_MAPS = 6
_TC.ENVIRONMENT.MAX_QUERY_LENGTH = 16 		# NUM_AGENTS * MAX_CONTEXT_LENGTH
# -----------------------------------------------------------------------------
# simulator config
# -----------------------------------------------------------------------------
_TC.SIMULATOR.SEED = -1
_TC.SIMULATOR.SCENE_DATASET = "mp3d"
_TC.SIMULATOR.MAX_EPISODE_STEPS = 7 
_TC.SIMULATOR.MAX_CONTEXT_LENGTH = 8
_TC.SIMULATOR.VISUAL_BUDGET = 100
_TC.SIMULATOR.GRID_SIZE = 1.0
_TC.SIMULATOR.USE_RENDERED_OBSERVATIONS = False
_TC.SIMULATOR.RENDERED_OBSERVATIONS = "data/scene_observations/"
_TC.SIMULATOR.RENDER_LOCAL_EGO_OCC_MAPS_FROM_DEPTH_IMAGES = False
_TC.SIMULATOR.RENDERED_LOCAL_EGO_OCC_MAPS_DIR = "data/gt_topdown_maps/occupancy/localEgoMaps_perScenePerRRotn/mp3d/mp_sz31_scl01"
_TC.SIMULATOR.RENDERED_GT_GLOBAL_CANONICAL_OCC_MAP_EGO_CROP_N_ROTS_DIR = "data/gt_topdown_maps/occupancy/stitchedEgoMaps_egoCropRot_perScenePerRRotn/mp3d/mp_sz64_scl01_srcEgLclMp_sz51"
# -----------------------------------------------------------------------------
# SIMULATOR DEPTH SENSOR
# -----------------------------------------------------------------------------
_TC.SIMULATOR.DEPTH_SENSOR.ADD_REDWOOD_NOISE = False
_TC.SIMULATOR.DEPTH_SENSOR.REDWOOD_NOISE_RAND_NUMS_PATH = "data/noise/redwood_depth_sensor_noise/seed0_h128w128.pkl"
_TC.SIMULATOR.DEPTH_SENSOR.REDWOOD_NOISE_MULTIPLIER = 1.0
_TC.SIMULATOR.DEPTH_SENSOR.REDWOOD_DEPTH_NOISE_DIST_MODEL = "data/noise/redwood_depth_sensor_noise/redwood-depth-dist-model.npy"
# -----------------------------------------------------------------------------
# CONTEXT_DEPTH_SENSOR (putting in the middle of sim cfg to copy the sim.sensor cfg)
# -----------------------------------------------------------------------------
_TC.TASK.CONTEXT_DEPTH_SENSOR = _TC.SIMULATOR.DEPTH_SENSOR.clone()
_TC.TASK.CONTEXT_DEPTH_SENSOR.TYPE = "ContextDepthSensor"
_TC.TASK.CONTEXT_DEPTH_SENSOR.FEATURE_SHAPE = [2, 8, 128, 128, 1]		# NUM_AGENTS x MAX_CONTEXT_LENGTH x H x W x C
# -----------------------------------------------------------------------------
# SIMULATOR RGB SENSOR
# -----------------------------------------------------------------------------
_TC.SIMULATOR.RGB_SENSOR.ADD_GAUSSIAN_NOISE = False
_TC.SIMULATOR.RGB_SENSOR.GAUSSIAN_NOISE_RAND_NUMS_PATH = "data/noise/gaussian_rgb_sensor_noise/seed0_h128w128_mu0.0sigma1.0.pkl"
_TC.SIMULATOR.RGB_SENSOR.GAUSSIAN_NOISE_MULTIPLIER = 0.2
# -----------------------------------------------------------------------------
# SIMULATOR VIEW_POSE_SENSOR
# -----------------------------------------------------------------------------
_TC.SIMULATOR.VIEW_POSE_SENSOR = CN()
_TC.SIMULATOR.VIEW_POSE_SENSOR.ADD_TRUNCATED_GAUSSIAN_NOISE = False
_TC.SIMULATOR.VIEW_POSE_SENSOR.TRUNCATED_GAUSSIAN_NOISE = CN()
_TC.SIMULATOR.VIEW_POSE_SENSOR.GAUSSIAN_NOISE_MULTIPLIERS_PATH = "data/noise/pose_noise/seed0.pkl"
_TC.SIMULATOR.VIEW_POSE_SENSOR.TRUNCATED_GAUSSIAN_NOISE.TRANSLATION_MEAN = 0.025
_TC.SIMULATOR.VIEW_POSE_SENSOR.TRUNCATED_GAUSSIAN_NOISE.TRANSLATION_STD = 0.001
_TC.SIMULATOR.VIEW_POSE_SENSOR.TRUNCATED_GAUSSIAN_NOISE.ROTATION_DEGREES_MEAN = 0.9
_TC.SIMULATOR.VIEW_POSE_SENSOR.TRUNCATED_GAUSSIAN_NOISE.ROTATION_DEGREES_STD = 0.057
_TC.SIMULATOR.VIEW_POSE_SENSOR.TRUNCATED_GAUSSIAN_NOISE.NUM_STDS_FOR_TRUNCATION = 2
# -----------------------------------------------------------------------------
# audio config
# -----------------------------------------------------------------------------
_TC.SIMULATOR.AUDIO = CN()
_TC.SIMULATOR.AUDIO.ANECHOIC_AUDIO_SLICE_LENGTH = 3.0
_TC.SIMULATOR.AUDIO.ANECHOIC_DIR = "data/sounds/libri100Classes/heard_16k_rmsNorm400"
_TC.SIMULATOR.AUDIO.VAL_UNHEARD_ANECHOIC_DIR = "data/sounds/libri100Classes/valUnheard_16k_rmsNorm400"
_TC.SIMULATOR.AUDIO.TEST_UNHEARD_ANECHOIC_DIR = "data/sounds/libri100Classes/testUnheard_16k_rmsNorm400"
_TC.SIMULATOR.AUDIO.DIST_ANECHOIC_DIR = "data/sounds/ESC50/esc_16k_rmsNorm400_onefilePerClass" 		# distractor audio (ambient env. sounds) directory
_TC.SIMULATOR.AUDIO.ANECHOIC_AUDIO_TARGET_RMS = 400
_TC.SIMULATOR.AUDIO.RIR_DIR = f"data/ambisonic_rirs/{_TC.SIMULATOR.SCENE_DATASET}"
_TC.SIMULATOR.AUDIO.META_DIR = f"data/metadata/{_TC.SIMULATOR.SCENE_DATASET}"
_TC.SIMULATOR.AUDIO.GRAPH_FILE = 'graph.pkl'
_TC.SIMULATOR.AUDIO.POINTS_FILE = 'points.txt'
_TC.SIMULATOR.AUDIO.NUM_WORKER = 4
_TC.SIMULATOR.AUDIO.BATCH_SIZE = 128
_TC.SIMULATOR.AUDIO.RIR_SAMPLING_RATE = 16000
_TC.SIMULATOR.AUDIO.HOP_LENGTH = 133 
_TC.SIMULATOR.AUDIO.N_FFT = 511
_TC.SIMULATOR.AUDIO.WIN_LENGTH = 400
_TC.SIMULATOR.AUDIO.MAX_VALID_IMPULSE_LENGTH_AFTER_REMOVING_LEADING_ZEROS = 34065
_TC.SIMULATOR.AUDIO.ADD_GAUSSIAN_NOISE = False
_TC.SIMULATOR.AUDIO.NOISE_LEVEL_IN_DB = 40
# -----------------------------------------------------------------------------
# local map config
# -----------------------------------------------------------------------------
_TC.SIMULATOR.LOCAL_MAP = CN()
_TC.SIMULATOR.LOCAL_MAP.SIZE = 31
_TC.SIMULATOR.LOCAL_MAP.SCALE = 0.1
_TC.SIMULATOR.LOCAL_MAP.AGENT_POSITION = [0, 1.25, 0] 
_TC.SIMULATOR.LOCAL_MAP.HEIGHT_THRESH = [0.2, 1.5] 
_TC.SIMULATOR.LOCAL_MAP.WIDTH_DEPTH_IMG = 128
_TC.SIMULATOR.LOCAL_MAP.HEIGHT_DEPTH_IMG = 128
_TC.SIMULATOR.LOCAL_MAP.HFOV_DEPTH_IMG = 90
_TC.SIMULATOR.LOCAL_MAP.MIN_DEPTH = 0.0
_TC.SIMULATOR.LOCAL_MAP.MAX_DEPTH = 67.16327 
_TC.SIMULATOR.LOCAL_MAP.NORMALIZE_DEPTH_IMG = False
# -----------------------------------------------------------------------------
# input ego local occ map config
# -----------------------------------------------------------------------------
_TC.SIMULATOR.EGO_LOCAL_OCC_MAP = CN()
_TC.SIMULATOR.EGO_LOCAL_OCC_MAP.SIZE = 31
_TC.SIMULATOR.EGO_LOCAL_OCC_MAP.NUM_CHANNELS = 2
# -----------------------------------------------------------------------------
# gt ego global canonical occ map target crop config
# -----------------------------------------------------------------------------
_TC.SIMULATOR.GT_GLOBAL_CANONICAL_OCC_MAP_EGO_CROP = CN()
_TC.SIMULATOR.GT_GLOBAL_CANONICAL_OCC_MAP_EGO_CROP.SIZE = 64
_TC.SIMULATOR.GT_GLOBAL_CANONICAL_OCC_MAP_EGO_CROP.NUM_CHANNELS = 2
_TC.SIMULATOR.GT_GLOBAL_CANONICAL_OCC_MAP_EGO_CROP.SCALE = 0.1
# -----------------------------------------------------------------------------
# ego stitched global canonical occ map config
# -----------------------------------------------------------------------------
_TC.SIMULATOR.EGO_STITCHED_GLOBAL_CANONICAL_OCC_MAP = CN()
_TC.SIMULATOR.EGO_STITCHED_GLOBAL_CANONICAL_OCC_MAP.SIZE = 1284
# -----------------------------------------------------------------------------
# local map config
# -----------------------------------------------------------------------------
_TC.SIMULATOR.ALL_AGENTS = CN()
_TC.SIMULATOR.ALL_AGENTS.NUM = 2
# -----------------------------------------------------------------------------
# sim misc. config for transfer from passive to active sampling
# -----------------------------------------------------------------------------
_TC.SIMULATOR.STITCH_TOP_DOWN_MAPS = False
_TC.SIMULATOR.SIM_ENV = CN()
_TC.SIMULATOR.SIM_TASK = CN()
_TC.SIMULATOR.SIM_TRAINER = CN() 
# -----------------------------------------------------------------------------
# Dataset extension
# -----------------------------------------------------------------------------
_TC.DATASET.VERSION = 'v1'
# -----------------------------------------------------------------------------
# Passive Sampling Dataset
# -----------------------------------------------------------------------------
_TC.PASSIVE_SAMPLING_DATASET = CN()
_TC.PASSIVE_SAMPLING_DATASET.TRAIN_DATASET_DIR = None
_TC.PASSIVE_SAMPLING_DATASET.VAL_DATASET_DIR = None
_TC.PASSIVE_SAMPLING_DATASET.TEST_DATASET_DIR = None


def merge_from_path(config, config_paths):
	"""
	merge config with configs from config paths
	:param config: original unmerged config
	:param config_paths: config paths to merge configs from
	:return: merged config
	"""

	if config_paths:
		if isinstance(config_paths, str):
			if CONFIG_FILE_SEPARATOR in config_paths:
				config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
			else:
				config_paths = [config_paths]

		for config_path in config_paths:
			config.merge_from_file(config_path)

	return config


def get_config(
		config_paths: Optional[Union[List[str], str]] = None,
		opts: Optional[list] = None,
		model_dir: Optional[str] = None,
		run_type: Optional[str] = None
) -> CN:
	"""
	Create a unified config with default values overwritten by values from
	`config_paths` and overwritten by options from `opts`.
	:param config_paths: List of config paths or string that contains comma separated list of config paths.
	:param opts: Config options (keys, values) in a list (e.g., passed from command line into the config. For example,
				`opts = ['FOO.BAR',0.5]`. Argument can be used for parameter sweeping or quick tests.
	:param model_dir: suffix for output dirs
	:param run_type: either train or eval
	:return:
	"""

	config = merge_from_path(_C.clone(), config_paths)
	config.TASK_CONFIG = get_task_config(config_paths=config.BASE_TASK_CONFIG_PATH)

	if opts:
		config.CMD_TRAILING_OPTS = opts
		config.merge_from_list(opts)

	assert model_dir is not None, "set --model-dir"
	config.MODEL_DIR = model_dir
	config.TENSORBOARD_DIR = os.path.join(config.MODEL_DIR, config.TENSORBOARD_DIR)
	config.CHECKPOINT_FOLDER = os.path.join(config.MODEL_DIR, 'data')
	config.VIDEO_DIR = os.path.join(config.MODEL_DIR, 'video_dir')
	config.AUDIO_DIR = os.path.join(config.MODEL_DIR, 'audio_dir')
	config.LOG_FILE = os.path.join(config.MODEL_DIR, config.LOG_FILE)
	if config.EVAL_CKPT_PATH == "data/checkpoints":
		config.EVAL_CKPT_PATH = os.path.join(config.MODEL_DIR, 'data')

	dirs = [config.VIDEO_DIR, config.AUDIO_DIR, config.TENSORBOARD_DIR, config.CHECKPOINT_FOLDER]
	if (run_type == 'train') and (not config.RESUME_AFTER_PREEMPTION):
		# check dirs
		if any([os.path.exists(d) for d in dirs]):
			for d in dirs:
				if os.path.exists(d):
					print('{} exists'.format(d))
			key = input('Output directory already exists! Overwrite the folder? (y/n)')
			if key == 'y':
				for d in dirs:
					if os.path.exists(d):
						shutil.rmtree(d)

	config.TASK_CONFIG.defrost()

	# ------------------ modifying SIMULATOR cfg --------------------
	# setting SIMULATOR'S USE_SYNC_VECENV flag
	config.TASK_CONFIG.SIMULATOR.USE_SYNC_VECENV = config.USE_SYNC_VECENV

	# setting max. number of steps of simulator
	config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS = config.TASK_CONFIG.ENVIRONMENT.MAX_CONTEXT_LENGTH - 1
	config.TASK_CONFIG.SIMULATOR.MAX_EPISODE_STEPS = config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS
	config.TASK_CONFIG.SIMULATOR.MAX_CONTEXT_LENGTH = config.TASK_CONFIG.ENVIRONMENT.MAX_CONTEXT_LENGTH
	config.TASK_CONFIG.SIMULATOR.VISUAL_BUDGET = config.TASK_CONFIG.ENVIRONMENT.VISUAL_BUDGET

	# setting simulator attrs from task attrs
	config.TASK_CONFIG.SIMULATOR.EGO_LOCAL_OCC_MAP.SIZE = config.TASK_CONFIG.TASK.CONTEXT_EGO_LOCAL_MAP_SENSOR.SIZE
	config.TASK_CONFIG.SIMULATOR.EGO_LOCAL_OCC_MAP.NUM_CHANNELS = config.TASK_CONFIG.TASK.CONTEXT_EGO_LOCAL_MAP_SENSOR.NUM_CHANNELS

	config.TASK_CONFIG.SIMULATOR.EGO_STITCHED_GLOBAL_CANONICAL_OCC_MAP.SIZE =\
		config.TASK_CONFIG.TASK.CONTEXT_STITCHED_EGO_LOCAL_MAP_SENSOR.SIZE

	config.TASK_CONFIG.SIMULATOR.GT_GLOBAL_CANONICAL_OCC_MAP_EGO_CROP.SIZE =\
		config.TASK_CONFIG.TASK.QUERY_GT_GLOB_CAN_MAP_EGO_CROP_SENSOR.SIZE
	config.TASK_CONFIG.SIMULATOR.GT_GLOBAL_CANONICAL_OCC_MAP_EGO_CROP.NUM_CHANNELS =\
		config.TASK_CONFIG.TASK.QUERY_GT_GLOB_CAN_MAP_EGO_CROP_SENSOR.NUM_CHANNELS
	config.TASK_CONFIG.SIMULATOR.GT_GLOBAL_CANONICAL_OCC_MAP_EGO_CROP.SCALE =\
		config.TASK_CONFIG.TASK.QUERY_GT_GLOB_CAN_MAP_EGO_CROP_SENSOR.SCALE

	config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH = config.TASK_CONFIG.TASK.CONTEXT_RGB_SENSOR.WIDTH
	config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT = config.TASK_CONFIG.TASK.CONTEXT_RGB_SENSOR.HEIGHT

	config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH = config.TASK_CONFIG.TASK.CONTEXT_DEPTH_SENSOR.WIDTH
	config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT = config.TASK_CONFIG.TASK.CONTEXT_DEPTH_SENSOR.HEIGHT
	config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH = config.TASK_CONFIG.TASK.CONTEXT_DEPTH_SENSOR.MIN_DEPTH
	config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH = config.TASK_CONFIG.TASK.CONTEXT_DEPTH_SENSOR.MAX_DEPTH
	config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = config.TASK_CONFIG.TASK.CONTEXT_DEPTH_SENSOR.NORMALIZE_DEPTH
	config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.ADD_REDWOOD_NOISE = config.TASK_CONFIG.TASK.CONTEXT_DEPTH_SENSOR.ADD_REDWOOD_NOISE
	config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.REDWOOD_NOISE_RAND_NUMS_PATH =\
		config.TASK_CONFIG.TASK.CONTEXT_DEPTH_SENSOR.REDWOOD_NOISE_RAND_NUMS_PATH
	config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.REDWOOD_NOISE_MULTIPLIER =\
		config.TASK_CONFIG.TASK.CONTEXT_DEPTH_SENSOR.REDWOOD_NOISE_MULTIPLIER
	config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.REDWOOD_DEPTH_NOISE_DIST_MODEL =\
		config.TASK_CONFIG.TASK.CONTEXT_DEPTH_SENSOR.REDWOOD_DEPTH_NOISE_DIST_MODEL

	config.TASK_CONFIG.SIMULATOR.VIEW_POSE_SENSOR.ADD_TRUNCATED_GAUSSIAN_NOISE =\
		config.TASK_CONFIG.TASK.CONTEXT_VIEW_POSE_SENSOR.ADD_TRUNCATED_GAUSSIAN_NOISE
	config.TASK_CONFIG.SIMULATOR.VIEW_POSE_SENSOR.TRUNCATED_GAUSSIAN_NOISE =\
		config.TASK_CONFIG.TASK.CONTEXT_VIEW_POSE_SENSOR.TRUNCATED_GAUSSIAN_NOISE

	# ------------------------ modifying TASK cfg ----------------------
	config.TASK_CONFIG.TASK.CONTEXT_STITCHED_EGO_LOCAL_MAP_SENSOR.NUM_CHANNELS =\
		config.TASK_CONFIG.TASK.CONTEXT_EGO_LOCAL_MAP_SENSOR.NUM_CHANNELS

	config.TASK_CONFIG.TASK.QUERY_STITCHED_GT_GLOB_CAN_MAP_EGO_CROP_SENSOR.SIZE =\
		config.TASK_CONFIG.TASK.CONTEXT_STITCHED_EGO_LOCAL_MAP_SENSOR.SIZE
	config.TASK_CONFIG.TASK.QUERY_STITCHED_GT_GLOB_CAN_MAP_EGO_CROP_SENSOR.NUM_CHANNELS =\
		config.TASK_CONFIG.TASK.QUERY_GT_GLOB_CAN_MAP_EGO_CROP_SENSOR.NUM_CHANNELS

	config.TASK_CONFIG.TASK.CONTEXT_RGB_SENSOR.FEATURE_SHAPE = [config.TASK_CONFIG.SIMULATOR.ALL_AGENTS.NUM,
																config.TASK_CONFIG.SIMULATOR.MAX_CONTEXT_LENGTH,
																config.TASK_CONFIG.TASK.CONTEXT_RGB_SENSOR.HEIGHT,
																config.TASK_CONFIG.TASK.CONTEXT_RGB_SENSOR.WIDTH,
																3]

	config.TASK_CONFIG.TASK.CONTEXT_EGO_LOCAL_MAP_SENSOR.FEATURE_SHAPE = [config.TASK_CONFIG.SIMULATOR.ALL_AGENTS.NUM,
																		  config.TASK_CONFIG.SIMULATOR.MAX_CONTEXT_LENGTH,
																		  config.TASK_CONFIG.TASK.CONTEXT_EGO_LOCAL_MAP_SENSOR.SIZE,
																		  config.TASK_CONFIG.TASK.CONTEXT_EGO_LOCAL_MAP_SENSOR.SIZE,
																		  config.TASK_CONFIG.TASK.CONTEXT_EGO_LOCAL_MAP_SENSOR.NUM_CHANNELS]

	config.TASK_CONFIG.TASK.CONTEXT_STITCHED_EGO_LOCAL_MAP_SENSOR.FEATURE_SHAPE =\
		[config.TASK_CONFIG.SIMULATOR.MAX_CONTEXT_LENGTH - 1,
		 config.TASK_CONFIG.TASK.CONTEXT_STITCHED_EGO_LOCAL_MAP_SENSOR.SIZE,
		 config.TASK_CONFIG.TASK.CONTEXT_STITCHED_EGO_LOCAL_MAP_SENSOR.SIZE,
		 config.TASK_CONFIG.TASK.CONTEXT_STITCHED_EGO_LOCAL_MAP_SENSOR.NUM_CHANNELS]

	config.TASK_CONFIG.TASK.CONTEXT_VIEW_POSE_SENSOR.FEATURE_SHAPE = [config.TASK_CONFIG.SIMULATOR.ALL_AGENTS.NUM,
																	  config.TASK_CONFIG.SIMULATOR.MAX_CONTEXT_LENGTH,
																	  5]

	config.TASK_CONFIG.TASK.CONTEXT_VIEW_R_N_AZ_SENSOR.FEATURE_SHAPE = [config.TASK_CONFIG.SIMULATOR.ALL_AGENTS.NUM,
																		config.TASK_CONFIG.SIMULATOR.MAX_CONTEXT_LENGTH,
																		config.TASK_CONFIG.TASK.CONTEXT_VIEW_R_N_AZ_SENSOR.FEATURE_SHAPE[-1]]

	config.TASK_CONFIG.TASK.CONTEXT_OTHER_AUDIO_POSE_SENSOR.FEATURE_SHAPE = config.TASK_CONFIG.TASK.CONTEXT_VIEW_POSE_SENSOR.FEATURE_SHAPE

	config.TASK_CONFIG.TASK.AMBI_WAV_SENSOR.FEATURE_SHAPE[0] =\
		config.TASK_CONFIG.SIMULATOR.AUDIO.MAX_VALID_IMPULSE_LENGTH_AFTER_REMOVING_LEADING_ZEROS

	config.TASK_CONFIG.TASK.CONTEXT_SELF_AUDIO_SENSOR.FEATURE_SHAPE[0] = config.TASK_CONFIG.SIMULATOR.ALL_AGENTS.NUM
	config.TASK_CONFIG.TASK.CONTEXT_SELF_AUDIO_SENSOR.FEATURE_SHAPE[1] = config.TASK_CONFIG.ENVIRONMENT.MAX_CONTEXT_LENGTH
	config.TASK_CONFIG.TASK.CONTEXT_SELF_AUDIO_SENSOR.FEATURE_SHAPE[2] = config.TASK_CONFIG.TASK.AMBI_WAV_SENSOR.FEATURE_SHAPE[0]

	config.TASK_CONFIG.TASK.CONTEXT_OTHER_AUDIO_SENSOR.FEATURE_SHAPE[0] = config.TASK_CONFIG.SIMULATOR.ALL_AGENTS.NUM
	config.TASK_CONFIG.TASK.CONTEXT_OTHER_AUDIO_SENSOR.FEATURE_SHAPE[1] =\
		config.TASK_CONFIG.ENVIRONMENT.MAX_CONTEXT_LENGTH * (config.TASK_CONFIG.SIMULATOR.ALL_AGENTS.NUM - 1)
	config.TASK_CONFIG.TASK.CONTEXT_OTHER_AUDIO_SENSOR.FEATURE_SHAPE[2] = config.TASK_CONFIG.TASK.AMBI_WAV_SENSOR.FEATURE_SHAPE[0]

	config.TASK_CONFIG.TASK.QUERY_GT_GLOB_CAN_MAP_EGO_CROP_SENSOR.FEATURE_SHAPE =\
		[config.TASK_CONFIG.SIMULATOR.ALL_AGENTS.NUM,
		 config.TASK_CONFIG.SIMULATOR.MAX_CONTEXT_LENGTH,
		 config.TASK_CONFIG.TASK.QUERY_GT_GLOB_CAN_MAP_EGO_CROP_SENSOR.SIZE,
		 config.TASK_CONFIG.TASK.QUERY_GT_GLOB_CAN_MAP_EGO_CROP_SENSOR.SIZE,
		 config.TASK_CONFIG.TASK.QUERY_GT_GLOB_CAN_MAP_EGO_CROP_SENSOR.NUM_CHANNELS,
		 ]

	config.TASK_CONFIG.TASK.QUERY_GT_GLOB_CAN_MAP_EGO_CROP_EXPLORED_PART_MASK_SENSOR.FEATURE_SHAPE =\
		config.TASK_CONFIG.TASK.QUERY_GT_GLOB_CAN_MAP_EGO_CROP_SENSOR.FEATURE_SHAPE
	config.TASK_CONFIG.TASK.QUERY_GT_GLOB_CAN_MAP_EGO_CROP_EXPLORED_PART_MASK_SENSOR.SIZE =\
		config.TASK_CONFIG.TASK.QUERY_GT_GLOB_CAN_MAP_EGO_CROP_SENSOR.SIZE
	config.TASK_CONFIG.TASK.QUERY_GT_GLOB_CAN_MAP_EGO_CROP_EXPLORED_PART_MASK_SENSOR.NUM_CHANNELS =\
		config.TASK_CONFIG.TASK.QUERY_GT_GLOB_CAN_MAP_EGO_CROP_SENSOR.NUM_CHANNELS
	config.TASK_CONFIG.TASK.QUERY_GT_GLOB_CAN_MAP_EGO_CROP_EXPLORED_PART_MASK_SENSOR.SCALE =\
		config.TASK_CONFIG.TASK.QUERY_GT_GLOB_CAN_MAP_EGO_CROP_SENSOR.SCALE

	config.TASK_CONFIG.TASK.QUERY_STITCHED_GT_GLOB_CAN_MAP_EGO_CROP_SENSOR.FEATURE_SHAPE =\
		config.TASK_CONFIG.TASK.CONTEXT_STITCHED_EGO_LOCAL_MAP_SENSOR.FEATURE_SHAPE
	config.TASK_CONFIG.TASK.QUERY_STITCHED_GT_GLOB_CAN_MAP_EGO_CROP_SENSOR.FEATURE_SHAPE[-1] =\
		config.TASK_CONFIG.TASK.QUERY_STITCHED_GT_GLOB_CAN_MAP_EGO_CROP_SENSOR.NUM_CHANNELS

	config.TASK_CONFIG.TASK.CONTEXT_AUDIO_MASK_SENSOR.FEATURE_SHAPE = [config.TASK_CONFIG.SIMULATOR.ALL_AGENTS.NUM, 
																	  config.TASK_CONFIG.SIMULATOR.MAX_CONTEXT_LENGTH,]
	config.TASK_CONFIG.TASK.ALL_CONTEXT_AUDIO_MASK_SENSOR.FEATURE_SHAPE = config.TASK_CONFIG.TASK.CONTEXT_AUDIO_MASK_SENSOR.FEATURE_SHAPE
	config.TASK_CONFIG.TASK.PREV_CONTEXT_VIEW_MASK_SENSOR.FEATURE_SHAPE = config.TASK_CONFIG.TASK.CONTEXT_AUDIO_MASK_SENSOR.FEATURE_SHAPE

	config.TASK_CONFIG.TASK.QUERY_MASK_SENSOR.FEATURE_SHAPE = [config.TASK_CONFIG.SIMULATOR.ALL_AGENTS.NUM,
															   config.TASK_CONFIG.SIMULATOR.MAX_CONTEXT_LENGTH]
	config.TASK_CONFIG.TASK.ALL_QUERY_MASK_SENSOR.FEATURE_SHAPE = config.TASK_CONFIG.TASK.QUERY_MASK_SENSOR.FEATURE_SHAPE

	# -------------------------------- updating sim misc. config for transfer from passive to active mapping -------------------------------------
	config.TASK_CONFIG.SIMULATOR.STITCH_TOP_DOWN_MAPS = config.STITCH_TOP_DOWN_MAPS_ACTIVE_MAPPING
	config.TASK_CONFIG.SIMULATOR.SIM_ENV = config.TASK_CONFIG.ENVIRONMENT
	config.TASK_CONFIG.SIMULATOR.SIM_TASK = config.TASK_CONFIG.TASK
	config.TASK_CONFIG.SIMULATOR.SIM_TRAINER = config.PassiveMapping

	if config.STITCH_TOP_DOWN_MAPS_ACTIVE_MAPPING:
		config.TASK_CONFIG.TASK.SENSORS += ["EPISODE_SCENE_IDX_SENSOR",
											"EPISODE_REF_RECEIVER_AZIMUTH_SENSOR",
											"QUERY_STITCHED_GT_GLOB_CAN_MAP_EGO_CROP_SENSOR",
											"CONTEXT_VIEW_R_N_AZ_SENSOR"]

	if "CONTEXT_VIEW_R_N_AZ_SENSOR" not in config.TASK_CONFIG.TASK.SENSORS:
		config.TASK_CONFIG.TASK.SENSORS.append("CONTEXT_VIEW_R_N_AZ_SENSOR")

	if (run_type not in ["train"]) and (config.EVAL.SPLIT is not None) and (config.EVAL.SPLIT[:3] == "val"):
		config.PassiveMapping.TrainLosses.types = []
		config.PassiveMapping.EvalMetrics.types = []

	config.TASK_CONFIG.DATASET.EVAL_SPLIT = config.EVAL.SPLIT
	config.TASK_CONFIG.DATASET.EVAL_EPISODE_COUNT = config.EVAL.EPISODE_COUNT

	if run_type == "eval":
		config.TASK_CONFIG.ENVIRONMENT.VISUAL_BUDGET_COMPARE_AGAINST_LAST_TOP_DOWN_MAPS = config.TASK_CONFIG.ENVIRONMENT.VISUAL_BUDGET
		config.TASK_CONFIG.ENVIRONMENT.VISUAL_BUDGET =\
			config.TASK_CONFIG.ENVIRONMENT.MAX_CONTEXT_LENGTH * config.TASK_CONFIG.SIMULATOR.ALL_AGENTS.NUM
		config.TASK_CONFIG.SIMULATOR.VISUAL_BUDGET = config.TASK_CONFIG.ENVIRONMENT.VISUAL_BUDGET

	config.TASK_CONFIG.freeze()

	config.freeze()

	# ---------------------------- assertions for metrics --------------------------------
	if (config.TRAINER_NAME == "chat2map") and (run_type == "train"):
		assert config.PassiveMapping.EvalMetrics.type_for_ckpt_dump in config.PassiveMapping.EvalMetrics.types

	return config


def get_task_config(
		config_paths: Optional[Union[List[str], str]] = None,
		opts: Optional[list] = None
) -> habitat.Config:
	"""
	get config after merging configs stored in yaml files and command line arguments
	:param config_paths: paths to configs
	:param opts: optional command line arguments
	:return: merged config
	"""

	config = _TC.clone()
	if config_paths:
		if isinstance(config_paths, str):
			if CONFIG_FILE_SEPARATOR in config_paths:
				config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
			else:
				config_paths = [config_paths]

		for config_path in config_paths:
			config.merge_from_file(config_path)

	if opts:
		config.merge_from_list(opts)

	config.freeze()
	return config
