from neroRRL.environments.wrappers.frame_skip import FrameSkipEnv
from neroRRL.environments.wrappers.stacked_observation import StackedObservationEnv
from neroRRL.environments.wrappers.scaled_visual_observation import ScaledVisualObsEnv
from neroRRL.environments.wrappers.grayscale_visual_observation import GrayscaleVisualObsEnv
from neroRRL.environments.wrappers.spotlights import SpotlightsEnv
from neroRRL.environments.wrappers.positional_encoding import PositionalEncodingEnv
from neroRRL.environments.wrappers.pytorch_shape import PyTorchEnv
from neroRRL.environments.wrappers.last_action_to_obs import LastActionToObs
from neroRRL.environments.wrappers.last_reward_to_obs import LastRewardToObs
from neroRRL.environments.wrappers.reward_normalization import RewardNormalizer
from neroRRL.environments.wrappers.last_expert_distr_to_obs import LastExpertDistibutionToObs
from neroRRL.environments.wrappers.last_expert_reward_to_obs import LastExpertRewardToObs

def wrap_environment(config, worker_id, realtime_mode = False, record_trajectory = False):
    """This function instantiates an environment and applies wrappers based on the specified config.

    Arguments:
        config {dict} -- The to be applied wrapping configuration
        worker_id {int} -- The worker id that sets off the port for communication with Unity environments
        realtime_mode {bool} -- Whether to render and run the environment in realtime
        record_trajectory {bool} -- Whether to record the trajectory of an entire episode. This can be used for video recording. (default: {False})

    Returns:
        {Env} -- The wrapped environment
    """
    # Instantiate environment
    if config["type"] == "MemoryGym":
        from neroRRL.environments.memory_gym_wrapper import MemoryGymWrapper
        env = MemoryGymWrapper(config["name"], config["reset_params"], realtime_mode=realtime_mode, record_trajectory=record_trajectory)
    elif config["type"] == "Unity":
        from neroRRL.environments.unity_wrapper import UnityWrapper
        env = UnityWrapper(config["name"], config["reset_params"], worker_id, realtime_mode=realtime_mode, record_trajectory=record_trajectory)
    elif config["type"] == "ObstacleTower":
        from neroRRL.environments.obstacle_tower_wrapper import ObstacleTowerWrapper
        env = ObstacleTowerWrapper(config["name"], config["reset_params"], worker_id, realtime_mode=realtime_mode, record_trajectory=record_trajectory)
    elif config["type"] == "Minigrid":
        from neroRRL.environments.minigrid_wrapper import MinigridWrapper
        env = MinigridWrapper(config["name"], config["reset_params"], realtime_mode=realtime_mode, record_trajectory=record_trajectory)
    elif config["type"] == "MinigridVec":
        from neroRRL.environments.minigrid_vec_wrapper import MinigridVecWrapper
        env = MinigridVecWrapper(config["name"], config["reset_params"], realtime_mode=realtime_mode, record_trajectory=record_trajectory)
    elif config["type"] == "Procgen":
        from neroRRL.environments.procgen_wrapper import ProcgenWrapper
        env = ProcgenWrapper(config["name"], config["reset_params"], realtime_mode=realtime_mode, record_trajectory=record_trajectory)
    elif config["type"] == "CartPole":
        from neroRRL.environments.cartpole_wrapper import CartPoleWrapper
        env = CartPoleWrapper(config["name"], config["reset_params"], realtime_mode=realtime_mode, record_trajectory=record_trajectory)
    elif config["type"] == "Ballet":
        from neroRRL.environments.ballet_wrapper import BalletWrapper
        env = BalletWrapper(config["reset_params"], realtime_mode=realtime_mode, record_trajectory=record_trajectory)
    elif config["type"] == "RandomMaze":
        from neroRRL.environments.maze_wrapper import MazeWrapper
        env = MazeWrapper(config["reset_params"], realtime_mode=realtime_mode, record_trajectory=record_trajectory)
    elif config["type"] == "Crafter":
        from neroRRL.environments.crafter_wrapper import CrafterWrapper
        env = CrafterWrapper(config["reset_params"], realtime_mode=realtime_mode, record_trajectory=record_trajectory)

    # Wrap environment
    # Frame Skip
    if config["frame_skip"] > 1:
        env = FrameSkipEnv(env, config["frame_skip"])
    # Last action to obs
    if config["last_action_to_obs"]:
        env = LastActionToObs(env)
    # Last reward to obs
    if config["last_reward_to_obs"]:
        env = LastRewardToObs(env)
    # Grayscale
    if config["grayscale"] and env.visual_observation_space is not None:
        env = GrayscaleVisualObsEnv(env)
    # Spotlight perturbation
    if "spotlight_perturbation" in config:
        env = SpotlightsEnv(env, config["spotlight_perturbation"])
    # Rescale Visual Observation
    if env.visual_observation_space is not None:
        env = ScaledVisualObsEnv(env, config["resize_vis_obs"][0], config["resize_vis_obs"][1])
    # Positional Encoding
    if config["positional_encoding"]:
        env = PositionalEncodingEnv(env)
    # Stack Observation
    if config["obs_stacks"] > 1:
        env = StackedObservationEnv(env, config["obs_stacks"])
    if config["reward_normalization"] > 1:
        env = RewardNormalizer(env, config["reward_normalization"])
    if "current_expert_distr_to_obs" in config and config["current_expert_distr_to_obs"]:
        env = LastExpertDistibutionToObs(env)
    if "last_expert_reward_to_obs" in config and config["last_expert_reward_to_obs"]:
        env = LastExpertRewardToObs(env)
        
    return PyTorchEnv(env)