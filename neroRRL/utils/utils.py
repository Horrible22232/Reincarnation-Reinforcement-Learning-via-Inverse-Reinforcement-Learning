import numpy as np
import os
import torch
import random

from gymnasium import spaces
import matplotlib.pyplot as plt

from neroRRL.environments.wrapper import wrap_environment
from neroRRL.utils.monitor import Tag

def set_library_seeds(seed:int) -> None:
    """Applies the submitted seed to PyTorch, Numpy and Python

    Arguments:
        int {seed} -- The to be applied seed
    """
    random.seed(seed)
    random.SystemRandom().seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def compute_gradient_stats(modules_dict, prefix = ""):
    """Computes the gradient norm and the gradient mean for each parameter of the model and the entire model itself.

    Arguments:
        model_parameters {dict} -- Main modules of the models
        tag {string} -- To distinguish entire models from each other, a tag can be supplied

    Returns:
        {dict}: Returns all results as a dictionary
    """
    results = {}
    all_grads = []

    for module_name, module in modules_dict.items():
        if module is not None:
            grads = []
            for param in module.parameters():
                grads.append(param.grad.view(-1))
            results[module_name + "_norm"] = (Tag.GRADIENT_NORM, module.grad_norm())
            # results[module_name + "_mean"] = (Tag.GRADIENT_MEAN, module.grad_mean())
            all_grads = all_grads + grads
    results[prefix + "_model_norm"] = (Tag.GRADIENT_NORM, torch.linalg.norm(torch.cat(all_grads)).item())
    # results[prefix + "_model_mean"] = (Tag.GRADIENT_MEAN, torch.mean(torch.cat(all_grads)).item())
    return results

def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)

def get_environment_specs(env_config, worker_id, realtime_mode = False):
    """Creates a dummy environments, resets it, and hence obtains all environment specifications .

    Arguments:
        env_config {dict} -- Configuration of the environment
        worker_id {int} -- Worker id that is necessary for socket-based environments like Unity

    Returns:
        {tuple} -- Returns visual observation space, vector observations space, action space and max episode steps
    """
    dummy_env = wrap_environment(env_config, worker_id, realtime_mode)
    vis_obs, vec_obs = dummy_env.reset(env_config["reset_params"])
    max_episode_steps = dummy_env.max_episode_steps
    visual_observation_space = dummy_env.visual_observation_space
    vector_observation_space = dummy_env.vector_observation_space
    if isinstance(dummy_env.action_space, spaces.Discrete):
        action_space_shape = (dummy_env.action_space.n,)
    else:
        action_space_shape = tuple(dummy_env.action_space.nvec)
    dummy_env.close()
    return visual_observation_space, vector_observation_space, action_space_shape, max_episode_steps

def aggregate_episode_results(episode_infos):
    """Takes in a list of episode info dictionaries. All episode results (episode reward, length, success, ...) are
    aggregate using min, max, mean, std.

    Arguments:
        episode_infos {list} -- List of dictionaries containing episode infos such as episode reward, length, ...

    Returns:
        {dict} -- Result dictionary featuring all aggregated metrics
    """
    results = {}
    if len(episode_infos) > 0:
        keys = episode_infos[0].keys()
        # Compute mean, std, min and max for each information, skip seed
        for key in keys:
            if key == "seed":
                continue
            results[key + "_mean"] = np.mean([info[key] for info in episode_infos])
            results[key + "_min"] = np.min([info[key] for info in episode_infos])
            results[key + "_max"] = np.max([info[key] for info in episode_infos])
            results[key + "_std"] = np.std([info[key] for info in episode_infos])
    return results

import numpy as np
import matplotlib.pyplot as plt

def plot_action_histogram(values, filename, normalize=True, y_label="Normalized Frequency", title="Action Distribution"):
    """
    Plots a normalized histogram from a list of values with custom labels. 
    The bar with the highest value will be colored red.
    Saves the plot to a specified PDF file.

    Parameters:
        values {list}: A list of numerical values to be normalized and plotted.
        filename {str}: Name of the PDF file to save the plot to.
    """
    
    # Check if the lengths of values and labels are equal
    labels = ["Noop", "Move Left", "Move Right", "Move Up", "Move Down", "Do", "Sleep", "Place Stone", "Place Table", "Place Furnace", "Place Plant", "Make Wood Pickaxe", "Make Stone Pickaxe", "Make Iron Pickaxe", "Make Wood Sword", "Make Stone Sword", "Make Iron Sword"]
    if len(values) != len(labels):
        raise ValueError("Length of values and labels must be the same")

    if normalize:
        # Normalize the values between 0 and 1
        normalized_values = np.array(values) / sum(values)
    else:
        normalized_values = values

    # Default color for bars
    colors = ['blue' for _ in range(len(values))]

    # Set the color of the largest bar to red
    max_index = np.argmax(normalized_values)
    colors[max_index] = 'red'

    # Plotting
    plt.figure()
    plt.bar(labels, normalized_values, color=colors)
    plt.ylabel(y_label)
    plt.title(title)
    plt.ylim(0, 1)  # Set y-axis limits to be between 0 and 1
    plt.xticks(rotation=90)  # Make the labels vertical
    
    folder_name = "action_histograms"
    # Create the specified folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Combine the folder name and the filename to get the complete path
    full_path = os.path.join(folder_name, filename)

    # Save the plot to a PDF file
    plt.savefig(full_path, format='pdf', bbox_inches='tight')
    plt.close()

def create_expert_policy(config, visual_observation_space = None, vector_observation_space = None, action_space = None):
    """Creates an expert policy based on the environment configuration.

    Arguments:
        config {dict} -- Environment expert configuration
        visual_observation_space {gym.spaces} -- Visual observation space of the environment (Currently not used)
        vector_observation_space {gym.spaces} -- Vector observation space of the environment (Currently not used)
        action_space {gym.spaces} -- Action space of the environment (Currently not used)
        
    Returns:
        {torch.model} -- Expert policy
    """

    # Check if expert configuration is available and if not, return None to indicate that no expert is available
    if "expert" not in config:
        return None
    # Create the expert policy based on the configuration
    if config["expert"]["env_type"] == "Crafter" and config["expert"]["model"] == "DreamerV3":
        # Importing here to make sure that the dependencies are only loaded if necessary
        from neroRRL.expert.dreamerv3_wrapper import DreamerV3Wrapper
        from neroRRL.expert.modules.dreamerv3.embodied.envs import from_gym
        from neroRRL.expert.modules.dreamerv3.embodied.core import Checkpoint
        from neroRRL.expert.modules import dreamerv3
        from neroRRL.expert.modules.dreamerv3 import embodied
        import crafter
        # Create a dummy environment to obtain the observation and action space
        env = crafter.Env() 
        env = from_gym.FromGym(env)
        env = dreamerv3.wrap_env(env, embodied.Config.load(config["expert"]["config_path"]))
        # Return the expert policy
        return DreamerV3Wrapper(config["expert"]["config_path"], config["expert"]["model_path"],  env.obs_space, env.act_space, torch.device(config["expert"]["device"]))