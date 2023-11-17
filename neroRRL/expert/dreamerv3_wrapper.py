from neroRRL.expert.modules.dreamerv3.embodied.envs import from_gym
from neroRRL.expert.modules.dreamerv3.embodied.core import Checkpoint
from neroRRL.expert.modules import dreamerv3
from neroRRL.expert.modules.dreamerv3 import embodied
import numpy as np
import crafter
from pathlib import Path
import torch
import jax
import torch.nn.functional as F
import tensorflow as tf
from torch.distributions import Categorical

class DreamerV3Wrapper:
    """Converts the DreamerV3 model to a PyTorch model."""
    
    def __init__(self, config_path, model_path, observation_space, action_space, device):    
        """Loads the DreamerV3 model and config from the given paths.
        
        Arguments:
            config_path {str} -- The config path to the model
            model_path {str} -- The path to the model
            observation_space {box} -- The observation space
            action_space {tuple} -- The action space
            device {str} -- The device to run the model on
        """
        # Convert to Path objects
        config_path = Path(config_path)
        model_path = Path(model_path)
        # Load the config
        config = embodied.Config.load(config_path)
        # Set the device
        if device.type == "cpu":
            config = config.update({"jax.platform" : "cpu"})
        elif device.type == "cuda":
            config = config.update({"jax.platform" : "gpu"})
        # Create the agent
        step = embodied.Counter()
        agent = dreamerv3.Agent(observation_space, action_space, step, config)
        # Create the checkpoint to load the model
        checkpoint = Checkpoint()
        checkpoint.agent = agent
        checkpoint.load(model_path, keys=['agent'])
        # Set the final loaded agent
        self.agent = agent
    
    
    def forward(self, obs, state):
        """The forward pass of the dreamerv3 model.

        Arguments:
            obs {dict} -- The observation
            state {tuple} -- The state

        Returns:
            {OneHotCategorical} -- The action distribution
            {dict} -- The state
            {torch.tensor} -- The logits of the action distribution
        """
        # Get the task outputs from the agent
        _, state, task_outs = self.agent.policy(obs, state, mode='eval')
        # Get the logits from the task outputs and move them to the cpu
        action_logits = jax.device_get(task_outs['action'].logits)
        action_logits = action_logits.tolist()
        # Convert the logits to a tensor
        logits = torch.tensor(action_logits)
        # Create an equivalent PyTorch Categorical distribution
        policy = Categorical(logits=logits)
        # Return the policy and state
        return policy, state, logits
        
    def __call__(self, obs, state):
        """Calls the forward pass of the model."""
        forward_pass = self.forward(obs, state)
        # assert False
        return forward_pass


def test():
    """Tests the DreamerV3Wrapper class.
    """
    # Set the used necessary path variables for the model and environment
    base_path = "./model/expert/crafter/"
    config_path = Path(base_path + 'config.yaml')
    model_path = Path(base_path + 'checkpoint.ckpt')
    # Load the config
    config = embodied.Config.load(config_path)
    # Create the environment and wrap with the dreamerv3 wrapper
    env = crafter.Env() 
    env = from_gym.FromGym(env)
    env = dreamerv3.wrap_env(env, config)
    # Create the agent
    agent = DreamerV3Wrapper(config_path, model_path, env.obs_space, env.act_space, torch.device("cpu"))
    # Create the initial state and action
    state = None
    act = {'action': env.act_space['action'].sample(), 'reset': np.array(True)}
    done, rewards, iter = False, [], 0
    while not done:
        obs = env.step(act)

        obs = {k: [v] if isinstance(v, (list, dict)) else np.array([v]) for k, v in obs.items()}
        policy, state = agent(obs, state)
        
        act = {'action': policy.sample().cpu().numpy()[0], 'reset': obs['is_last'][0]}
        
        # Log result
        # clear_output()
        # time.sleep(0.5)
        rewards.append(obs["reward"][0])
        done = obs["is_terminal"][0]
        print("\riter:", iter, "reward:", np.sum(rewards), "done:", done, end='', flush=True)
        iter += + 1
    # Get the logits from the task outputs
    # action_logits = task_outs['action'].logits.tolist()

    # Move the distribution to CPU
    #logits = torch.tensor(action_logits) 

    #print(logits)

    # Create an equivalent PyTorch Categorical distribution
    #distribution = OneHotCategorical(logits=logits)
    # Sample from the distribution
    # sample = policy.sample()

    # print("sample:", sample)
    # print("log_prob:", policy.log_prob(sample))
    # print("prob:", torch.exp(policy.log_prob(sample)))
    
# test()