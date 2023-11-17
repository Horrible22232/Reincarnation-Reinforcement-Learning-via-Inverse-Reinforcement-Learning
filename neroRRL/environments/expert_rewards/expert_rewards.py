import numpy as np
import torch
from scipy.spatial import distance

def jsd_reward(policy, expert_policy, actions):
    """Calculates the expert reward based on the policy distribution using the Jensen-Shannon Divergence

    Arguments:
        policy {list} -- The policy of the agent
        expert_policy {torch.distributions.Categorical} -- The policy of the expert
        actions {torch.Tensor} -- The actions taken by the agent (not used)
        
    Returns:
        {np.ndarray} -- The expert reward for each action
    """
    agent_probs = policy[0].probs.cpu().numpy()
    expert_probs = expert_policy.probs.cpu().numpy()

    # Compute JSD for each pair of distributions
    jsd = np.array([distance.jensenshannon(agent_prob, expert_prob, 2.0) 
                    for agent_prob, expert_prob in zip(agent_probs, expert_probs)])

    # Clamp JSD values and compute similarity scores
    jsd = np.maximum(jsd, 1e-10)
    expert_reward = 1 - jsd

    return expert_reward

def prob_reward(policy, expert_policy, actions):
    """Calculates the expert reward based on the probability of the agent's action under the expert's policy.

    Arguments:
        policy {list} -- The policy of the agent
        expert_policy {torch.distributions.Categorical} -- The policy of the expert
        actions {torch.Tensor} -- The actions taken by the agent
        
    Returns:
        {np.ndarray} -- The expert reward for each action
    """
    # Get the probability of the agent's action under the expert's policy
    prob = expert_policy.probs[range(len(actions)), actions]
    
    # Convert the probability to a reward
    expert_reward = prob.detach().cpu().numpy()

    return expert_reward

def euclidean_similarity_reward(policy, expert_policy, actions, exponent=3):
    """Calculates the expert reward based on the Euclidean similarity of the agent's action probabilities under the agent's and expert's policies.

    Arguments:
        policy {list} -- The policy of the agent
        expert_policy {torch.distributions.Categorical} -- The policy of the expert
        actions {torch.Tensor} -- The actions taken by the agent
        
    Returns:
        {np.ndarray} -- The expert reward for each action
    """
    # Get the probability of the agent's action under the agent's policy
    agent_prob = policy[0].probs[range(len(actions)), actions]

    # Get the probability of the agent's action under the expert's policy
    expert_prob = expert_policy.probs[range(len(actions)), actions]
    
    # Calculate the Euclidean similarity between the probabilities
    similarity = 1 / (1 + ((agent_prob - expert_prob) ** 2).sqrt())
    
    # Convert the similarity to a reward
    expert_reward = similarity ** exponent * expert_prob
    
    # Convert the reward to a numpy array
    expert_reward = expert_reward.detach().cpu().numpy() 

    return expert_reward

def chebyshev_similarity_reward(policy, expert_policy, actions, exponent=3):
    """Calculates the expert reward based on the Chebyshev similarity of the agent's action probabilities under the agent's and expert's policies.

    Arguments:
        policy {list} -- The policy of the agent
        expert_policy {torch.distributions.Categorical} -- The policy of the expert
        actions {torch.Tensor} -- The actions taken by the agent
        
    Returns:
        {np.ndarray} -- The expert reward for each action
    """
    # Get the probability of the agent's action under the agent's policy
    agent_prob = policy[0].probs[range(len(actions)), actions]

    # Get the probability of the agent's action under the expert's policy
    expert_prob = expert_policy.probs[range(len(actions)), actions]
    
    # Calculate the Chebyshev similarity between the probabilities
    similarity = 1 / (1 + (agent_prob - expert_prob).abs())
    
    # Convert the similarity to a reward
    expert_reward = similarity ** exponent * expert_prob
    
    # Convert the reward to a numpy array
    expert_reward = expert_reward.detach().cpu().numpy() 

    return expert_reward

def best_action_reward(policy, expert_policy, actions):
    """Calculates the expert reward only when the best action under the expert's policy is taken by the agent.

    Arguments:
        policy {list} -- The policy of the agent
        expert_policy {torch.distributions.Categorical} -- The policy of the expert
        actions {torch.Tensor} -- The actions taken by the agent
        
    Returns:
        {np.ndarray} -- The expert reward for each action
    """
    # Get the probability of the agent's action under the expert's policy
    prob = expert_policy.probs[range(len(actions)), actions]
    
    # Identify the action with the highest probability under the expert's policy
    best_action = torch.argmax(expert_policy.probs, dim=1)
    
    # Create a mask where the agent's actions match the best actions
    mask = (actions == best_action).float()

    # Use the mask to assign rewards only to the best actions
    expert_reward = (prob * mask).detach().cpu().numpy()
    
    return expert_reward

def best_action_with_penalty_reward(policy, expert_policy, actions):
    """Calculates the expert reward only when the best action under the expert's policy is taken by the agent.
    Punishes the agent with the best action's probability when the agent does not take the best action.

    Arguments:
        policy {list} -- The policy of the agent
        expert_policy {torch.distributions.Categorical} -- The policy of the expert
        actions {torch.Tensor} -- The actions taken by the agent
        
    Returns:
        {np.ndarray} -- The expert reward for each action
    """
    # Get the probability of the agent's action under the expert's policy
    prob = expert_policy.probs[range(len(actions)), actions]

    # Identify the action with the highest probability under the expert's policy
    best_action = torch.argmax(expert_policy.probs, dim=1)

    # Create a mask where the agent's actions match the best actions
    mask = (actions == best_action).float()

    # Calculate the best action's probability
    best_action_prob = expert_policy.probs[range(len(actions)), best_action]

    # If the mask is True (best action was taken), prob is reward, else the punishment is the best action's prob
    expert_reward = (prob * mask + (1 - mask) * (-best_action_prob)).detach().cpu().numpy()

    return expert_reward

class HistogramReward:
    def __init__(self):
        """
        The constructor for the HistogramReward class.
        """
        self._num_actions = 17
        self.expert_action_histogram = None 
        self.agent_action_histogram = None
        
    def histogram_reward(self, policy, expert_policy, actions, dones, sample):
        """Calculates the expert reward based on the probability of the agent's action under the expert's policy.

        Arguments:
            policy {list} -- The policy of the agent
            expert_policy {torch.distributions.Categorical} -- The policy of the expert
            actions {torch.Tensor} -- The actions taken by the agent
            dones {np.ndarray} -- The done flag for each episode
            sample {bool} -- Whether to update the histogram or not (used for calculating the statistics)
            
        Returns:
            {np.ndarray} -- The expert reward for each action
        """
        # Initialize the histogram if it is None
        if self.expert_action_histogram is None:
            # The histogram is a 2D array where the rows represent the number of worker and the columns represent the actions
            self.expert_action_histogram = np.array([[0 for _ in range(self._num_actions)] for _ in range(len(actions))])
            self.agent_action_histogram = np.array([[0 for _ in range(self._num_actions)] for _ in range(len(actions))])
            
        # Move the agent's actions to the CPU and convert them to a numpy array
        actions = actions.cpu().numpy()
        # Sample the optimal expert's actions and convert them to a numpy array
        expert_actions = expert_policy.sample().cpu().numpy()
        # Update the histogram
        if sample: # if sample is True, update the histogram (Check if not used for statistics)
            self.expert_action_histogram[range(len(actions)), expert_actions] += 1
            self.agent_action_histogram[range(len(actions)), actions] += 1
        # Create a mask where the agent's actions have to be in the budget
        mask = self.expert_action_histogram[range(len(actions)), actions] >= self.agent_action_histogram[range(len(actions)), actions]
        # Calculate the expert reward
        denominator = 2 * self.expert_action_histogram[range(len(actions)), actions] - self.agent_action_histogram[range(len(actions)), actions]
        safe_denominator = np.where(denominator != 0, denominator, 1)
        expert_reward = self.expert_action_histogram[range(len(actions)), actions] / safe_denominator
        exponent = self.expert_action_histogram[range(len(actions)), actions] - self.agent_action_histogram[range(len(actions)), actions]
        exponent = exponent.clip(min=0, max=10)
        expert_reward = expert_reward ** exponent
        expert_reward = expert_reward * mask
        # Reset the histogram if the episode is done
        for i in range(len(actions)):
            if dones is not None and dones[i]:
                self.expert_action_histogram[i] = np.array([0 for _ in range(self._num_actions)])
                self.agent_action_histogram[i] = np.array([0 for _ in range(self._num_actions)])

        # Return the expert reward
        return expert_reward

    def __call__(self, policy, expert_policy, actions, dones, sample):
        """
        This method is called when the HistogramReward object is called. It calls the histogram_reward 
        method.

        Arguments:
            policy {list} -- The policy of the agent
            expert_policy {torch.distributions.OneHotCategorical} -- The policy of the expert
            actions {torch.Tensor} -- The actions taken by the agent
            dones {np.ndarray} -- The done flag for each episode
            sample {bool} -- Whether to update the histogram or not (used for calculating the statistics)
            
        Returns:
            {np.ndarray} -- The expert reward
        """
        return self.histogram_reward(policy, expert_policy, actions, dones, sample)


class OnActionChangeReward:
    """
    This class represents a reward calculator that calculates the reward based on the change in best action.
    The reward is only granted when the new best action differs from the previous best action.
    """
    def __init__(self):
        """
        The constructor for the OnActionChangeReward class. It initializes the previous best action to None.
        """
        self.prev_best_action = None

    def on_action_change_reward(self, policy, expert_policy, actions):
        """
        This method calculates the expert reward only when the best action under the expert's policy is taken by the 
        agent, and this best action is different from the previous best action.

        Arguments:
            policy {list} -- The policy of the agent
            expert_policy {torch.distributions.OneHotCategorical} -- The policy of the expert
            actions {torch.Tensor} -- The actions taken by the agent
            
        Returns:
            {np.ndarray} -- The expert reward for each action. The reward is only given for actions that are both 
            the best action under the expert's policy and different from the previous best action.
        """
        # Get the probability of the agent's action under the expert's policy
        prob = expert_policy.probs[range(len(actions)), actions]
        
        # Identify the action with the highest probability under the expert's policy
        best_action = torch.argmax(expert_policy.probs, dim=1)

        # Create a mask where the agent's actions match the best actions and the best action is not the previous best action
        mask = (actions == best_action).float()
        
        if self.prev_best_action is not None:
            mask = mask * (best_action != self.prev_best_action).float()

        # Use the mask to assign rewards only to the best actions
        reward_on_action_change = (prob * mask).detach().cpu().numpy()
        
        # Update the previous best action
        self.prev_best_action = best_action
        
        return reward_on_action_change
    
    def __call__(self, policy, expert_policy, actions):
        """
        This method is called when the OnActionChangeReward object is called. It calls the on_action_change_reward 
        method.

        Arguments:
            policy {list} -- The policy of the agent
            expert_policy {torch.distributions.OneHotCategorical} -- The policy of the expert
            actions {torch.Tensor} -- The actions taken by the agent
            
        Returns:
            {np.ndarray} -- The expert reward for each action. The reward is only given for actions that are both 
            the best action under the expert's policy and different from the previous best action.
        """
        return self.on_action_change_reward(policy, expert_policy, actions)
    
on_action_change_reward = OnActionChangeReward()
historgram_reward = HistogramReward()
