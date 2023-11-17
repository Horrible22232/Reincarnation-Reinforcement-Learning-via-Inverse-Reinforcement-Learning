import torch
from neroRRL.environments.expert_rewards.expert_rewards import *

def test_jsd_reward():
    """Test the jsd_reward function
    """
    # Create some dummy data
    probs = [[0.999, 0.001]]
    actions = torch.randn(10, 1)  # Add an extra dimension
    policy_probs = torch.tensor(probs)  # Add an extra dimension
    
    # Case 1: Similar policies
    # Create slightly perturbed expert_policy_probs
    expert_policy_probs_similar = torch.tensor(probs)
    # Create dummy policies
    policy_similar = [type('', (), {'probs': policy_probs})()]
    expert_policy_similar = type('', (), {'probs': expert_policy_probs_similar})

    # Call the function with the dummy data
    reward_similar = jsd_reward(policy_similar, expert_policy_similar, actions)
    
    # Print the output
    print(f"Reward with similar policies: {reward_similar}")
    
    # Case 2: Different policies
    expert_policy_probs_different = torch.tensor([[0.01, 0.99]]) 
    
    # Create dummy policies
    policy_different = [type('', (), {'probs': policy_probs})()]
    expert_policy_different = type('', (), {'probs': expert_policy_probs_different})

    # Call the function with the dummy data
    reward_different = jsd_reward(policy_different, expert_policy_different, actions)
    
    # Print the output
    print(f"Reward with different policies: {reward_different}")
    
def test_prob_reward():
    """Test the prob_reward function"""
    # Create some dummy data
    probs = [[0.8, 0.2], [0.1, 0.9]]
    actions = torch.tensor([0, 1])  # Actions taken by the agent
    policy_probs = torch.tensor(probs) 
    
    # Create expert_policy
    expert_policy = type('', (), {'probs': policy_probs})

    # Case 1: Actions are likely according to expert_policy
    reward_likely = prob_reward(None, expert_policy, actions)
    print(f"Reward when actions are likely according to expert_policy: {reward_likely}")
    
    # Case 2: Actions are unlikely according to expert_policy
    actions_unlikely = torch.tensor([1, 0])  # Actions are reversed, now they are unlikely
    reward_unlikely = prob_reward(None, expert_policy, actions_unlikely)
    print(f"Reward when actions are unlikely according to expert_policy: {reward_unlikely}")
    
def test_best_action_reward():
    """Test the best_action_reward function"""
    # Create some dummy data
    probs = [[0.8, 0.2], [0.1, 0.9]]
    actions = torch.tensor([0, 1])  # Actions taken by the agent
    policy_probs = torch.tensor(probs) 
    
    # Create expert_policy
    expert_policy = type('', (), {'probs': policy_probs})

    # Case 1: Actions are the best according to expert_policy
    reward_best = best_action_reward(None, expert_policy, actions)
    print(f"Reward when actions are the best according to expert_policy: {reward_best}")
    
    # Case 2: Actions are not the best according to expert_policy
    actions_not_best = torch.tensor([1, 0])  # Actions are reversed, now they are not the best
    reward_not_best = best_action_reward(None, expert_policy, actions_not_best)
    print(f"Reward when actions are not the best according to expert_policy: {reward_not_best}")

print("Testing prob_reward")
test_prob_reward()
print("Testing best_action_reward")
test_best_action_reward()
print("Testing jsd_reward")
test_jsd_reward()
