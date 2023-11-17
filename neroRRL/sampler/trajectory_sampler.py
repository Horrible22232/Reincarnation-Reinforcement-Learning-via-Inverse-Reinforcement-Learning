import numpy as np
import torch

from neroRRL.sampler.buffer import Buffer
from neroRRL.utils.worker import Worker, BufferedEnv
from neroRRL.utils.utils import create_expert_policy, plot_action_histogram
from neroRRL.utils.decay_schedules import polynomial_decay
from neroRRL.environments.expert_rewards.expert_rewards import *

class TrajectorySampler():
    """The TrajectorySampler employs n environment workers to sample data for s worker steps regardless if an episode ended.
    Hence, the collected trajectories may contain multiple episodes or incomplete ones."""
    def __init__(self, run_id, configs, worker_id, visual_observation_space, vector_observation_space, action_space_shape, model, sample_device, train_device) -> None:
        """Initializes the TrajectorSampler and launches its environment workers.

        Arguments:
            run_id {str} -- Short tag that describes the underlying training run
            configs {dict} -- The whole set of configurations (e.g. training and environment configs)
            worker_id {int} -- Specifies the offset for the port to communicate with the environment, which is needed for Unity ML-Agents environments.
            visual_observation_space {box} -- Dimensions of the visual observation space (None if not available)
            vector_observation_space {tuple} -- Dimensions of the vector observation space (None if not available)
            action_space_shape {tuple} -- Dimensions of the action space
            model {nn.Module} -- The model to retrieve the policy and value from
            sample_device {torch.device} -- The device that is used for retrieving the data from the model
            train_device {torch.device} -- The device that is used for training the model
        """
        # Set member variables
        self.run_id = run_id
        self.visual_observation_space = visual_observation_space
        self.vector_observation_space = vector_observation_space
        self.action_space_shape = action_space_shape
        self.model = model
        self.n_workers = configs["sampler"]["n_workers"]
        self.worker_steps = configs["sampler"]["worker_steps"]
        self.current_expert_distr_to_obs = configs["environment"]["current_expert_distr_to_obs"] if "current_expert_distr_to_obs" in configs["environment"] else False
        self.last_expert_reward_to_obs = configs["environment"]["last_expert_reward_to_obs"] if "last_expert_reward_to_obs" in configs["environment"] else False
        self.sample_device = sample_device
        self.train_device = train_device
        self.update = 0

        # Initialize expert
        self._init_expert(configs, visual_observation_space, vector_observation_space, action_space_shape)
        # Create Buffer
        self.buffer = Buffer(configs, visual_observation_space, vector_observation_space,
                        action_space_shape, self.train_device, self.model.share_parameters, self)
        
        # Launch workers
        self.workers = [Worker(configs["environment"], worker_id + 200 + w) for w in range(self.n_workers)]
        # Launch worker list
        # self.worker_list = WorkerList(worker_id, configs["environment"], self.n_workers, self.expert)
        # Setup timestep placeholder
        self.worker_current_episode_step = torch.zeros((self.n_workers, ), dtype=torch.long)
        
        # Setup initial observations
        if visual_observation_space is not None:
            self.vis_obs = np.zeros((self.n_workers,) + visual_observation_space.shape, dtype=np.float32)
        else:
            self.vis_obs = None
        if vector_observation_space is not None:
            self.vec_obs = np.zeros((self.n_workers,) + vector_observation_space, dtype=np.float32)
        else:
            self.vec_obs = None

        self.bonus = np.zeros((self.n_workers, ), dtype=np.float32)
        # Reset workers
        for worker in self.workers:
            worker.child.send(("reset", None))
        # Grab initial observations
        for i, worker in enumerate(self.workers):
            vis_obs, vec_obs = worker.child.recv()
            if self.vis_obs is not None:
                self.vis_obs[i] = vis_obs
            if self.vec_obs is not None:
                self.vec_obs[i] = vec_obs

    def _init_expert(self, configs, visual_observation_space, vector_observation_space, action_space_shape):
        self.expert = create_expert_policy(configs, visual_observation_space, vector_observation_space, action_space_shape)
        self.use_expert = True if "expert" in configs else False
        self.expert_model_name = configs["expert"]["model"] if "expert" in configs else ""
        self.expert_reward_type = configs["expert"]["reward_type"] if "expert" in configs else ""
        self.expert_episode_rewards = [[] for _ in range(self.n_workers)]
        self.expert_optimal_rewards = [[] for _ in range(self.n_workers)]
        self.expert_reward_schedule = configs["expert"]["reward_schedule"] if "expert" in configs else None
        self.expert_state = None
        self.expert_historical_actions = [0 for _ in range(action_space_shape[0])] if "expert" in configs else None
        self.expert_reward_per_action = [[] for _ in range(action_space_shape[0])] if "expert" in configs else None
        self.agent_historical_actions = [0 for _ in range(action_space_shape[0])] if "expert" in configs else None
        self.test_expert = configs["expert"]["test"] if "expert" in configs else False

    def sample(self) -> list:
        """Samples training data (i.e. experience tuples) using n workers for t worker steps.

        Returns:
            {list} -- List of completed episodes. Each episode outputs a dictionary containing at least the
            achieved reward and the episode length.
        """
        # Initialize episode information list
        episode_infos = []
        # Initialize expert reward coefficient for scaling the expert reward
        self._init_expert_reward_coefficient()
        
        # Sample actions from the model and collect experiences for training
        for t in range(self.worker_steps):
            # Gradients can be omitted for sampling data
            with torch.no_grad():
                # Expert policy
                # Forward the expert model to retrieve the policy (making decisions)
                expert_policy = None
                if self.use_expert:
                    expert_policy = self._forward_expert(self.vis_obs, self.vec_obs, t)
                    # Add the expert policy probs to the vector observation for the agent model if selected
                    if self.current_expert_distr_to_obs:
                        if self.vec_obs is not None:
                            if self.last_expert_reward_to_obs:
                                self.vec_obs[:, -expert_policy.probs.shape[1] - 1: -1] = expert_policy.probs.cpu().numpy()
                            else:
                                self.vec_obs[:, -expert_policy.probs.shape[1]:] = expert_policy.probs.cpu().numpy()
                    if self.last_expert_reward_to_obs and t > 0:
                        self.vec_obs[:, -1] = self.buffer.expert_rewards[:, t - 1]

                # Agent policy
                # Write observations and recurrent cell to buffer (Agent policy only)
                self.previous_model_input_to_buffer(t)
                # Convert observations to tensors
                vis_obs_batch = torch.tensor(self.vis_obs) if self.vis_obs is not None else None
                vec_obs_batch = torch.tensor(self.vec_obs) if self.vec_obs is not None else None
                # Forward the agent model to retrieve the policy (making decisions)
                policy, value, expert_value = self.forward_model(vis_obs_batch, vec_obs_batch, t)
                    
                # Sample actions from each individual policy branch
                actions = []
                log_probs = []
                for action_branch in policy:
                    action = action_branch.sample()
                    actions.append(action)
                    log_probs.append(action_branch.log_prob(action))
                    
                # Overwrite actions with expert actions for debugging
                if self.use_expert and self.test_expert:
                    actions = [expert_policy.sample()]
                    
                # Write actions, log_probs, and values to buffer
                self.buffer.actions[:, t] = torch.stack(actions, dim=1)
                self.buffer.log_probs[:, t] = torch.stack(log_probs, dim=1)
                self.buffer.values[:, t] = value.data
                if expert_value is not None:
                    self.buffer.expert_values[:, t] = expert_value.data

            # Execute actions
            actions = self.buffer.actions[:, t].cpu().numpy() # send actions as batch to the CPU, to save IO time
            for w, worker in enumerate(self.workers):
                worker.child.send(("step", actions[w]))

            # Retrieve results
            for w, worker in enumerate(self.workers):
                vis_obs, vec_obs, self.buffer.env_rewards[w, t], self.buffer.dones[w, t], info = worker.child.recv()
                if self.vis_obs is not None:
                    self.vis_obs[w] = vis_obs
                if self.vec_obs is not None:
                    self.vec_obs[w] = vec_obs
                if info:
                    # Reset the bonus
                    self.bonus[w] = 0.0
                    # Store the information of the completed episode (e.g. total reward, episode length)
                    info = self._compute_expert_episode_info(w, info)
                    episode_infos.append(info)
                    self.reset_worker(worker, w, t)
                else:
                    # Increment worker timestep
                    self.worker_current_episode_step[w] +=1
            
            # Process expert rewards
            self._process_expert_rewards(policy, expert_policy, t)

        return episode_infos

    def _compute_expert_episode_info(self, w, info):
        """Compute and update the reward information, then reset the expert rewards.
        
        Arguments:
            w {int} -- Index or ID for the worker.
            info {dict} -- Dictionary to store and update reward information.
            
        Returns:
            {dict} -- Updated dictionary with reward information.
        """
        if self.use_expert:
            info["expert_reward"] = sum(self.expert_episode_rewards[w])
            info["total_reward"] = info["reward"] + info["expert_reward"]
            info["expert_optimal_reward"] = sum(self.expert_optimal_rewards[w])
            info["rel_expert_optimal_reward"] = info["expert_reward"] / (info["expert_optimal_reward"] + 1e-8)
            self.expert_episode_rewards[w],  self.expert_optimal_rewards[w], self.expert_actions = [], [], []
        
        return info


    def _process_expert_rewards(self, policy, expert_policy, t):
        """Process expert rewards by generating, storing, and calculating optimal rewards.
            
            Arguments:
                policy {torch.Categorical} -- The current policy.
                expert_policy {torch.Categorical} -- The expert policy.
                t {int} -- Current timestep or index.
        """
        if self.use_expert:
            # Update the expert reward coefficient
            self._update_expert_reward_coefficient(t)
            # Create the expert reward
            expert_reward = self._generate_expert_reward(policy, expert_policy, self.buffer.actions[:, t], self.buffer.dones[:, t], sample = True)
            # Add the expert reward to the environment reward
            self.buffer.expert_rewards[:, t] = expert_reward
            # Collect the expert statistics
            self._collect_expert_statistics(expert_policy, expert_reward, t)   

    def _collect_expert_statistics(self, expert_policy, expert_reward, t):
        """
        Collect expert statistics based on given policy and timestep t.
        
        Parameters:
        expert_policy {torch.Categorical} -- The expert policy.
        expert_reward {np.ndarray} -- The expert reward.
        t {int} -- Current timestep or index.
        """
        
        # Calculate the optimal expert reward
        expert_optimal_actions = expert_policy.sample().unsqueeze(dim=1)
        expert_optimal_reward = self._generate_expert_reward([expert_policy], expert_policy, expert_optimal_actions, self.buffer.dones[:, t])
        
        # Collect the expert rewards in a list for the statistics
        for w in range(self.n_workers):
            self.expert_episode_rewards[w].append(expert_reward[w])
            self.expert_optimal_rewards[w].append(expert_optimal_reward[w])
            self.expert_historical_actions[expert_optimal_actions[w]] += 1
            self.agent_historical_actions[self.buffer.actions[:, t][w].squeeze()] += 1
            
        # Collect the expert reward per action in a list for the statistics
        for a in range(self.action_space_shape[0]):
            actions = torch.tensor([a for _ in range(self.n_workers)]).unsqueeze(dim=1)
            expert_reward_per_current_action = self._generate_expert_reward([expert_policy], expert_policy, actions, self.buffer.dones[:, t]).mean()
            self.expert_reward_per_action[a].append(expert_reward_per_current_action)

    def previous_model_input_to_buffer(self, t):
        """Add the model's previous input to the buffer.

        Arguments:
            t {int} -- Current step of sampling
        """
        if self.vis_obs is not None:
            self.buffer.vis_obs[:, t] = torch.tensor(self.vis_obs)
        if self.vec_obs is not None:
            self.buffer.vec_obs[:, t] = torch.tensor(self.vec_obs)

    def forward_model(self, vis_obs, vec_obs, t):
        """Forwards the model to retrieve the policy and the value of the to be fed observations.

        Arguments:
            vis_obs {torch.tensor} -- Visual observations batched across workers
            vec_obs {torch.tensor} -- Vector observations batched across workers
            t {int} -- Current step of sampling

        Returns:
            {tuple} -- policy {list of categorical distributions}, values of environment and {torch.tensor}
        """
        policy, value, expert_value, _, _ = self.model(vis_obs, vec_obs, expert_reward_coefficient = self.expert_reward_coefficient)
        return policy, value, expert_value
    
    def _forward_expert(self, vis_obs, vec_obs, t):
        """ Forwards the expert model to retrieve the policy of the to be fed observations.

        Arguments:
            vis_obs {torch.tensor} -- Visual observations batched across workers
            vec_obs {torch.tensor} -- Vector observations batched across workers
            t {int} -- Current step of sampling

        Returns:
            {torch.distribution} -- policy
        """
        expert_policy = None
        if self.expert_model_name == "DreamerV3":
            # Get expert visual observation
            image = []
            for w, worker in enumerate(self.workers):
                worker.child.send(("image", None))
                image.append(worker.child.recv())
            image = np.array(image).astype(np.uint8)
            # Create the expert model's input
            obs = {"image": image, "reward": self.buffer.env_rewards[:, t], "is_last": self.buffer.dones[:, t], "is_terminal": self.buffer.dones[:, t]}
            obs["is_first"] = np.zeros((self.n_workers, ), dtype=np.bool)
            for w in range(self.n_workers):
                if t == 0 or self.buffer.dones[w, t-1]:
                    obs["is_first"][w] = True
            # Forward the expert model to retrieve the Categorical-Policy and hidden states
            expert_policy, self.expert_state, self.buffer.expert_logits[:, t] = self.expert(obs, self.expert_state)
            
        return expert_policy
    
    def _generate_expert_reward(self, policy, expert_policy, actions, dones, sample = False):
        """Generates the expert reward based on the expert model's policy and the agent's policy.

        Arguments:
            policy {list of categorical distributions}: The agent's policy
            expert_policy {torch.distribution}: The expert model's policy
            actions {torch.tensor}: The agent's actions
            dones {np.ndarray} -- The done flag for each episode
            sample {bool} -- Specifies if the expert reward is used for the sampling (this affects the expert rewards internal state)
        Returns:
            {np.ndarray}: The expert reward
        """
        expert_reward = 0
        # Calculate the expert reward based on the expert model's policy and the agent's policy
        if self.expert_model_name == "DreamerV3":
            actions = actions.squeeze(dim=1)
            # Generate an expert Jensen-Shannon-Divergence (JSD) reward if selected
            if self.expert_reward_type == "jsd":
                expert_reward = jsd_reward(policy, expert_policy, actions)
            # Generate an expert probability reward if selected
            if self.expert_reward_type == "probs":
                expert_reward = prob_reward(policy, expert_policy, actions)
            # Generate an expert best action reward if selected
            if self.expert_reward_type == "best_action":
                expert_reward = best_action_reward(policy, expert_policy, actions)
            if self.expert_reward_type == "best_action_with_penalty":
                expert_reward = best_action_with_penalty_reward(policy, expert_policy, actions)
            if self.expert_reward_type == "on_action_change_reward":
                expert_reward = on_action_change_reward(policy, expert_policy, actions)
            if self.expert_reward_type == "euclidean":
                expert_reward = euclidean_similarity_reward(policy, expert_policy, actions)
            if self.expert_reward_type == "chebyshev":
                expert_reward = chebyshev_similarity_reward(policy, expert_policy, actions)
            if self.expert_reward_type == "histogramm":
                expert_reward = historgram_reward(policy, expert_policy, actions, dones, sample)
            if self.expert_reward_type == "none":
                expert_reward = np.zeros((self.n_workers, ), dtype=np.float32)
                
        # Scale the expert reward
        expert_reward *= self.expert_reward_coefficient
        
        # Return the expert reward
        return expert_reward
    
    def _init_expert_reward_coefficient(self):
        """Initializes the expert reward coefficient."""
        # If no expert reward is used, return
        if not self.use_expert:
            self.expert_reward_coefficient = None
            return
        # Else, calculate the expert reward coefficient
        self.update += 1
        self.expert_reward_coefficient = polynomial_decay(self.expert_reward_schedule["initial"], self.expert_reward_schedule["final"], self.expert_reward_schedule["max_decay_steps"], self.expert_reward_schedule["power"], self.update) * np.ones((self.n_workers, ))
        self.init_expert_reward_coefficient = self.expert_reward_coefficient
        self.buffer.expert_reward_coefficient[:, 0] = self.expert_reward_coefficient 
        
    def _update_expert_reward_coefficient(self, t):
        """Updates the expert reward coefficient.

        Arguments:
            t {int} -- Current step of sampling
        """
        # Calculate the bonus if used
        self.bonus = self.bonus + self.buffer.env_rewards[:, t] / 15.0 if self.expert_reward_schedule["bonus"] else self.bonus
        # Update the expert reward coefficient
        self.expert_reward_coefficient = self.init_expert_reward_coefficient + np.maximum(self.bonus, 0) * self.init_expert_reward_coefficient
        # Store expert reward coefficient
        self.buffer.expert_reward_coefficient[:, t] = self.expert_reward_coefficient

    def reset_worker(self, worker, id, t):
        """Resets the specified worker.

        Arguments:
            worker {remote} -- The to be reset worker
            id {int} -- The ID of the to be reset worker
            t {int} -- Current step of sampling data
        """
        # Reset the worker's current timestep
        self.worker_current_episode_step[id] = 0
        # Reset agent (potential interface for providing reset parameters)
        worker.child.send(("reset", None))
        # Get data from reset
        vis_obs, vec_obs = worker.child.recv()
        # Get new worker
        #new_worker, (vis_obs, vec_obs) = self.worker_list.reset(worker, id)
        # Replace worker
        #self.workers[id] = new_worker
        
        if self.vis_obs is not None:
            self.vis_obs[id] = vis_obs
        if self.vec_obs is not None:
            self.vec_obs[id] = vec_obs

    def get_last_value(self):
        """Returns the last value of the current observation to compute GAE.

        Returns:
            {tuple} -- Last value of the environment and the expert model 
        """
        _, last_value, last_expert_value, _, _ = self.model(torch.tensor(self.vis_obs) if self.vis_obs is not None else None,
                                        torch.tensor(self.vec_obs) if self.vec_obs is not None else None,
                                        None, expert_reward_coefficient = self.expert_reward_coefficient)
        return last_value, last_expert_value

    def close(self) -> None:
        """Closes the sampler and shuts down its environment workers."""
        if self.use_expert:
            plot_action_histogram(self.agent_historical_actions, self.run_id + "_agent_actions_distribution.pdf")
            plot_action_histogram(self.expert_historical_actions, self.run_id + "_expert_actions_distribution.pdf")
            self.expert_reward_per_action = [np.mean(expert_reward) for expert_reward in self.expert_reward_per_action]
            plot_action_histogram(self.expert_reward_per_action, self.run_id + "_expert_reward_per_action.pdf", normalize=False, y_label="Average Expert Reward", title="Average Expert Reward per Action")
        try:
            for worker in self.workers:
                worker.close()
        except:
            pass

    def to(self, device):
        """Args:
            device {torch.device} -- Desired device for all current tensors"""
        for k in dir(self):
            att = getattr(self, k)
            if torch.is_tensor(att):
                setattr(self, k, att.to(device))