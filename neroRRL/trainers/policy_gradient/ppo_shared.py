import torch
import numpy as np
from torch import optim

from neroRRL.nn.actor_critic import create_actor_critic_model
from neroRRL.trainers.policy_gradient.base import BaseTrainer
from neroRRL.utils.utils import compute_gradient_stats, batched_index_select
from neroRRL.utils.decay_schedules import polynomial_decay
import torch.nn.functional as F
from neroRRL.utils.monitor import Tag

class PPOTrainer(BaseTrainer):
    """PPO implementation according to Schulman et al. 2017. It supports multi-discrete action spaces as well as visual 
    and vector obsverations (either alone or simultaenously). Parameters can be shared or not. If gradients shall be decoupled,
    go for the DecoupledPPOTrainer.
    """
    def __init__(self, configs, sample_device, train_device, worker_id, run_id, out_path, seed = 0, compile_model = False):
        """
        Initializes distinct members of the PPOTrainer

        Arguments:
            configs {dict} -- The whole set of configurations (e.g. training and environment configs)
            sample_device {torch.device} -- The device used for sampling training data.
            train_device {torch.device} -- The device used for model optimization.
            worker_id {int} -- Specifies the offset for the port to communicate with the environment, which is needed for Unity ML-Agents environments (default: {1})
            run_id {string} -- The run_id is used to tag the training runs (directory names to store summaries and checkpoints) (default: {"default"})
            out_path {str} -- Determines the target directory for saving summaries, logs and model checkpoints. (default: "./")
            compile_model {bool} -- If true, the model is compiled before training (default: {False})
        """
        super().__init__(configs, sample_device, train_device, worker_id, run_id=run_id, out_path=out_path, seed=seed, compile_model=compile_model)

        # Hyperparameter setup
        self.epochs = configs["trainer"]["epochs"]
        self.vf_loss_coef = self.configs["trainer"]["value_coefficient"]
        self.n_mini_batches = configs["trainer"]["n_mini_batches"]
        batch_size = self.n_workers * self.worker_steps
        assert (batch_size % self.n_mini_batches == 0), "Batch Size divided by number of mini batches has a remainder."
        self.max_grad_norm = configs["trainer"]["max_grad_norm"]

        self.lr_schedule = configs["trainer"]["learning_rate_schedule"]
        self.beta_schedule = configs["trainer"]["beta_schedule"]
        self.cr_schedule = configs["trainer"]["clip_range_schedule"]
        self.kl_divergence_schedule = configs["trainer"]["kl_coefficient_schedule"] if "kl_coefficient_schedule" in configs["trainer"] else None

        self.learning_rate = self.lr_schedule["initial"]
        self.beta = self.beta_schedule["initial"]
        self.clip_range = self.cr_schedule["initial"]
        self.kl_divergence_coef = self.kl_divergence_schedule["initial"] if self.kl_divergence_schedule is not None else None

        # Instantiate optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)

    def create_model(self) -> None:
        return create_actor_critic_model(self.configs["model"], self.configs["trainer"]["share_parameters"],
        self.vis_obs_space, self.vec_obs_space, self.action_space_shape, self.sample_device)

    def train(self):
        train_info = {}

        # Train policy and value function for e epochs using mini batches
        for epoch in range(self.epochs):
            # Refreshes buffer with current model for every refresh_buffer_epoch
            if epoch > 0 and epoch % self.refresh_buffer_epoch == 0 and self.refresh_buffer_epoch > 0:
                self.sampler.buffer.refresh(self.model, self.gamma, self.lamda)

            # Normalize advantages batch-wise if desired
            # This is done during every epoch just in case refreshing the buffer is used
            if self.configs["trainer"]["advantage_normalization"] == "batch":
                advantages = self.sampler.buffer.samples_flat["advantages"]
                self.sampler.buffer.samples_flat["normalized_advantages"] = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Retrieve the to be trained mini_batches via a generator
            # Use the recurrent mini batch generator for training a recurrent policy
            if self.recurrence is not None:
                mini_batch_generator = self.sampler.buffer.recurrent_mini_batch_generator(self.n_mini_batches)
            else:
                mini_batch_generator = self.sampler.buffer.mini_batch_generator(self.n_mini_batches)
            # Conduct the training
            for mini_batch in mini_batch_generator:
                res = self.train_mini_batch(mini_batch)
                # Collect all values of the training procedure in a list
                for key, (tag, value) in res.items():
                    train_info.setdefault(key, (tag, []))[1].append(value)

        # Calculate mean of the collected training statistics
        for key, (tag, values) in train_info.items():
            train_info[key] = (tag, np.mean(values))

        # Format specific values for logging inside the base class
        formatted_string = "loss={:.3f} pi_loss={:.3f} vf_loss={:.3f} entropy={:.3f}".format(
            train_info["loss"][1], train_info["policy_loss"][1], train_info["value_loss"][1], train_info["entropy"][1])
        if self.kl_divergence_coef is not None:
            formatted_string += " kl_loss={:.3f}".format(train_info["kl_loss"][1])

        # Return the mean of the training statistics
        return train_info, formatted_string

    def train_mini_batch(self, samples):
        """Optimizes the policy based on the PPO algorithm

        Arguments:
            samples {dict} -- The sampled mini-batch to optimize the model
        
        Returns:
            training_stats {dict} -- Losses, entropy, kl-divergence and clip fraction
        """
        # Retrieve the agent's memory to feed the model
        memory, mask, memory_indices = None, None, None
        # Case Recurrence: the recurrent cell state is treated as the memory. Only the initial hidden states are selected.
        if self.recurrence is not None:
            if self.recurrence["layer_type"] == "gru":
                memory = samples["hxs"]
            elif self.recurrence["layer_type"] == "lstm":
                memory = (samples["hxs"], samples["cxs"])
        # Case Transformer: the episodic memory is based on activations that were previously gathered throughout an episode
        if self.transformer is not None:
            # Slice memory windows from the whole sequence of episode memories
            memory = batched_index_select(samples["memories"], 1, samples["memory_indices"])
            mask = samples["memory_mask"]
            memory_indices = samples["memory_indices"]
        expert_reward_coefficient = samples["expert_reward_coefficient"] if "expert_reward_coefficient" in samples else None

        # Forward model -> policy, value, memory, gae
        policy, value, value_expert, _, _ = self.model(samples["vis_obs"] if self.vis_obs_space is not None else None,
                                    samples["vec_obs"] if self.vec_obs_space is not None else None,
                                    memory = memory, mask = mask, memory_indices = memory_indices,
                                    sequence_length = self.sampler.buffer.actual_sequence_length,
                                    expert_reward_coefficient = expert_reward_coefficient)
        # Policy Loss
        # Retrieve and process log_probs from each policy branch
        log_probs, entropies = [], []
        for i, policy_branch in enumerate(policy):
            log_probs.append(policy_branch.log_prob(samples["actions"][:, i]))
            entropies.append(policy_branch.entropy())
        log_probs = torch.stack(log_probs, dim=1)
        entropies = torch.stack(entropies, dim=1).sum(1).reshape(-1)
        
        # Remove paddings if recurrence is used
        if self.recurrence is not None:
            value = value[samples["loss_mask"]]
            value_expert = value_expert[samples["loss_mask"]] if "expert_values" in samples else None
            log_probs = log_probs[samples["loss_mask"]]
            entropies = entropies[samples["loss_mask"]]
            actions = samples["actions"][samples["loss_mask"]]
        
        # Compute surrogates
        # Determine advantage normalization
        if self.configs["trainer"]["advantage_normalization"] == "minibatch":
            normalized_advantage = (samples["advantages"] - samples["advantages"].mean()) / (samples["advantages"].std() + 1e-8)
        elif self.configs["trainer"]["advantage_normalization"] == "no":
            normalized_advantage = samples["advantages"]
        else:
            normalized_advantage = samples["normalized_advantages"]
        # Repeat is necessary for multi-discrete action spaces
        normalized_advantage = normalized_advantage.unsqueeze(1).repeat(1, len(self.action_space_shape))
        log_ratio = log_probs - samples["log_probs"]
        ratio = torch.exp(log_ratio)
        surr1 = ratio * normalized_advantage
        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * normalized_advantage
        policy_loss = torch.min(surr1, surr2)
        policy_loss = policy_loss.mean()

        # Value loss
        # Note: Only if expert rewards and multiple value heads are used,
        # the advantages get split into two parts: the environment and the expert advantages
        advantages = samples["env_advantages"] if "env_advantages" in samples else samples["advantages"]
        sampled_return = samples["values"] + advantages
        clipped_value = samples["values"] + (value - samples["values"]).clamp(min=-self.clip_range, max=self.clip_range)
        vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
        vf_loss = vf_loss.mean()

        # Entropy Bonus
        entropy_bonus = entropies.mean()
        

        # Complete loss
        loss = -(policy_loss - self.vf_loss_coef * vf_loss + self.beta * entropy_bonus)
        
        # Add expert loss if multiple value heads are used
        if "expert_values" in samples:
            sampled_return_expert = samples["expert_values"] + samples["advantages_expert"]
            clipped_value_expert = samples["expert_values"] + (value_expert - samples["expert_values"]).clamp(min=-self.clip_range, max=self.clip_range)
            vf_loss_expert = torch.max((value_expert - sampled_return_expert) ** 2, (clipped_value_expert - sampled_return_expert) ** 2)
            vf_loss_expert = vf_loss_expert.mean()
            loss += self.vf_loss_coef * vf_loss_expert

        # Add KL Divergence Bonus if expert logits are available
        if self.kl_divergence_coef is not None and "expert_logits" in samples:
            # Inspired by:
            # https://github.com/unixpickle/obs-tower2/blob/474eadf32f6321f9df96f11f5d75fd06d6c94786/obs_tower2/prierarchy.py#L66
            agent_log_probs = F.log_softmax(policy[0].logits[samples["loss_mask"]], dim=-1)
            expert_log_probs = F.log_softmax(samples["expert_logits"], dim=-1)

            expert_kl_divergence = torch.mean(torch.sum(torch.exp(agent_log_probs) * (agent_log_probs - expert_log_probs), dim=-1))
            loss += self.kl_divergence_coef * expert_kl_divergence
            
        # Compute gradients
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()

        # Monitor additional training statistics
        approx_kl = ((ratio - 1.0) - log_ratio).mean()  # http://joschu.net/blog/kl-approx.html
        clip_fraction = (abs((ratio - 1.0)) > self.clip_range).float().mean()

        # Retrieve modules for monitoring the gradient norm
        if self.model.share_parameters:
            modules = self.model.actor_critic_modules
        else:
            modules = {**self.model.actor_modules, **self.model.critic_modules}

        return {**compute_gradient_stats(modules),
                "policy_loss": (Tag.LOSS, policy_loss.cpu().data.numpy()),
                "value_loss": (Tag.LOSS, vf_loss.cpu().data.numpy()),
                "value_loss_expert": (Tag.LOSS, vf_loss_expert.cpu().data.numpy()) if "expert_values" in samples else (Tag.LOSS, 0.0),
                "loss": (Tag.LOSS, loss.cpu().data.numpy()),
                "entropy": (Tag.OTHER, entropy_bonus.cpu().data.numpy()),
                "kl_divergence": (Tag.OTHER, approx_kl.cpu().data.numpy()),
                "clip_fraction": (Tag.OTHER, clip_fraction.cpu().data.numpy()),
                "kl_loss": (Tag.LOSS, expert_kl_divergence.cpu().data.numpy()) if self.kl_divergence_coef is not None else (Tag.LOSS, 0.0)}

    def step_decay_schedules(self, update):
        self.learning_rate = polynomial_decay(self.lr_schedule["initial"], self.lr_schedule["final"],
                                        self.lr_schedule["max_decay_steps"], self.lr_schedule["power"], update)
        self.beta = polynomial_decay(self.beta_schedule["initial"], self.beta_schedule["final"],
                                        self.beta_schedule["max_decay_steps"], self.beta_schedule["power"], update)
        self.clip_range = polynomial_decay(self.cr_schedule["initial"], self.cr_schedule["final"],
                                        self.cr_schedule["max_decay_steps"], self.cr_schedule["power"], update)
        if self.kl_divergence_schedule is not None:
            self.kl_divergence_coef = polynomial_decay(self.kl_divergence_schedule["initial"], self.kl_divergence_schedule["final"],
                                        self.kl_divergence_schedule["max_decay_steps"], self.kl_divergence_schedule["power"], update)

        # Apply learning rate to optimizer
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.learning_rate

        return {
            "learning_rate": (Tag.DECAY, self.learning_rate),
            "beta": (Tag.DECAY, self.beta),
            "clip_range": (Tag.DECAY, self.clip_range)
        }


    def collect_checkpoint_data(self, update):
        checkpoint_data = super().collect_checkpoint_data(update)
        checkpoint_data["model"] = self.model.state_dict()
        checkpoint_data["optimizer"] = self.optimizer.state_dict()
        return checkpoint_data

    def apply_checkpoint_data(self, checkpoint):
        super().apply_checkpoint_data(checkpoint)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])