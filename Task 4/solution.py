import torch
import torch.optim as optim
from torch.distributions import Normal
import torch.nn as nn
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import warnings
from typing import Union
from utils import ReplayBuffer, get_env, run_episode
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int,
                 hidden_layers: int, activation: str):
        super(NeuralNetwork, self).__init__()
        layers = [nn.Linear(input_dim, hidden_size), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers.append(nn.Linear(hidden_size, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.model(s)


class Actor(NeuralNetwork):
    def __init__(self, hidden_size: int, hidden_layers: int, actor_lr: float,
                 state_dim: int = 3, action_dim: int = 1, device: torch.device = torch.device('cpu')):
        super(Actor, self).__init__(state_dim, 2 * action_dim, hidden_size, hidden_layers, 'relu')
        self.optimizer = optim.Adam(self.parameters(), lr=actor_lr)
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.device = device

    def clamp_log_std(self, log_std: torch.Tensor) -> torch.Tensor:
        return torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

    def get_action_and_log_prob(self, state: torch.Tensor, deterministic: bool) -> (torch.Tensor, torch.Tensor):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        mu, log_std = self.forward(state).chunk(2, dim=-1)
        log_std = self.clamp_log_std(log_std)
        std = log_std.exp()
        normal = Normal(mu, std)
        z = normal.rsample() if not deterministic else mu
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)

        # Check if log_prob has a second dimension. If it does, sum over it.
        if log_prob.ndim > 1:
            log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob


class Critic(NeuralNetwork):
    def __init__(self, hidden_size: int, hidden_layers: int, critic_lr: float,
                 state_dim: int = 3, action_dim: int = 1, device: torch.device = torch.device('cpu')):
        super(Critic, self).__init__(state_dim + action_dim, 1, hidden_size, hidden_layers, 'relu')
        self.optimizer = optim.Adam(self.parameters(), lr=critic_lr)
        self.device = device

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # Assuming the state and action are concatenated as the input to the network.
        cat_input = torch.cat([state, action], dim=-1)
        return super().forward(cat_input)


class Agent:
    def __init__(self):
        self.state_dim = 3
        self.action_dim = 1
        self.batch_size = 200
        self.min_buffer_size = 1000
        self.max_buffer_size = 100000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: {}".format(self.device))
        self.memory = ReplayBuffer(self.min_buffer_size, self.max_buffer_size, self.device)
        # Agent specific parameters
        self.hidden_size = 256
        self.hidden_layers = 3
        self.actor_lr = 3e-4
        self.critic_lr = 3e-4
        self.tau = 0.005
        self.gamma = 0.99
        self.alpha = 0.2

        # Initialize actor and critic networks
        self.actor = Actor(self.hidden_size, self.hidden_layers, self.actor_lr, self.state_dim, self.action_dim,
                           self.device)
        self.critic1 = Critic(self.hidden_size, self.hidden_layers, self.critic_lr, self.state_dim, self.action_dim,
                              self.device)
        self.critic2 = Critic(self.hidden_size, self.hidden_layers, self.critic_lr, self.state_dim, self.action_dim,
                              self.device)
        self.critic1_target = Critic(self.hidden_size, self.hidden_layers, self.critic_lr, self.state_dim,
                                     self.action_dim, self.device)
        self.critic2_target = Critic(self.hidden_size, self.hidden_layers, self.critic_lr, self.state_dim,
                                     self.action_dim, self.device)

        # Copy weights to the target networks
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

    def get_action(self, s: np.ndarray, train: bool) -> np.ndarray:
        with torch.no_grad():
            state = torch.tensor(s, dtype=torch.float32).to(self.device)
            action, _ = self.actor.get_action_and_log_prob(state, deterministic=not train)
        return action.cpu().numpy()

    @staticmethod
    def run_gradient_update_step(object: Union[Actor, Critic], loss: torch.Tensor):
        object.optimizer.zero_grad()
        loss.mean().backward()
        object.optimizer.step()

    def critic_target_update(self, base_net: NeuralNetwork, target_net: NeuralNetwork, tau: float, soft_update: bool):
        for param_target, param in zip(target_net.parameters(), base_net.parameters()):
            if soft_update:
                param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
            else:
                param_target.data.copy_(param.data)

    def train_agent(self):
        if self.memory.size() < self.min_buffer_size:
            return

        # Sample a batch from the memory
        state_batch, action_batch, reward_batch, next_state_batch = self.memory.sample(self.batch_size)

        with torch.no_grad():
            next_action_batch, next_log_prob_batch = self.actor.get_action_and_log_prob(next_state_batch,
                                                                                        deterministic=False)
            q1_next_target = self.critic1_target(next_state_batch, next_action_batch)
            q2_next_target = self.critic2_target(next_state_batch, next_action_batch)
            min_q_target = torch.min(q1_next_target, q2_next_target) - self.alpha * next_log_prob_batch
            q_target = reward_batch + self.gamma * min_q_target  # Assume non-terminal transitions

        # Update critics
        q1 = self.critic1(state_batch, action_batch)
        q2 = self.critic2(state_batch, action_batch)
        critic1_loss = F.mse_loss(q1, q_target)
        critic2_loss = F.mse_loss(q2, q_target)
        self.run_gradient_update_step(self.critic1, critic1_loss)
        self.run_gradient_update_step(self.critic2, critic2_loss)

        # Update actor
        new_action_batch, log_prob_batch = self.actor.get_action_and_log_prob(state_batch, deterministic=False)
        q1_new = self.critic1(state_batch, new_action_batch)
        q2_new = self.critic2(state_batch, new_action_batch)
        min_q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_prob_batch - min_q_new).mean()
        self.run_gradient_update_step(self.actor, actor_loss)

        # Soft update of target networks
        self.critic_target_update(self.critic1, self.critic1_target, self.tau, True)
        self.critic_target_update(self.critic2, self.critic2_target, self.tau, True)


# This main function is provided here to enable some basic testing.
# ANY changes here WON'T take any effect while grading.
if __name__ == '__main__':

    TRAIN_EPISODES = 50
    TEST_EPISODES = 300

    # You may set the save_video param to output the video of one of the evalution episodes, or
    # you can disable console printing during training and testing by setting verbose to False.
    save_video = True
    verbose = True

    agent = Agent()
    env = get_env(g=10.0, train=True)

    for EP in range(TRAIN_EPISODES):
        run_episode(env, agent, None, verbose, train=True)

    if verbose:
        print('\n')

    test_returns = []
    env = get_env(g=10.0, train=False)

    if save_video:
        video_rec = VideoRecorder(env, "pendulum_episode.mp4")

    for EP in range(TEST_EPISODES):
        rec = video_rec if (save_video and EP == TEST_EPISODES - 1) else None
        with torch.no_grad():
            episode_return = run_episode(env, agent, rec, verbose, train=False)
        test_returns.append(episode_return)

    avg_test_return = np.mean(np.array(test_returns))

    print("\n AVG_TEST_RETURN:{:.1f} \n".format(avg_test_return))

    if save_video:
        video_rec.close()