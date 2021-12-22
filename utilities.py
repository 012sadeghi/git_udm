import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from scipy import stats

# Define a costumized CDF
class MyRandomVariableClass_grid(stats.rv_continuous):
    def __init__(self, xtol=1e-14, seed=None):
        super().__init__(a=0, xtol=xtol, seed=seed)

    def _cdf(self, x):
        return 1-np.exp(-x**2)

class ComptRandomVariableClass_cmpt(stats.rv_continuous):
    def __init__(self, xtol=1e-14, seed=None):
        super().__init__(a=0, xtol=xtol, seed=seed)

    def _cdf(self, x):
        return 1-np.exp(-x**2)

class ComptRandomVariableClass_renewable(stats.rv_continuous):
    def __init__(self, xtol=1e-14, seed=None):
        super().__init__(a=0, xtol=xtol, seed=seed)

    def _cdf(self, x):
        return 1-np.exp(-x**10)


class FF_v_estimator(nn.Module):
    def __init__(self, input_shpae, output_shape, no_layers, activation, gamma):
        super(FF_v_estimator, self).__init__()
        self.gamma = gamma
        self.fc1 = nn.Linear(input_shpae, 128)
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, 1)
        self.relu = torch.nn.ReLU()
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=1)
        self.loss = nn.MSELoss()

    def forward(self, x):
        x = torch.from_numpy(x)
        output = self.fc1(x)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        # output = self.relu(output)

        # output = output.detach().numpy()
        return output


class FF_policy(object):
    def __init__(self, stat_dim, action_dim):
        super(FF_policy, self).__init__()

        self.fc1 = nn.Linear(stat_dim, 128)
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, action_dim)
        self.relu = torch.nn.ReLU()
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=0.1)
        self.loss = nn.MSELoss()


    def forward(self, x):
        x = torch.from_numpy(x)
        output = self.fc1(x)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        # output = self.relu(output)

        # output = output.detach().numpy()
        return output


class Buffer(object):
    def __init__(self, buffer_capacity=100000, batch_size=64, states_dim=3, action_dim=2):
        # Number of experiences to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on
        self.batch_size = batch_size
        # Num  recorded instances
        self.buffer_counter = 0

        self.state_buffer = np.zeros((self.buffer_capacity, states_dim))
        self.action_buffer = np.zeros((self.buffer_capacity, action_dim))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, states_dim))

    # Save (s,a,r,s') tuple in buffer
    def save_in_buffer(self, obs_tuple):
        # replacing old records if needed
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # def update(self, state_batch, action_batch, reward_batch, next_state_batch):
    #     target_actions = target_actor(next_state_batch, training=True)
    #
    #     return None

    # Sample from Buffer
    def sample(self):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)
        state_batch = self.state_buffer[batch_indices]
        action_batch = self.action_buffer[batch_indices]
        reward_batch = self.reward_buffer[batch_indices]
        # reward_batch = reward_batch.type(torch.float32)
        next_state_batch = self.next_state_buffer[batch_indices]

        return [np.float32(state_batch), np.float32(action_batch), np.float32(reward_batch), np.float32(next_state_batch)]


    def learn(self):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Sampled indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = torch.from_numpy(self.state_buffer[batch_indices])
        action_batch = torch.from_numpy(self.action_buffer[batch_indices])
        reward_batch = torch.from_numpy(self.reward_buffer[batch_indices])
        reward_batch = reward_batch.type(torch.float32)
        #     torch.dtype(reward_batch, dtype=torch.float32)
        # next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])
        next_state_batch = torch.from_numpy(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))




class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.stored_energy = 0
        self.capacity = np.inf

        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, self.output_dim)
        self.relu = torch.nn.ReLU()
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=0.1)
        self.loss = nn.MSELoss()

    def forward(self, x):
        output = self.fc1(x)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)

        return output

    def take_action(self, state, noise):
        sampled_actions = self.forward(state)
        # noise = noise.sample()
        noise = noise
        # Adding noise to action
        # sampled_actions = sampled_actions.detach().numpy() + noise
        sampled_actions = sampled_actions + noise
        # Project to feasible set if needed
        # legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
        return sampled_actions

    # def update_stored_energy(self, renewable, grid, cmpt):
    #     self.stored_energy = np.maximum(0, np.minimum(self.stored_energy + renewable + grid + cmpt, self.capacity))
    #
    # def projection(self, action, renewable, grid, cmpt, upper_bnd_cmpt, lower_bnd_cmpt, upper_bnd_grid, lower_bnd_grid):
    #     action = action.detach().numpy()
    #
    #     grid_action = np.maximum(self.lower_bnd_grid, np.minimum(action[0], self.upper_bnd_grid))
    #     cmpt_action = np.maximum(self.lower_bnd_cmpt, np.minimum(action[1], self.upper_bnd_cmpt))
    #
    #     self.update_stored_energy(renewable, grid, cmpt)
    #     return action




    # def update(self, state_batch, action_batch, reward_batch, next_state_batch):
    #     actions_batch = self.forward(state_batch)
    #     loss = - torch.mean(critic_model(state_batch, actions_batch))
    #
    #     self.zero_grad()
    #     loss.backward(self.parameters())
    #     self.optimizer.step()


        # with tf.GradientTape() as tape:
        #     actions = actor_model(state_batch, training=True)
        #     critic_value = critic_model([state_batch, actions], training=True)
        #     # Used `-value` as we want to maximize the value given
        #     # by the critic for our actions
        #     actor_loss = -tf.math.reduce_mean(critic_value)
        #
        # actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        # actor_optimizer.apply_gradients(
        #     zip(actor_grad, actor_model.trainable_variables)
        # )


class Critic(nn.Module):
    def __init__(self, stat_dim, action_dim):
        super(Critic, self).__init__()

        self.state_dim = stat_dim
        self.action_dim = action_dim
        self.action_dim = action_dim
        self.concatenation_dim = 10
        # self.gamma = gamma

        self.fc1_s = nn.Linear(self.state_dim, 128)
        self.fc2_s = nn.Linear(128, 84)
        self.fc3_s = nn.Linear(84, self.concatenation_dim)

        self.fc1_a = nn.Linear(self.action_dim, 128)
        self.fc2_a = nn.Linear(128, 84)
        self.fc3_a = nn.Linear(84, self.concatenation_dim)

        self.fc1_c = nn.Linear(self.concatenation_dim * 2, 32)
        self.fc2_c = nn.Linear(32, 16)
        self.fc3_c = nn.Linear(16, 1)

        self.relu = torch.nn.ReLU()

        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=0.1)
        self.loss = nn.MSELoss()

    def forward(self, state, action):
        # if type(state) == np.ndarray:
        #     state = torch.from_numpy(state)
        # if type(action) == np.ndarray:
        #     action = torch.from_numpy(action)

        state = self.fc1_s(state)
        state = self.relu(state)
        state = self.fc2_s(state)
        state = self.relu(state)
        state = self.fc3_s(state)

        action = self.fc1_a(action)
        action = self.relu(action)
        action = self.fc2_a(action)
        action = self.relu(action)
        action = self.fc3_a(action)

        concatenate = torch.cat((state, action), dim=1)
        concatenate = self.fc1_c(concatenate)
        concatenate = self.relu(concatenate)
        concatenate = self. fc2_c(concatenate)
        concatenate = self.relu(concatenate)
        concatenate = self.fc3_c(concatenate)

        return concatenate

    # def update(self, target_actor, target_critic, Buffer):
    #     state_batch, action_batch, reward_batch, next_state_batch = Buffer.sample()
    #     target_actions = target_actor(next_state_batch, training=True) # this should be true or false?
    #     y = reward_batch + self.gamma * target_critic([next_state_batch, target_actions], training=True) # this should be true or false?
    #     critic_value = self.forward(state_batch,action_batch)
    #     loss = self.loss(critic_value, y)
    #
    #     self.zero_grad()
    #     loss.backward()
    #
    #     self.optimizer.step()

class OUNoise(object):
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = np.zeros_like(self.mean)
        self.reset()

    def sample(self):
        # https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=1)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = np.zeros_like(self.mean)

    # Gen Renewable [Gaussian]
class environmrnt(object):
    def __init__(self, state_dim, action_dim, w_cmpt, w_grid, cmpt_dst=None, grid_dist=None, renewable=None,
                 real_renewable=None,
                 real_auction=None):
        self.w_cmpt = w_cmpt
        self.w_grid = w_grid
        self.renewable = renewable
        self.real_renewable = real_renewable
        self.real_grid = real_auction

        self.cmpt = cmpt_dst
        self.grid = grid_dist
        self.state_dim = state_dim
        self.state = self.reset()

    def state_of_env(self):
        return self.state

    def execute(self, action, iteration):
        # TODO: Remove this part that have synthetic data
        cmpt_rv = self.cmpt.rvs(size=1)
        grid_rv = self.grid.rvs(size=1)

        # TODO: Real data for Renewable and Grid
        renewable_rv = self.real_renewable.renewable_sample(iteration)
        # cmpt_rv = self.cmpt.rvs(size=1)
        # grid_rv = self.real_grid.auction_sample(iteration)
        # loss = self.w_cmpt/(self.w_grid + self.w_cmpt) * torch.square(action[0] - cmpt_rv) + \
        #        self.w_grid/(self.w_grid + self.w_cmpt) * torch.square(action[0] - grid_rv)
        # loss = np.square(action.detach().numpy() - cmpt_rv)

        # np_action = action.detach().numpy()
        np_action = action

        # TODO: thoughts form presentation
        # loss = - self.w_cmpt / (self.w_grid + self.w_cmpt) * np.square(self.cmpt.cdf(np_action[0])) - \
        #        self.w_grid / (self.w_grid + self.w_cmpt) * np.square(self.grid.cdf(np_action[1]))

        loss = self.w_cmpt/(self.w_grid + self.w_cmpt) * np.square(np_action[0] - cmpt_rv) + \
                self.w_grid/(self.w_grid + self.w_cmpt) * np.square(np_action[1] - grid_rv)

        state = np.array([np.squeeze(renewable_rv), np.squeeze(grid_rv), np.squeeze(self.cmpt.rvs(size=1))], dtype='float32')
        self.state = state

        return loss, self.state

    def reset(self):
        # re = []
        # for i in range(self.state_dim):
        #     re.append(np.random.random(size=1).astype('float32'))
        re = np.random.random(size=self.state_dim).astype('float32')
        return re

# def renew_gen():
#     def __ini__(self, mean, var):
#         self.mean = mean
#         self.var = var