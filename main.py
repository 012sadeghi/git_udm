"""
1- change Hello
2- change is to modify the code on github (implicitly using git);
3- git reset HEAD <file name>
4- git checkout --
"""

import pandas
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from celluloid import Camera


from solar_real_data import solar_generation
from auction_volums import auction_data_generation
import numpy as np
import csv
from itertools import count
from dual_decomposition import dual_decomp_agent
from utilities import FF_v_estimator, Actor, Critic, Buffer, environmrnt, MyRandomVariableClass_grid, \
    ComptRandomVariableClass_cmpt, ComptRandomVariableClass_renewable
import torch
from matplotlib import pyplot as plt

np.random.seed(seed=1)
torch.manual_seed(0)
import os

# plt.close(all)
# TODO: Solar Forecast
start1 = 286866
end1 = 287250
start2, end2 = 286972, 287020
start3, end3 = 287050, 287121
start4, end4 = 287159, 287219


# Load data
# df = pandas.read_csv("real_data/Elia_dso.csv")

# start, end = start4, end4
# # TODO: Generation power demand
# # d_realtime = df['Real-time Upscaled Measurement [MW]']
# # x = np.sort(d_realtime[start:end])
# # y = np.arange(np.size(d_realtime[start:end]))/np.size(d_realtime[start:end])
# # ax0 = plt.plot(x, y)
#
# # d_realtime = df['Real-time Upscaled Measurement [MW]']
# # x = np.sort(d_realtime[start:end])
# # y = np.arange(np.size(d_realtime[start:end]))/np.size(d_realtime[start:end])
# # ax1 = plt.plot(x, y)
# #
# # d_mostrecent = df['Most recent forecast [MW]']
# # x = np.sort(d_mostrecent[start:end])
# # y = np.arange(np.size(d_mostrecent[start:end]))/np.size(d_mostrecent[start:end])
# # ax2 = plt.plot(x, y)
# #
# #
# # d_dayahead = df['Day-Ahead forecast [MW]']
# # x = np.sort(d_dayahead[start:end])
# # y = np.arange(np.size(d_dayahead[start:end]))/np.size(d_dayahead[start:end])
# # ax3 = plt.plot(x, y)
# #
# #
# # d_weekahead = df['Week-Ahead forecast [MW]']
# # # count, bins_count = np.histogram(d_weekahead, bins=10)
# # # pdf = count / sum(count)
# # # cdf = np.cumsum(pdf)
# # # plt.plot(bins_count[1:], pdf, color="red", label="PDF")
# # # plt.plot(bins_count[1:], cdf, label="CDF")
# # # plt.legend()
# # x = np.sort(d_weekahead[start:end])
# # y = np.arange(np.size(d_weekahead[start:end]))/np.size(d_weekahead[start:end])
# # ax4 = plt.plot(x, y)
# #
# #
# # plt.xlabel('Power')
# # plt.ylabel('CDF')
# # plt.ylim(-.01, 0.99999)
# # plt.legend(['Real-time Measurement [MW]','Most recent forecast [MW]','Day-Ahead forecast [MW]','Week-Ahead forecast [MW]'])
# # # plt.show()
#
# # sample = df.sample(n=1)
# # print(sample)
#
# ## PDFs
# ax2 = df['Real-time Upscaled Measurement [MW]'][start:end].plot.kde()
# ax2 = df['Most recent forecast [MW]'][start:end].plot.kde()
# ax2 = df['Day-Ahead forecast [MW]'][start:end].plot.kde()
# ax2 = df['Week-Ahead forecast [MW]'][start:end].plot.kde()
# plt.legend(['Real-time Measurement [MW]','Most recent forecast [MW]','Day-Ahead forecast [MW]','Week-Ahead forecast [MW]'])
#
#
# # plt.show()
#
# samples = df['Day-Ahead forecast [MW]'][start:end].sample(n=3)
# sample = samples.sample(n=1)
# print(samples)
# ax = df.plot(y="Real-time Upscaled Measurement [MW]", kind='line', xlim=(start, end))
# df.plot(ax=ax, y="Most recent forecast [MW]", kind='line', xlim=(start, end))
# df.plot(ax=ax, y="Day-Ahead forecast [MW]", kind='line', xlim=(start, end))
# df.plot(ax=ax, y="Week-Ahead forecast [MW]", kind='line', xlim=(start, end))
# df.plot(ax=ax, y="Day-Ahead forecast (11h00) [MW]", kind='line', xlim=(start, end))

# TODO: State generator module
class state_gene_(object):
    def __init__(self, shape):
        self.name = "state_generator"
        self.shape = shape
        self.capacity = +100
        self.state = np.zeros(shape=self.shape, dtype=np.float64)

    def restart(self):
        self.state = np.zeros(shape=self.shape, dtype=np.float64)

    def update(self, rw, wkld, energy_trade):
        self.state[0] = rw
        self.state[1] = wkld
        self.state[2] = max(min(self.state[2] + rw + energy_trade, self.capacity), 0)

    def state_status(self):
        return self.state


# TODO: Define a neural network for value function estimation
cnt_state = state_gene_(shape=3)
print(cnt_state.state_status())


class v_estimateor(object):
    def __init__(self, gamma, input_shpae, output_shape, no_layers, activation):
        self.gamma = gamma
        self.state_dim = input_shpae
        self.action_dim = 3
        self.outputshape = output_shape
        self.number_layers = no_layers
        self.activations = activation
        self.v_estimator = FF_v_estimator(self.state_dim, self.outputshape, self.number_layers, self.activations,
                                          self.gamma)
        self.policy = FF_policy(self.state_dim, self.action_dim)
        self.loss = self.v_estimator.loss
        self.optimizer = self.v_estimator.optimizer

    def update(self, st, s_tp, cost):
        # st = torch.from_numpy(st)
        # s_tp = torch.from_numpy(s_tp)
        cost = torch.from_numpy(cost)

        value = self.v_estimator.forward(st)
        target = cost + self.gamma * value

        # Next state
        next_value = self.v_estimator.forward(s_tp)
        self.v_estimator.optimizer.zero_grad()
        loss = self.v_estimator.loss(value + target, next_value)
        loss.backward()
        self.v_estimator.optimizer.step()

        return loss.detach().numpy()

    def v_score(self, st):
        return self.v_estimator.forward(st)

    def action(self, st):
        # TODO: explore

        # TODO: exploit
        x1, x2 = self.policy(st)
        x1, x2 = self.project(x1, x2)
        return

    # TODO: How to make updates ?
    #    Done
    # TODO: update neural network weights
    #   Done

    # TODO: for a given inout what is the output?

    # TODO: optimize the cost


#
# T = 1000
# v_est = v_estimateor(0.999, 3, 1, 3, "relu")
# input = np.ones(shape=(T+1, 3), dtype='float32')
# output = np.ones(shape=(T, 1), dtype='float32') * 10
#
# for t in range(T):
#     st = input[t]
#     v_value = v_est.v_score(st)
#
#     st_p = input[t+1]
#     isupdated = v_est.update(st, st_p, output[t])
#     print(['loss is', isupdated])
#     print(v_value.detach().numpy())

# print(v_est.v_score(st).detach().numpy())

# TODO: Wind Forecast

# TODO: Kernel density estimation of the PDFs

class Agent(object):
    def __init__(self, state_dim, action_dim, buffer_size, gamma, min_capacity=0, max_capacity=np.inf):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.stored_energy = 0
        # TODO: interesting observation this max- and min-capacity play a crucial role in the generated results
        #  :=> just changing them as hyper parameter might be interesting
        self.max_capacity =  max_capacity   #MWh
        self.min_capacity = min_capacity

        self.action = np.zeros(shape=action_dim)
        self.buffer = Buffer(buffer_capacity=buffer_size, batch_size=64, states_dim=state_dim, action_dim=action_dim)

        # Actor
        self.actor = Actor(self.state_dim, self.action_dim)
        self.actor_target = Actor(self.state_dim, self.action_dim)

        # Critic
        self.critic = Critic(self.state_dim, self.action_dim)
        self.critic_target = Critic(self.state_dim, self.action_dim)

    def take_action(self,
                    state,
                    noise,
                    upper_bnd_cmpt=np.inf,
                    lower_bnd_cmpt=0,
                    upper_bnd_grid=np.inf,
                    lower_bnd_grid=-np.inf):

        agent_action = self.actor.take_action(state, noise)

        agent_projected_action = self.projection(
            agent_action,
            renewable=state[0],
            upper_bnd_cmpt=upper_bnd_cmpt,
            lower_bnd_cmpt=lower_bnd_cmpt,
            upper_bnd_grid=upper_bnd_grid,
            lower_bnd_grid=lower_bnd_grid)


        return agent_projected_action

    def update_stored_energy(self, grid_action_, cmpt_action_, renewable_):
        self.stored_energy = self.stored_energy + renewable_ - grid_action_ - cmpt_action_
        #     np.maximum(0, np.minimum(self.stored_energy + renewable_ - grid_action_ - cmpt_action_
        #                                              , self.capacity))
        # print("stored enery input:", renewable_ - grid_action_ - cmpt_action_)

    def energy_level(self):
        return self.stored_energy

    def projection(self,
                   action,
                   renewable,
                   upper_bnd_cmpt,
                   lower_bnd_cmpt,
                   upper_bnd_grid,
                   lower_bnd_grid):
        '''
        Prjecting action into a "feasible" action
        action: two dimensional tensor; how much energy buying/selling to grid; and allocating to computation market
        '''

        action = action.detach().numpy()

        grid_action = np.maximum(lower_bnd_grid, np.minimum(action[0], upper_bnd_grid))
        cmpt_action = np.maximum(lower_bnd_cmpt, np.minimum(action[1], upper_bnd_cmpt))

        self.stored_energy = np.minimum(self.stored_energy + renewable - grid_action - cmpt_action, self.max_capacity)

        if self.stored_energy < self.min_capacity:
            grid_action = self.stored_energy
            self.stored_energy = self.min_capacity


        # self.update_stored_energy(grid_action, cmpt_action, renewable)
        # Return projected action
        re_action = np.array([grid_action, cmpt_action])

        # Update energy buffer
        # self.update_stored_energy(
        #     grid_action_=grid_action,
        #     cmpt_action_=cmpt_action,
        #     renewable_=renewable)

        return re_action

    def critic_update(self):
        state_batch, action_batch, reward_batch, next_state_batch = self.buffer.sample()
        target_actions = self.actor_target.forward(torch.from_numpy(next_state_batch))  # this should be true or false?
        y = torch.from_numpy(reward_batch) + self.gamma * self.critic_target.forward(torch.from_numpy(next_state_batch),
                                                                                     target_actions)  # this should be true or false?
        critic_value = self.critic.forward(torch.from_numpy(state_batch), torch.from_numpy(action_batch))
        loss = self.critic.loss(critic_value, y)

        self.critic.zero_grad()
        self.actor_target.zero_grad()
        self.critic.zero_grad()
        self.critic_target.zero_grad()

        loss.backward()
        self.critic.optimizer.step()

    def actor_update(self):
        state_batch, action_batch, reward_batch, next_state_batch = self.buffer.sample()

        actions_drawn_from_policy = self.actor.forward(torch.from_numpy(state_batch))

        # actions_batch = actions_batch.type(torch.float32)
        value = self.critic.forward(torch.from_numpy(state_batch), actions_drawn_from_policy)
        loss = torch.mean(value)

        self.actor.zero_grad()
        self.actor_target.zero_grad()
        self.critic.zero_grad()
        self.critic_target.zero_grad()

        loss.backward()
        self.actor.optimizer.step()

    def update_actor_target(self, tau):
        # TODO: is this doing what I want?
        for (a, b) in zip(self.actor.parameters(), self.actor_target.parameters()):
            b.data = (1 - tau) * b.data + tau * a.data

    def update_critic_target(self, tau):
        # TODO: Check this too
        for (a, b) in zip(self.critic.parameters(), self.critic_target.parameters()):
            b.data = (1 - tau) * b.data + tau * a.data



cmpt_dst = ComptRandomVariableClass_cmpt()
grid_dst = MyRandomVariableClass_grid()
renewable = ComptRandomVariableClass_renewable()
real_renewable = solar_generation()
real_auction = auction_data_generation()

T = 500
C = 10
my_agent = Agent(3, 2, 100, 0.9, min_capacity=2, max_capacity=5)
my_dual_deecom_agent = dual_decomp_agent(action_dim=2, state_dim=3, w_cmpt=1, w_grid=1,
                                         cmpt_dst=cmpt_dst, grid_dist=grid_dst,
                                         real_renewable=real_renewable, real_auction=real_auction,
                                         min_capacity=0, max_capacity=10)


env = environmrnt(state_dim=3, action_dim=2, w_cmpt=1, w_grid=1, cmpt_dst=cmpt_dst, grid_dist=grid_dst,
                  renewable=renewable, real_renewable=real_renewable, real_auction=real_auction)

# def __init__(self, state_dim, action_dim, w_cmpt, w_grid, cmpt_dst=None, grid_dist=None, renewable=None):

energy_in = 0
energy_out = 0

report, dual_decomp_report = dict(), dict()

report['loss'] = []
report['action'] = []
report['state'] = []
report['action'] = []
report['stored energy'] = []
report['renewable'] = []
report['energy_in_out'] = []
report['auction_volums'] = []

dual_decomp_report['dual_actions'] = []
dual_decomp_report['dual_variable'] = []
dual_decomp_report['dual_energy_level'] = []
dual_decomp_report['dual_renewable_injection'] = []



for t in range(T):
    old_state = env.state_of_env()
    # st = input[t]
    action = my_agent.take_action(torch.from_numpy(old_state), np.random.randn() * 0.2)
    loss, state = env.execute(action=action, iteration=t)
    # loss = np.square(action.detach().numpy() - output[t])
    # print("real_auction sample", real_auction.auction_sample(t))

    dual_decomposition_action, deual_variable, loss_dual, dual_energy_level, dual_renewable_injections = \
        my_dual_deecom_agent.take_action(state=old_state, iteration=t)

    energy_in = energy_in + np.sum(action) + state[0]
    energy_out = energy_out + np.sum(action)
    report['loss'].append(loss)
    report['state'].append(state)
    report['action'].append(action)
    report['stored energy'].append(my_agent.energy_level())
    report['renewable'].append(old_state[0])
    report['energy_in_out'].append(np.append(action, old_state[0]))
    report['auction_volums'].append(real_auction.auction_sample(t))
    print('=======================RL============================')
    print('iteration', t, '| action', action, '| loss', loss)
    print('=======================Opt============================')
    print('Opt==>iteration', t, '| action', dual_decomposition_action, 'energy_level', dual_energy_level, '|loss', loss_dual)
    print('===================================================')
    dual_decomp_report['dual_variable'].append(deual_variable)
    dual_decomp_report['dual_actions'].append(dual_decomposition_action)
    dual_decomp_report['dual_energy_level'].append(dual_energy_level)
    dual_decomp_report['dual_renewable_injection'].append(dual_renewable_injections)



    # Dual params update
    my_agent.buffer.save_in_buffer((old_state, action, loss, state))
    my_agent.actor_update()
    my_agent.critic_update()

    if t % C == 0:
        my_agent.update_actor_target(tau=0.1)
        my_agent.update_critic_target(tau=0.1)

index = count()

# fig1 = plt.figure()
# ax_loss = fig1.add_subplot(1, 2, 1)


fig_rl = plt.figure()
ax_energylevel = fig_rl.add_subplot(4, 1, 1)

ax_actions = fig_rl.add_subplot(4, 1, 2)

ax_renewable = fig_rl.add_subplot(4, 1, 3)

#
# ax_loss = fig_rl.add_subplot(5, 1, 4)

ax_auction = fig_rl.add_subplot(4, 1, 4)

# ax_energy_in_out = fig_rl.add_subplot(5, 1, 5)


# fig_dual = plt.figure()
# # ax_dual_energy_level = fig_dual.add_subplot(4, 1, 1)
# ax_dual_variable = fig_dual.add_subplot(3, 1, 1)
# ax_dual_action = fig_dual.add_subplot(3, 1, 2)
# ax_dual_renewable = fig_dual.add_subplot(3, 1, 3)


# plt.gca().set_color_cycle(['red', 'green', 'blue', 'yellow', 'black'])
camera = Camera(fig_rl)

for idx in range(T):

    # ax_energylevel.cla()
    ax_energylevel.plot(report['stored energy'][:idx], color="blue")
    ax_energylevel.set_ylabel('stored energy')
    ax_energylevel.grid()
    # a = report['action'][:idx]

    # ax_actions.cla()
    ax_actions.set_prop_cycle(color=['green', 'red'])
    ax_actions.plot(report['action'][:idx])
    # ax_actions.plot(report['action'][:idx], color='green')
    ax_actions.set_ylabel('action')
    ax_actions.grid()
    ax_actions.set_ylim(-5, 5)

    # ax_actions.yscale('log')
    # ax_energylevel.set_ylim(0, 50)
    ax_actions.legend(['utility market', 'computation market'], loc='lower left')
    # ax_actions.legend(loc='upper left')
    # ax_actions.tight_layout()
    plt.grid()

    # ax_renewable.cla()
    ax_renewable.plot(report['renewable'][:idx], color="black")
    ax_renewable.set_ylabel('Renewable')
    ax_renewable.grid()
    # ax_energylevel.set_ylim(0, 50)

    # ax_auction.cla()
    ax_auction.plot(report['auction_volums'][:idx], color="green")
    ax_auction.set_ylabel('auction_volums')
    ax_auction.grid()

    # ax_energy_in_out.cla()
    # ax_energy_in_out.plot(report['energy_in_out'][:idx], color="brown")
    # ax_energy_in_out.set_ylabel('energy in and out')
    # ax_energy_in_out.grid()

    # plt.pause(0.000000000000000000000000000000001)
    camera.snap()

animation = camera.animate()
animation.save('animation_alireza4.gif', fps=10)

# plt.show()

def animate(i):
    idx = next(index)

#     # ax_dual_energy_level.cla()
#     # ax_dual_energy_level.plot(dual_decomp_report['dual_energy_level'][:idx])
#     # ax_dual_energy_level.set_ylabel('energy level')
#     # # plt.xscale('log')
#     # ax_dual_energy_level.grid()
#
#     ax_dual_variable.cla()
#     ax_dual_variable.plot(dual_decomp_report['dual_variable'][:idx])
#     ax_dual_variable.set_ylabel('dual variable')
#     # plt.xscale('log')
#     ax_dual_variable.grid()
#
#     ax_dual_action.cla()
#     ax_dual_action.plot(dual_decomp_report['dual_actions'][:idx])
#     ax_dual_action.set_ylabel('actions')
#     # plt.xscale('log')
#     ax_dual_action.grid()
#
#     ax_dual_renewable.cla()
#     ax_dual_renewable.plot(dual_decomp_report['dual_renewable_injection'][:idx])
#     ax_dual_renewable.set_ylabel('Renewable')
#     # plt.xscale('log')
#     ax_dual_renewable.grid()
#     #

    #
    ax_energylevel.cla()
    ax_energylevel.plot(report['stored energy'][:idx])
    ax_energylevel.set_ylabel('stored energy')
    ax_energylevel.grid()
    # ax_energylevel.set_ylim(0, 50)

    ax_actions.cla()
    ax_actions.plot(report['action'][:idx])
    ax_actions.set_ylabel('action')
    ax_actions.grid()
    ax_actions.set_ylim(-4, 4)
    # ax_actions.yscale('log')
    # ax_energylevel.set_ylim(0, 50)
    ax_actions.legend(['utility market', 'computation market'], loc='lower left')
    # ax_actions.legend(loc='upper left')
    # ax_actions.tight_layout()
    plt.grid()

    ax_renewable.cla()
    ax_renewable.plot(report['renewable'][:idx])
    ax_renewable.set_ylabel('Renewable')
    ax_renewable.grid()
    ax_energylevel.set_ylim(0, 50)


    ########################
    ax_auction.cla()
    ax_auction.plot(report['auction_volums'][:idx])
    ax_auction.set_ylabel('auction_volums')
    ax_auction.grid()

    # ax_energy_in_out.cla()
    # ax_energy_in_out.plot(report['energy_in_out'][:idx])
    # ax_energy_in_out.set_ylabel('energy in and out')
    # ax_energy_in_out.grid()

    # take an snap with camera
    # camera.snap()

    # ax_energylevel.set_ylim(0, 50)
    # ax_energylevel.set_ylim(0, 50)
    # ax_loss.cla()
    # ax_loss.plot(report['loss'][:idx])
    # ax_loss.set_ylabel('loss')
    # ax_loss.grid()

    # plt.grid()
    # fig_loss = plt.xlabel('iteration')
    # plt.ylabel('loss')
    # plt.xscale('log')
    # plt.grid()
    # plt.close()


# print('cumulative loss', np.mean(report['loss']), '| storage capacity', my_agent.max_capacity)
ani = FuncAnimation(fig_rl, animate, interval=1)
plt.show()
#


# plt.show()
with open('mycsvfile.csv', 'w') as f:
    writer = csv.DictWriter(f, fieldnames=report.keys())
    writer.writeheader()
    writer.writerow(rowdict=report)

    # fig_loss = plt.figure()
    # ax_loss = fig_loss.add_subplot(1,1,1)

    # def animate_actions():
    #     y = data['loss']
    #     x = range(len(y))
    #     # print(y[index])
    #     print(index)
    #     ax_loss.clear()
    #     ax_loss.plot(y)


    # for key in report.keys():
    #     f.write("%s,%s\n"%(key, report[key]))

# print("Day-Ahead forecast (11h00) [MW]")
# print(df["DateTime"[0]].dtypes)
# ax = df.plot.area()

# pandas.DataFrame.plot(["Day-Ahead forecast (11h00) [MW]")
# df["Day-Ahead forecast (11h00) [MW]"]
# df["Day-Ahead forecast (11h00) [MW]"].plot()


# df["Day-Ahead forecast"].plot.line
# plt.show()

# print(df)
# df["Day-Ahead forecast"].plot()
# print(df)

# os.chdir("")
# data_path = os.path.abspath

# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
#
# a
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')
#
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
