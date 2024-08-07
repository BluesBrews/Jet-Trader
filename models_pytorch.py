import numpy as np 
import torch as T 
import torch.nn as nn
import torch.nn.functional as F 
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions = 3, gamma = 0.99, tau = 1.0):
        super(ActorCritic, self).__init__()

        self.gamma = gamma
        self.tau = tau

        self.conv1 = nn.Conv1d(input_dims, 32, 3, stride = 2, padding = 1)
        self.conv2 = nn.Conv1d(32, 32, 3, stride = 2, padding = 1)
        self.conv3 = nn.Conv1d(32, 32, 3, stride = 2, padding = 1)
        self.conv4 = nn.Conv1d(32, 32, 3, stride = 2, padding = 1)

        conv_shape = self.calc_conv_output(input_dims)

        self.gru = nn.GRUCell(conv_shape, 256)
        self.pi = nn.Linear(256, n_actions)
        self.v = nn.Linear(256, 1)

    def calc_conv_output(self, input_dims):
        state = T.zeros(1, input_dims, 1)
        
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        dims = self.conv4(dims)

        return int(np.prod(dims.size()))

    def forward(self, state, hx):
        conv = F.elu(self.conv1(state))
        conv = F.elu(self.conv2(conv))
        conv = F.elu(self.conv3(conv))
        conv = F.elu(self.conv4(conv))
        
        conv_state = conv.view((conv.size()[0], -1))

        hx = self.gru(conv_state, (hx))

        pi = self.pi(hx)
        v = self.v(hx)
        
        probs = T.softmax(pi, dim = 1)
        dist = Categorical(probs)
        action = dist.sample()
        log_probs = dist.log_prob(action)

        return action.numpy()[0], v, log_probs, hx

    def calc_R(self, done, rewards, values):
        values = T.cat(values).squeeze()

        if len(values.size()) == 1:   # batch of states
            R = values[-1] * (1 - int(done))
        elif len(values.size()) == 0:   # single state
            R = values * (1 - int(done))
        
        batch_return = []
        for reward in rewards[ :: -1]:
            R = self.gamma * R + reward
            batch_return.append(R)
        batch_return.reverse()
        batch_return = T.tensor(batch_return, dtype = T.float).reshape(values.size())
        return batch_return

    def calc_cost(self, new_state, hx, done, rewards, values, log_probs, beta = 0.01):
        returns = self.calc_R(done, rewards, values)

        next_v = T.zeros(1, 1) if done else self.forward(T.tensor(new_state, dtype = T.float), hx)[1]
    
        values.append(next_v.detach())
        values = T.cat(values).squeeze()
        log_probs = T.cat(log_probs)
        rewards = T.tensor(rewards)

        delta_t = rewards + self.gamma * values[1 : ] - values[ : -1]
        n_steps = len(delta_t)
        gae = np.zeros(n_steps)

        for t in range(n_steps):
            for k in range(n_steps - t):
                temp = (self.gamma * self.tau) ** k * delta_t[t+k]
                gae[t] += temp
        gae = T.tensor(gae, dtype = T.float)

        actor_loss = -(gae * log_probs).sum()
        critic_loss = F.mse_loss(values[: -1].squeeze(), returns)
        entropy_loss = (-log_probs * T.exp(log_probs)).sum()

        total_loss = actor_loss + critic_loss - beta * entropy_loss
        return total_loss


