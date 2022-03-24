import numpy as np
import torch as T
from models_pytorch import ActorCritic
from memory import Memory
from environments import Simulator
from utils import plot_portfolio


def a3c_worker(name, input_dims, n_actions, global_agent, optimizer, data, initial_investment, bar_length, time_horizon, units, max_eps = 100, n_threads = 4, T_max = 20):
    local_agent = ActorCritic(input_dims, n_actions)

    memory = Memory()
    env = Simulator(data, initial_investment, bar_length, time_horizon, units)

    episode, t_steps, scores = 0, 0, []

    while episode < max_eps:
        obs = env.reset()
        score, done, ep_steps = 0, False, 0
        hx = T.zeros(1, 256)
        while not done:
            state = T.tensor(obs, dtype = T.float)
            action, value, log_prob, hx = local_agent(state, hx)
            obs_, reward, done, info = env.step(action)
            memory.remember(reward, value, log_prob)
            score += reward
            obs = obs_
            ep_steps += 1
            t_steps += 1

            if ep_steps % T_max == 0 or done:
                rewards, values, log_probs = memory.sample_memory()
                loss = local_agent.calc_cost(new_state = obs, hx = hx, done = done, rewards = rewards, values = values, log_probs = log_probs)
                optimizer.zero_grad()
                hx = hx.detach_()
                loss.backward(retain_graph = True)
                T.nn.utils.clip_grad_norm_(local_agent.parameters(), 40)
                for local_param, global_param in zip(local_agent.parameters(), global_agent.parameters()):
                    global_param._grad = local_param.grad 
                    optimizer.step()
                    local_agent.load_state_dict(global_agent.state_dict())
                    memory.clear_memory()
        episode += 1
        if name == "1":
            scores.append(score)
            avg_score = np.mean(scores[-100: ])
            print("A3C episode {}, thread {} of {}, steps {:.2f}, score {:.2f}, avg_score (100) {:.2f}".format(episode, name, n_threads, t_steps/1e6, score, avg_score))

    if name == "1":
        x = [z for z in range(episode)]
        plot_portfolio(x, scores, "A3C_portfolio_final.png")
