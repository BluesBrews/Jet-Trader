import torch.multiprocessing as mp
from models_pytorch import ActorCritic
from shared_optimizers import SharedAdam
from workers import a3c_worker

class ParallelEnv():
    def __init__(self, input_dims, data, initial_investment, bar_length, time_horizon, units, n_actions = 3, n_threads = 4):
        names = [str(i) for i in range(n_threads)]

        global_agent = ActorCritic(input_dims, n_actions = n_actions)
        global_agent.share_memory()
        global_optimizer = SharedAdam(global_agent.parameters())

        self.ps = [mp.Process(target = a3c_worker, args = (name, input_dims, n_actions, global_agent, global_optimizer, data, initial_investment, bar_length, time_horizon, units))
                                for name in names]
        
        [p.start() for p in self.ps]
        [p.join() for p in self.ps]
