import numpy as np
import pandas as pd

class Simulator:
    
    def __init__(self, data, initial_investment, bar_length, time_horizon, units = 100000):
        self.data = data
        self.initial_investment = initial_investment
        self.bar_length = pd.to_timedelta(bar_length)
        self.units = units
        self.time_horizon = pd.to_timedelta(time_horizon)
        
        self.position = None
        self.cur_step = None
        self.owned = None
        self.price = None
        self.cash_in_hand = None
        
        self.action_space = [-1, 0 ,1]
        
        self.reset()
  
    def reset(self):
        self.position = 0
        self.cur_step = 0
        self.owned = 0
        self.price = self.data.iloc[0][["Ask", "Bid"]].values
        self.cash_in_hand = self.initial_investment
        
        initial_state = self._get_obs()
        return initial_state
    
    def step(self, action):
        prev_val = self._get_val()
        
        self.cur_step += 1
        self.price = self.data.iloc[self.cur_step][["Ask", "Bid"]].values
        
        end_of_session = self.time_horizon / self.bar_length   # close any open positions at end of day
        if self.cur_step % end_of_session == 0:
            self._trade(0)
        else:
            self._trade(action)
        
        cur_val = self._get_val()
        
        reward = cur_val - prev_val
        
        done = self.cur_step == len(self.data) - 1
        
        info = {"cur_val" : cur_val}
               
        return self._get_obs(), reward, done, info
    
    def _get_obs(self): # input features should go here
        inputs = self.data.iloc[self.cur_step].values
        return np.concatenate([np.array([self.owned, self.cash_in_hand]), inputs])
        
    def _get_val(self):
        if self.position == 1:
            price = self.price[1]
        elif self.position == -1:
            price = self.price[0]
        elif self.position == 0:
            price = 0
        return self.cash_in_hand + self.owned * price
    
    def _trade(self, action):
        if action == 1:   # go long (buy)
            if self.position == 0:
                self.cash_in_hand -= self.units * self.price[0]
                self.owned = self.units
            if self.position == -1:
                self.cash_in_hand += (self.owned - self.units) * self.price[0]
                self.owned = self.units
            self.position = 1
        elif action == -1:   # go short (sell)
            if self.position == 0:
                self.cash_in_hand += self.units * self.price[1]
                self.owned = -self.units
            elif self.position == 1:
                self.cash_in_hand += (self.owned + self.units) * self.price[1]
                self.owned = -self.units
            self.position = -1
        elif action == 0:   # go neutral
            if self.position == 1:
                self.cash_in_hand += self.owned * self.price[1]
                self.owned = 0
            elif self.position == -1:
                self.cash_in_hand += self.owned * self.price[0]
                self.owned = 0
            self.position = 0