"""
MABP:- Creating simulated data for Multi Armed Bandit problem (N armed TestBed)
"""
import numpy as np
from collections import defaultdict
from typing import List
from numpy.core.fromnumeric import var

from seaborn.categorical import factorplot

flag = 0
class Bandit:
    def __init__(self, action_id, mean):
        """
        Constructor for a single bandit

        Args:
        action_id : unique id of an action
        mean: True mean of the reward distribution
        """
        self.id = action_id
        self.mean = mean

    def pull_arm(self):

        """
        function returning observed rewards after pressing lever

        with mean = self.mean and sigma = 1
        """
        return np.random.randn()+self.mean

class Logging:
    """
    class for implementing  
    """
    def __init__(self):
        self.total_actions = 0
        self.total_rewards = 0
        self.all_rewards = []
        self.all_actions = []
        self.record = defaultdict(lambda: dict(actions=0, reward=0))

    def record_action(self, bandit, reward):
        self.total_actions += 1
        self.total_rewards += reward
        self.all_rewards.append(reward)
        self.all_actions.append(bandit.id)
        self.record[bandit.id]['actions'] += 1
        self.record[bandit.id]['reward'] += reward

    def __getitem__(self, bandit):
        return self.record[bandit.id]

class EpsilonGreedyAgent:
    def __init__(self, bandits: List[Bandit], epsilon:float = None):
        """
        Constructor for Epsilon Greedy Agent on 10 armed test bed
        """
        self.bandits = bandits
        self.epsilon = epsilon
        # mu_pri and var_pri are hypermaterers for prior distribution
        self.mu_pri = np.zeros(len(self.bandits))
        self.var_pri = np.ones(len(self.bandits))*5
        # mu and var are hyperparameters for posterior distribution
        self.mu = self.mu_pri
        self.var = self.var_pri
        self.var0 = 1 # 1 is taken but any can be taken no prob upto a limit constant of inintial distribution
        self.logging = Logging()

    def _get_random_bandit(self)-> Bandit:
        """
        random choice of bandits 
        """
        return np.random.choice(self.bandits)
    
    def _get_max_estimated_bandit(self)->Bandit:
        """
        this function is used to estimate model based on mean/mode
        """
      #  print("mus - ", self.mu)
       # print("actions - ", np.argmax(self.mu))
        unique, counts = np.unique(self.mu, return_counts=True)
        lens = counts[np.argmax(unique)]  
        if lens>1: # if two actions have same argmax
            # then return arbitrarily from those max ones
            maxs = list(np.array(self.bandits)[self.mu==unique[np.argmax(unique)]])
            return np.random.choice(maxs)
        # otherwise return the max one
        return self.bandits[np.argmax(self.mu)]

    def _update(self, bandit):
        """
        returning maximum one using sam[ple averaging method
        """   
        
        bandit_logs = self.logging[bandit]
        bandit = bandit.id
        estimate = bandit_logs['reward'] / bandit_logs['actions'] # if not assigned
        actions =  bandit_logs['actions']
        self.mu[bandit] = (self.mu_pri[bandit]/self.var_pri[bandit] + actions*estimate/self.var0)/(actions/self.var0 + 1/self.var_pri[bandit])
        self.var[bandit] = 1/(actions/self.var0 + 1/self.var[bandit])

        
    def _choose_bandit(self)->Bandit:
        epsilon = self.epsilon
        global flag
        p = np.random.uniform(0, 1, 1)
        if p < epsilon or (flag==0 and epsilon==0):
            flag = 1
            bandit = self._get_random_bandit()
        else:
            bandit = self._get_max_estimated_bandit()

        return bandit
    def action(self):
        current_bandit = self._choose_bandit()
        reward = current_bandit.pull_arm()
        self.logging.record_action(current_bandit, reward)
        self._update(current_bandit)
        
    def actions(self, timesteps):
        for _ in range(timesteps):
            self.action()
    
        return self.logging.all_rewards, self.logging.all_actions


class ThomspsonSamplingAgent:
    
    def __init__(self, bandits: List[Bandit]):
        """
        Constructor for Thompson Sampling Agent on 10 armed test bed
        """
        self.bandits = bandits
        # mu_pri and var_pri are hypermaterers for prior distribution
        self.mu_pri = np.zeros(len(self.bandits)) # 5 for each
        self.var_pri = np.ones(len(self.bandits))*5
        # mu and var are hyperparameters for posterior distribution
        self.mu = self.mu_pri
        self.var = self.var_pri
        self.var0 = 1 # 1 is taken but any can be taken no prob upto a limit constant of inintial distribution
        self.logging = Logging()


    def _get_max_sampled_bandit(self)->Bandit:
        """
        this function is used to estimate model based on mean/mode
        """
        estimates = []
        for bandit in self.bandits:
            estimates.append(np.random.normal(loc =self.mu[bandit.id], scale = self.var[bandit.id]))
        return self.bandits[np.argmax(estimates)]

    def _update(self, bandit):
        """
        returning maximum one using sam[ple averaging method
        """   
        
        bandit_logs = self.logging[bandit]
        bandit = bandit.id
        if not bandit_logs['actions']:
            estimate = 0 # if not taken till now then 0 is assigned
            actions = 0
        else:
            estimate = bandit_logs['reward'] / bandit_logs['actions'] # if not assigned
            actions =  bandit_logs['actions']
        self.mu[bandit] = (self.mu_pri[bandit]/self.var_pri[bandit] + actions*estimate/self.var0)/(actions/self.var0 + 1/self.var_pri[bandit])
        self.var[bandit] = 1/(actions/self.var0 + 1/self.var[bandit])

        
    def _choose_bandit(self)->Bandit:
        
        bandit = self._get_max_sampled_bandit()

        return bandit
    def action(self):
        current_bandit = self._choose_bandit()
        reward = current_bandit.pull_arm()
        self.logging.record_action(current_bandit, reward)
        self._update(current_bandit)
        
    def actions(self, timesteps):
        for _ in range(timesteps):
            self.action()
    
        return self.logging.all_rewards, self.logging.all_actions

class DoubleSampling:
    pass