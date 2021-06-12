"""
MABP:- Creating simulated data for Multi Armed Bandit problem (N armed TestBed)
"""
import numpy as np
from collections import defaultdict
from typing import List

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
        self.logging = Logging()

    def _get_random_bandit(self)-> Bandit:
        """
        random choice of bandits 
        """
        return np.random.choice(self.bandits)

    def _get_max_estimated_bandits(self)->Bandit:
        """
        returning maximum one using sam[ple averaging method
        """   
        estimates = []     
        for bandit in self.bandits:
            bandit_logs = self.logging[bandit]
            if not bandit_logs['actions']:
                estimates.append(0) # if not taken till now then 0 is assigned
            else:
                estimates.append(bandit_logs['reward'] / bandit_logs['actions']) # if not assigned

        return self.bandits[np.argmax(estimates)]
    def _choose_bandit(self)->Bandit:
        epsilon = self.epsilon

        p = np.random.uniform(0, 1, 1)
        if p < epsilon:
            bandit = self._get_random_bandit()
        else:
            bandit = self._get_max_estimated_bandits()

        return bandit
    def action(self):
        current_bandit = self._choose_bandit()
        reward = current_bandit.pull_arm()
        self.logging.record_action(current_bandit, reward)
        
    def actions(self, timesteps):
        for _ in range(timesteps):
            self.action()
    
        return self.logging.all_rewards, self.logging.all_actions, 


class ThomspsonSamplingAgent:
    """
    Thompson Sampling Agent Will come
    """
    pass   

