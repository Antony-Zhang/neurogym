#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import copy
import numpy as np

import neurogym as ngym
from neurogym import spaces


class EEGSameDifferent(ngym.TrialEnv):
    """ Same/Different task 
    
    Derived from Delayed match-to-sample task.

    metadata = {
        'paper_link': '',
        'paper_name': '''''',
        'tags': ['']
    }
    """

    def __init__(self, dt=50, rewards=None, timing=None, sigma=1.0, sigma2 = 0.2,
                 dim_ring=4, cue=None):
        """
        dim_ring: dimension of stimulus space, form the final input with fixation and cue
        """
        super().__init__(dt=dt)
        self.choices = [2, 3]       # matched or not-matched between sample and test 
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise
        self.sigma2 = sigma2 / np.sqrt(self.dt)
        self.dim_ring = dim_ring

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        # Cue
        self.cue_condition = [0, 1, 2]
        if cue:
            self.cue = cue 
        else:
            self.cue = self.rng.choice(self.cue_condition)

        # Timing
        self.timing = {         # default timing
            'cue': 200,
            'fixation': 300,    # should vary among trials
            'sample': 200,
            'delay': 700 + self.cue * 300 - self.dt,
            'end': dt,
            'test': 200,
            'decision': 1000}
        if timing:
            self.timing.update(timing)

        self.seq_len = int(np.sum([*self.timing.values()])/dt) # The vector length of each trial

        self.abort = False

        # Spaces
        name = {'fixation': 0, 'cue': 1, 'stimulus': range(2, dim_ring + 2)}
        self.observation_space = spaces.Box(-10.0, 10.0,#-np.inf, np.inf,        # Obs.是在_new_trial中手动设置的，因而space的范围不需要特别关注
                                            shape=(2 + dim_ring,), 
                                            dtype=np.float32, 
                                            name=name)

        name = {'fixation': 0, 'delay': 1, 'match': 2, 'non-match': 3}
        self.action_space = spaces.Discrete(4, name=name)

        self.stimulus_space = np.arange(dim_ring)

    def _new_trial(self, **kwargs):
        # Trial
        trial = {
            'ground_truth': self.rng.choice(self.choices),
            'sample_theta': self.rng.choice(self.stimulus_space),
        }
        trial.update(kwargs)

        ground_truth = trial['ground_truth']
        sample_theta = trial['sample_theta']

        # sample_sequence
        temp = np.zeros(self.dim_ring)
        temp[sample_theta] = 1.
        sample_sequence = temp      # 其中只有下标为sample_theta的元素为1，其他为0

        # test_sequence, 若ground_truth为2，则与sample_sequence相同
        if ground_truth == 2:   
            test_theta = sample_theta
            test_sequence = sample_sequence
        else:
            new_space = np.delete(self.stimulus_space, sample_theta)    # 删去值为sample_theta的元素
            test_theta = self.rng.choice(new_space)         # 从剩下的元素中随机选择一个
            temp = np.zeros(self.dim_ring)
            temp[test_theta] = 1.
            test_sequence = temp

        trial['test_theta'] = test_theta

        # Periods
        self.add_period(['cue', 'fixation', 'sample', 'delay', 'end', 'test', 'decision'])

        # where表示在Obs.的哪个维度设置值(根据前文的name)，period表示在哪个时段设置值(默认为所有时段)
        self.add_ob(1, where='fixation')
        self.set_ob(0, period='cue', where='fixation')
        self.set_ob(0, period='decision', where='fixation')
        
        self.add_ob(0, where='cue')
        self.set_ob(self.cue+1.5, period='cue', where='cue')
        
        self.add_ob(sample_sequence, period='sample', where='stimulus')
        self.add_ob(test_sequence, period='test', where='stimulus')
        self.add_randn(0, self.sigma, period=['test'], where='stimulus')    # Add random noise
        self.add_randn(0, self.sigma2, period=['delay'], where='stimulus')

        self.set_groundtruth(ground_truth, period='test')
        self.set_groundtruth(1, period='end')

        return trial

    def _step(self, action):
        new_trial = False
        reward = 0

        ob = self.ob_now
        gt = self.gt_now

        if self.in_period('fixation'):
            if action != 0:
                new_trial = self.abort
                reward = self.rewards['abort']
        elif self.in_period('decision'):
            if action != 0:
                new_trial = True
                if action == gt:
                    reward = self.rewards['correct']
                    self.performance = 1
                else:
                    reward = self.rewards['fail']

        return ob, reward, False, {'new_trial': new_trial, 'gt': gt}
    
    # def __deepcopy__(self, memo):
    #     if id(self) in memo:
    #         return memo[id(self)]
    #     print("==self dict==")
    #     print(self.__dict__)
    #     new_env = EEGSameDifferent(**self.__dict__)
    #     # new_env.rng = np.random.RandomState()  # 你可以选择一个种子
    #     new_env.seed()
    #     memo[id(self)] = new_env

    #     return new_env
    