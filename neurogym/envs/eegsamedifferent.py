#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import neurogym as ngym
from neurogym import spaces


class EEGSameDifferent(ngym.TrialEnv):
    """Delayed match-to-sample task.

    A sample stimulus is shown during the sample period. The stimulus is
    characterized by a one-dimensional variable, such as its orientation
    between 0 and 360 degree. After a delay period, a test stimulus is
    shown. The agent needs to determine whether the sample and the test
    stimuli are equal, and report that decision during the decision period.
    
    metadata = {
        'paper_link': '',
        'paper_name': '''Neural Mechanisms of Visual Working Memory in 
        Prefrontal Cortex of the Macaque''',
        'tags': ['perceptual', 'working memory', 'two-alternative',
                 'supervised']
    }
    """

    def __init__(self, dt=50, rewards=None, timing=None, sigma=1.0, sigma2 = 0.2,
                 dim_ring=4, cue = None):
        super().__init__(dt=dt)
        self.choices = [2, 3]
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise
        self.sigma2 = sigma2 / np.sqrt(self.dt)
        self.dim_ring = dim_ring
        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.cue_condition = [0,1,2]

        if cue:
            self.cue = cue - 1
        else:
            self.cue = self.rng.choice(self.cue_condition)

        self.timing = {
            'cue': 200,
            'fixation': 300,
            'sample': 200,
            'delay': 700 + self.cue  * 300 - self.dt,
            'end': dt,
            'test': 200,
            'decision': 1000}

        if timing:
            self.timing.update(timing)

        self.abort = False

        name = {'fixation': 0, 'cue': 1, 'stimulus': range(2, dim_ring + 2)}
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(2 + dim_ring,), dtype=np.float32, name=name)

        name = {'fixation': 0, 'delay':3, 'match': 1, 'non-match': 2}
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

        temp = np.zeros(self.dim_ring)
        temp[sample_theta] = 1.
        sample_sequence = temp

        if ground_truth == 2:
            test_sequence = sample_sequence
            trial['test_theta'] = sample_theta
        else:
            new_space = np.delete(self.stimulus_space,sample_theta)
            test_theta = self.rng.choice(new_space)
            temp = np.zeros(self.dim_ring)
            temp[test_theta] = 1.
            test_sequence = temp
            trial['test_theta'] = test_theta

        # Periods
        self.add_period(['cue', 'fixation', 'sample', 'delay', 'end', 'test', 'decision'])

        self.add_ob(1, where='fixation')
        self.set_ob(0, 'cue', where='fixation')
        self.set_ob(0, 'decision', where='fixation')
        
        self.add_ob(0,  where='cue')
        self.set_ob(self.cue+1.5, 'cue', where='cue')
        
        self.add_ob(sample_sequence, 'sample', where='stimulus')
        self.add_ob(test_sequence, 'test', where='stimulus')
        self.add_randn(0, self.sigma, ['test'], where='stimulus')
        self.add_randn(0, self.sigma2, ['delay'], where='stimulus')

        self.set_groundtruth(ground_truth, 'test')
        self.set_groundtruth(1, 'end')

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