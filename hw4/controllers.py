import numpy as np
from cost_functions import trajectory_cost_fn
import time
import logging

def dd(s):
    logging.getLogger("hw4").debug(s)

def di(s):
    logging.getLogger("hw4").info(s)


class Controller():
    def __init__(self):
        pass

    # Get the appropriate action(s) for this state(s)
    def get_action(self, state):
        pass


class RandomController(Controller):
    def __init__(self, env):
        """ YOUR CODE HERE """
        self.env = env

    def get_action(self, state):
        """ YOUR CODE HERE """
        """ Your code should randomly sample an action uniformly from the action space """
        return self.env.action_space.sample()  # pick a random action


class MPCcontroller(Controller):
    """ Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """

    def __init__(self,
                 env,
                 dyn_model,
                 horizon=5,
                 cost_fn=None,
                 num_simulated_paths=10,
                 ):
        self.env = env
        self.dyn_model = dyn_model
        self.horizon = horizon
        self.cost_fn = cost_fn
        self.num_simulated_paths = num_simulated_paths

    def get_action(self, state):
        """ YOUR CODE HERE """
        """ Note: be careful to batch your simulations through the model for speed """
        # state is the initial state
        # we need to generate self.num_simulated_paths trajectories at length self.horizon steps
        curr_state = np.tile(state, (
            self.num_simulated_paths, 1))  # create a batch of start state: [num_simulated_paths,obs_dim]
        states = []
        actions = []
        next_states = []
        for i in range(self.horizon):
            # sample an action per each path
            curr_action = []
            for _ in range(self.num_simulated_paths):
                curr_action.append(self.env.action_space.sample())  # curr action per each path
            curr_action = np.concatenate([curr_action])  # shape : [num_simulated_paths,act_dim]
            next_state = self.dyn_model.predict(curr_state, curr_action)  # shape: [num_simulated_paths,obs_dim]
            # append it to the path data structure
            states.append(curr_state)
            actions.append(curr_action)
            next_states.append(next_state)

            # progress one step
            curr_state = next_state

        # at this point we have the following lists:
        # states = a list of numpy arrays, each is a set of states for a time step t, for num_simulated_paths
        #          so states[t] is numpy array of size [num_simulated_paths,obs_dim]
        # actions = list of numpy array of actions for all paths. actions[t] is of shape [num_simulated_paths,act_dim]
        # next_states = like states but its the state of time t+1. np array shape [num_simulated_paths,obs_dim]

        # we now need to find the cost of each path
        paths_costs = trajectory_cost_fn(self.cost_fn, states, actions, next_states)
        # now we have array of num_simulated_paths cost values. we need to find the argmin and take the corresponding action
        return actions[0][np.argmin(paths_costs)]
