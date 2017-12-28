import numpy as np
import tensorflow as tf
import gym
from dynamics import NNDynamicsModel
from controllers import MPCcontroller, RandomController
from cost_functions import cheetah_cost_fn, trajectory_cost_fn
import time
import logz
import os
import logging
import copy
import matplotlib.pyplot as plt
from cheetah_env import HalfCheetahEnvNew

def dd(s):
    logging.getLogger("hw4").debug(s)

def di(s):
    logging.getLogger("hw4").info(s)




def paths_to_data(paths):
    data=dict()
    data['observations'] = np.concatenate(paths['observations'])     # shape [num_paths*path_length,obs_dim]
    data['actions'] = np.concatenate(paths['actions'])             # shape [num_paths*path_length,act_dim]
    data['next_observations'] = np.concatenate(paths['next_observations'])     # shape [num_paths*path_length,obs_dim]
    data['rewards'] = np.concatenate(paths['rewards'])  # shape [num_paths*path_length,1]
    # note that path_length differs from path to path and always path_length<=horizon
    return data



def sample(env,
           controller,
           num_paths=10,
           horizon=1000,
           render=False,
           verbose=False):
    """
        Write a sampler function which takes in an environment, a controller (either random or the MPC controller),
        and returns rollouts by running on the env.
        Each path can have elements for observations, next_observations, rewards, returns, actions, etc.
    """
    path_cnt = 0 # counts how many paths we have collected
    paths = {'observations':[],'actions':[],'next_observations':[],'rewards':[]}
    dd("sampling {0} trajectories".format(num_paths))
    while path_cnt<num_paths:
        path_obs,path_acs,path_rewards,path_next_obs = [],[],[],[]
        ob = env.reset()
        steps = 0   # count the number of steps in the path

        while True:
            ac = controller.get_action(ob)  # sample an action from the contoller
            next_ob,reward,done,_=env.step(ac)   # progress one step with the environment
            path_obs.append(ob)
            path_acs.append(ac)
            path_rewards.append(reward)
            path_next_obs.append(next_ob)
            ob = next_ob
            steps += 1
            if done or steps > horizon:
                break
        dd("path {0} completed with {1} steps".format(path_cnt,steps))
        paths['observations'].append(path_obs)
        paths['actions'].append(path_acs)
        paths['next_observations'].append(path_next_obs)
        paths['rewards'].append(path_rewards)
        path_cnt += 1
    dd("finished sampling")
    return paths


# Utility to compute cost a path for a given cost function
def path_cost(cost_fn, path):
    return trajectory_cost_fn(cost_fn, path['observations'], path['actions'], path['next_observations'])

def compute_normalization(data):
    """
    Write a function to take in a dataset and compute the means, and stds.
    Return 6 elements: mean of s_t, std of s_t, mean of (s_t+1 - s_t), std of (s_t+1 - s_t), mean of actions, std of actions

    """

    """ YOUR CODE HERE """
    s_t = data['observations']
    a_t = data['actions']
    delta = data['next_observations'] - data['observations']


    mean_obs = np.mean(s_t,axis=0)
    std_obs = np.std(s_t,axis=0)
    mean_deltas = np.mean(delta,axis=0)
    std_deltas = np.std(delta,axis=0)
    mean_action = np.mean(a_t,axis=0)
    std_action = np.std(a_t,axis=0)
    return mean_obs, std_obs, mean_deltas, std_deltas, mean_action, std_action


def plot_comparison(env, dyn_model):
    """
    Write a function to generate plots comparing the behavior of the model predictions for each element of the state to the actual ground truth, using randomly sampled actions.
    """
    """ YOUR CODE HERE """
    pass

def train(env,
         cost_fn,
         logdir=None,
         render=False,
         learning_rate=1e-3,
         onpol_iters=10,
         dynamics_iters=60,
         batch_size=512,
         num_paths_random=10,
         num_paths_onpol=10,
         num_simulated_paths=10000,
         env_horizon=1000,
         mpc_horizon=15,
         n_layers=2,
         size=500,
         activation=tf.nn.relu,
         output_activation=None
         ):

    """

    Arguments:

    onpol_iters                 Number of iterations of onpolicy aggregation for the loop to run.

    dynamics_iters              Number of iterations of training for the dynamics model
    |_                          which happen per iteration of the aggregation loop.

    batch_size                  Batch size for dynamics training.

    num_paths_random            Number of paths/trajectories/rollouts generated
    |                           by a random agent. We use these to train our
    |_                          initial dynamics model.

    num_paths_onpol             Number of paths to collect at each iteration of
    |_                          aggregation, using the Model Predictive Control policy.

    num_simulated_paths         How many fictitious rollouts the MPC policy
    |                           should generate each time it is asked for an
    |_                          action.

    env_horizon                 Number of timesteps in each path.

    mpc_horizon                 The MPC policy generates actions by imagining
    |                           fictitious rollouts, and picking the first action
    |                           of the best fictitious rollout. This argument is
    |                           how many timesteps should be in each fictitious
    |_                          rollout.

    n_layers/size/activations   Neural network architecture arguments.

    """

    logz.configure_output_dir(logdir)

    #========================================================
    #
    # First, we need a lot of data generated by a random
    # agent, with which we'll begin to train our dynamics
    # model.

    """ YOUR CODE HERE """
    random_controller = RandomController(env)       # randomly (unifotm) samples from the action space
    paths = sample(env,random_controller,num_paths_random)
    # paths should be dictionary with the following keys:
    # 'observations' : a list of 'path_observations' (which is by itself a list of observations of a path )
    # 'actions': a list of 'path_actions' (which is by itself a list of actions of a path )
    # 'next_observations': a list of 'path_next_observations' (which is by itself a list of observations of a path )
    # 'rewards': a list of 'path_rewards' (which is by itself a list of the rewards along the path)




    #========================================================
    #
    # The random data will be used to get statistics (mean
    # and std) for the observations, actions, and deltas
    # (where deltas are o_{t+1} - o_t). These will be used
    # for normalizing inputs and denormalizing outputs
    # from the dynamics network.
    #
    data = paths_to_data(paths)   # make paths a dictionary with the above keys, concatenating all paths to one long list per each key
    normalization = compute_normalization(data)



    #========================================================
    #
    # Build dynamics model and MPC controllers.
    #
    sess = tf.Session()

    dyn_model = NNDynamicsModel(env=env,
                                n_layers=n_layers,
                                size=size,
                                activation=activation,
                                output_activation=output_activation,
                                normalization=normalization,
                                batch_size=batch_size,
                                iterations=dynamics_iters,  # number of epochs
                                learning_rate=learning_rate,
                                sess=sess)

    mpc_controller = MPCcontroller(env=env,
                                   dyn_model=dyn_model,
                                   horizon=mpc_horizon,
                                   cost_fn=cost_fn,
                                   num_simulated_paths=num_simulated_paths)


    #========================================================
    #
    # Tensorflow session building.
    #
    sess.__enter__()
    tf.global_variables_initializer().run()

    #========================================================
    #
    # Take multiple iterations of onpolicy aggregation at each iteration refitting the dynamics model to current dataset and then taking onpolicy samples and aggregating to the dataset.
    # Note: You don't need to use a mixing ratio in this assignment for new and old data as described in https://arxiv.org/abs/1708.02596
    #
    dd("Starting iterations in main loop:")
    for itr in range(onpol_iters):
        """ YOUR CODE HERE """
        # shuffle the buffer
        dd("iter {0} - shuffling data".format(itr))
        indxs=np.random.permutation(data['observations'].shape[0])
        data['observations'] = data['observations'][indxs]
        data['actions'] = data['actions'][indxs]
        data['next_observations'] = data['next_observations'][indxs]
        data['rewards'] = data['rewards'][indxs]

        dd("fit dynamic model")
        # fit dynamics model
        dyn_model.fit(data)

        dd("sample on policy trajectories")
        # sample a set of on-policy trajectories using policy derived from mpc controller
        new_paths = sample(env,mpc_controller,num_paths_onpol,env_horizon,render)

        # compute performance metrics
        costs = np.array([trajectory_cost_fn(cost_fn, new_paths['observations'][i], new_paths['actions'][i],
                                     new_paths['next_observations'][i]) for i in range(len(new_paths['observations']))])
        returns = np.array([sum(new_paths['rewards'][i]) for i in range(len(new_paths['rewards']))])
        # costs = np.array([path_cost(cost_fn,path) for path in new_paths])
        # returns = np.array([sum(path['rewards']) for path in new_paths])

        dd("aggregate the data")
        new_data = paths_to_data(new_paths)
        # aggregate the data
        data['observations'] = np.concatenate((data['observations'],new_data['observations']))
        data['actions'] = np.concatenate((data['actions'], new_data['actions']))
        data['next_observations'] = np.concatenate((data['next_observations'], new_data['next_observations']))
        data['rewards'] = np.concatenate((data['next_observations'], new_data['next_observations']))

        dd("Logging:")
        dd("Average Cost {0}, std cost {1}, AvgReturn {2}, std Return {3}".format(np.mean(costs),np.std(costs),np.mean(returns),np.std(returns)))
        # LOGGING
        # Statistics for performance of MPC policy using
        # our learned dynamics model
        logz.log_tabular('Iteration', itr)
        # In terms of cost function which your MPC controller uses to plan
        logz.log_tabular('AverageCost', np.mean(costs))
        logz.log_tabular('StdCost', np.std(costs))
        logz.log_tabular('MinimumCost', np.min(costs))
        logz.log_tabular('MaximumCost', np.max(costs))
        # In terms of true environment reward of your rolled out trajectory using the MPC controller
        logz.log_tabular('AverageReturn', np.mean(returns))
        logz.log_tabular('StdReturn', np.std(returns))
        logz.log_tabular('MinimumReturn', np.min(returns))
        logz.log_tabular('MaximumReturn', np.max(returns))

        logz.dump_tabular()

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v1')
    # Experiment meta-params
    parser.add_argument('--exp_name', type=str, default='mb_mpc')
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--render', action='store_true')
    # Training args
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--onpol_iters', '-n', type=int, default=1)
    parser.add_argument('--dyn_iters', '-nd', type=int, default=60)
    parser.add_argument('--batch_size', '-b', type=int, default=512)
    # Data collection
    parser.add_argument('--random_paths', '-r', type=int, default=10)
    parser.add_argument('--onpol_paths', '-d', type=int, default=10)
    parser.add_argument('--simulated_paths', '-sp', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=int, default=1000)
    # Neural network architecture args
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=500)
    # MPC Controller
    parser.add_argument('--mpc_horizon', '-m', type=int, default=15)
    # Misc
    parser.add_argument('--verbose', '-v', action="store_false")
    args = parser.parse_args()

    # Establish the logger.
    format = "[%(asctime)-15s %(pathname)s:%(lineno)-3s] %(message)s"
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(format))
    logger = logging.getLogger("hw4")
    logger.propagate = False
    logger.addHandler(handler)
    if args.verbose:
        logger.setLevel(logging.DEBUG)



    # Set seed
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # Make data directory if it does not already exist
    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    # Make env
    if args.env_name is "HalfCheetah-v1":
        env = HalfCheetahEnvNew()
        cost_fn = cheetah_cost_fn
    train(env=env,
                 cost_fn=cost_fn,
                 logdir=logdir,
                 render=args.render,
                 learning_rate=args.learning_rate,
                 onpol_iters=args.onpol_iters,
                 dynamics_iters=args.dyn_iters,
                 batch_size=args.batch_size,
                 num_paths_random=args.random_paths,
                 num_paths_onpol=args.onpol_paths,
                 num_simulated_paths=args.simulated_paths,
                 env_horizon=args.ep_len,
                 mpc_horizon=args.mpc_horizon,
                 n_layers = args.n_layers,
                 size=args.size,
                 activation=tf.nn.relu,
                 output_activation=None,
                 )

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    main()
