import logging
import tensorflow as tf
import numpy as np

def dd(s):
    logging.getLogger("hw4").debug(s)

def di(s):
    logging.getLogger("hw4").info(s)


# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder, 
              output_size,
              scope, 
              n_layers=2, 
              size=500, 
              activation=tf.nn.relu,    # following recommendations in the paper
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out

class NNDynamicsModel():
    def __init__(self, 
                 env, 
                 n_layers,
                 size, 
                 activation, 
                 output_activation, 
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        """ YOUR CODE HERE """
        """ Note: Be careful about normalization """
        self.env = env
        self.normalization = normalization  # will be used to normalize the inputs before feeding to the network
        self.batch_size = batch_size
        self.iterations = iterations    # number of iterations to fit the dynamic model in each call
        self.sess = sess

        # prepare the place holders
        # network input: [norm_state,norm_action] (concatenation)
        # network output: norm_dstate
        obs_dim = env.observation_space.shape[0]     # dimensionality of state space
        ac_dim = env.action_space.shape[0]      # dimensionality of action space
        self.obs_ph = tf.placeholder(dtype=tf.float32,shape=[None,obs_dim])
        self.act_ph = tf.placeholder(dtype=tf.float32,shape=[None,ac_dim])
        self.next_obs_ph = tf.placeholder(dtype=tf.float32,shape=[None,obs_dim])
        # normalize the inputs
        eps=1e-9
        mean_obs, std_obs, mean_deltas, std_deltas, mean_action, std_action = self.normalization
        # todo: should I do the following lines with numpy or with tensorflow ops ?
        # norm_obs = tf.div(tf.subtract(self.obs_ph,mean_obs),tf.add(std_obs,eps))
        # norm_act = tf.div(tf.subtract(self.act_ph, mean_action), tf.add(std_action, eps))

        norm_obs = (self.obs_ph-mean_obs)/(std_obs+eps)
        norm_act = (self.act_ph-mean_action)/(std_action+eps)

        # concat the observations and actions to feed to the mlp
        norm_obs_act = tf.concat([norm_obs,norm_act],axis=1,name='norm_obs_act')
        # build the mlp model
        self.pred_norm_delta_obs = build_mlp(norm_obs_act,obs_dim,'dynamics', n_layers, size, activation, output_activation)

        # define the target for the prediction
        delta_obs = self.next_obs_ph-self.obs_ph
        # norm_delta_obs = tf.div(tf.subtract(delta_obs, mean_deltas), tf.add(std_deltas, eps))
        norm_delta_obs = (delta_obs-mean_deltas)/(std_deltas+ eps)

        # define the cost function and optimization
        self.loss = tf.losses.mean_squared_error(predictions=self.pred_norm_delta_obs,labels=norm_delta_obs)
        self.update_op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.loss)

        # and the output of the model is the diff_state
        # self.pred_delta_obs = tf.add(tf.multiply(self.pred_norm_delta_obs,tf.add(std_deltas,eps)),mean_deltas)
        self.pred_delta_obs = self.pred_norm_delta_obs*std_deltas+mean_deltas

        # denormalize output and add the delta to get the next state
        # self.pred_next_obs = tf.add(self.obs_ph,self.pred_delta_obs)
        self.pred_next_obs = self.obs_ph+self.pred_delta_obs


    def fit(self, data):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)

        My Note:
        Each call to fit will run for self.iterations epochs on the provided dataset.
        each iteration is sampling a batch_size samples from the dataset

        """

        """YOUR CODE HERE """
        # the code should include one 'update' operation
        denorm_obs = data['observations']
        denorm_act = data['actions']
        denorm_next_obs = data['next_observations']
        dataset_size = denorm_obs.shape[0]
        # get groups of indexes to work on in each mini batch. since its a regression problem, there's no need to randomly sample
        # but rather sequentially scan the buffer (anyway, the data is shuffled externally)
        minibatch_start_indxs = np.arange(0,dataset_size,self.batch_size)
        minibatch_end_indxs = minibatch_start_indxs+self.batch_size

        losses=[]
        for t in range(self.iterations):
            # get batch_size samples from the dataset
            for i, (start,end) in enumerate(zip(minibatch_start_indxs,minibatch_end_indxs)):
                # work on the batch
                _,loss_value=self.sess.run([self.update_op, self.loss],
                                           feed_dict={self.obs_ph:denorm_obs[start:end],
                                                      self.act_ph:denorm_act[start:end],
                                                      self.next_obs_ph:denorm_next_obs[start:end]})
                losses.append(loss_value)
        return losses

    def predict(self, states, actions):
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model """
        """ YOUR CODE HERE """
        pred_next_state = self.sess.run([self.pred_next_obs],
                                        feed_dict={self.obs_ph:states,
                                                   self.act_ph:actions})
        return pred_next_state[0]

