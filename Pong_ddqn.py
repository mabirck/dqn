# coding:utf-8
import argparse
import os
import gym
import random
import json
import numpy as np
import tensorflow as tf
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, Dense, Activation

ENV_NAME = 'Pong-v0'  # Environment name #Default
FRAME_WIDTH = 84  # Resized frame width
FRAME_HEIGHT = 84  # Resized frame height
TOTAL_STEPS = 10000000
NUM_EPISODES = 30000  # Number of episodes the agent plays
EPOCH_LENGTH = 50000  # Steps of one epoch, perform 1 test per epoch
TEST_LENGTH = 12000
STATE_LENGTH = 4  # Number of most recent frames to produce the input to the network
GAMMA = 0.99  # Discount factor
EXPLORATION_STEPS = 1000000  # Number of steps over which the initial value of epsilon is linearly annealed to its final value
INITIAL_EPSILON = 1.0  # Initial value of epsilon in epsilon-greedy
FINAL_EPSILON = 0.1  # Final value of epsilon in epsilon-greedy
INITIAL_REPLAY_SIZE = 2000  # Number of steps to populate the replay memory before training starts
NUM_REPLAY_MEMORY = 400000  # Number of replay memory the agent uses for training
BATCH_SIZE = 32  # Mini batch size
TARGET_UPDATE_INTERVAL = 10000  # The frequency with which the target network is updated
TRAIN_INTERVAL = 4  # The agent selects 4 actions between successive updates
LEARNING_RATE = 0.00025  # Learning rate used by RMSProp
MOMENTUM = 0.95  # Momentum used by RMSProp
MIN_GRAD = 0.001  # Constant added to the squared gradient in the denominator of the RMSProp update
SAVE_INTERVAL = 300000  # The frequency with which the network is saved
NO_OP_STEPS = 30  # Maximum number of "do nothing" actions to be performed by the agent at the start of an episode
LOAD_NETWORK = True
SAVE_NETWORK_PATH = 'saved_networks/' + ENV_NAME
SAVE_SUMMARY_PATH = 'summary/' + ENV_NAME
NUM_EPISODES_AT_TEST = 30  # Number of episodes the agent plays at test time


class Agent():
    def __init__(self, num_actions, args):
        self.num_actions = num_actions
        self.args = args
        self.epsilon = INITIAL_EPSILON
        self.epsilon_step = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS
        self.t = 0
        # Parameters used for summary
        self.total_reward = 0
        self.total_q_max = 0
        self.total_loss = 0
        self.duration = 0
        self.episode = 0

        # Create replay memory
        self.replay_memory = deque()

        # Create q network
        self.s, self.q_values, q_network = self.build_network()
        q_network_weights = q_network.trainable_weights

        # Create target network
        self.st, self.target_q_values, target_network = self.build_network()
        target_network_weights = target_network.trainable_weights

        # Define target network update operation
        self.update_target_network = [target_network_weights[i].assign(q_network_weights[i]) for i in range(len(target_network_weights))]

        # Define loss and gradient update operation
        self.a, self.y, self.loss, self.grads_update = self.build_training_op(q_network_weights)

        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver(q_network_weights)
        #self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        #self.summary_writer = tf.summary.FileWriter(SAVE_SUMMARY_PATH, self.sess.graph)

        if not os.path.exists(SAVE_NETWORK_PATH):
            os.makedirs(SAVE_NETWORK_PATH)

        self.sess.run(tf.global_variables_initializer())

        # Load network
        if LOAD_NETWORK:
            self.load_network()

        # Initialize target network
        self.sess.run(self.update_target_network)

    def build_network(self):

        print("Now we build the model")
        model = Sequential()
        model.add(Convolution2D(32, (8,8), strides=(4, 4), padding='same',input_shape=(FRAME_WIDTH, FRAME_HEIGHT, STATE_LENGTH)))  #80*80*4
        model.add(Activation('relu'))
        model.add(Convolution2D(64, (4, 4), strides=(2, 2), padding='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, (3, 3), strides=(1, 1), padding='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.num_actions))

        s = tf.placeholder(tf.float32, [None, FRAME_WIDTH, FRAME_HEIGHT ,STATE_LENGTH])
        q_values = model(s)

        return s, q_values, model

    def build_training_op(self, q_network_weights):
        a = tf.placeholder(tf.int64, [None])
        y = tf.placeholder(tf.float32, [None])

        # Convert action to one hot vector
        a_one_hot = tf.one_hot(a, self.num_actions, 1.0, 0.0)
        q_value = tf.reduce_sum(tf.multiply(self.q_values, a_one_hot), reduction_indices=1)

        # Clip the error, the loss is quadratic when the error is in (-1, 1), and linear outside of that region
        error = tf.abs(y - q_value)
        quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, momentum=MOMENTUM, epsilon=MIN_GRAD)
        grads_update = optimizer.minimize(loss, var_list=q_network_weights)

        return a, y, loss, grads_update

    def get_initial_state(self, observation, last_observation):
        processed_observation = np.maximum(observation, last_observation)
        processed_observation = np.uint8(resize(rgb2gray(processed_observation), (FRAME_WIDTH, FRAME_HEIGHT), mode='reflect') * 255)
        state = [processed_observation for _ in range(STATE_LENGTH)]
        return np.stack(state, axis=2)

    def get_action(self, state):
        if self.epsilon >= random.random() or self.t < INITIAL_REPLAY_SIZE:
            action = random.randrange(self.num_actions)
        else:
            action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))

        # Anneal epsilon linearly over time
        if self.epsilon > FINAL_EPSILON and self.t >= INITIAL_REPLAY_SIZE:
            self.epsilon -= self.epsilon_step

        return action

    def run(self, state, action, reward, terminal, observation):

        observation = np.reshape(observation, (FRAME_WIDTH, FRAME_HEIGHT, 1))


        next_state = np.append(state[:, :, 1:], observation, axis=2)

        # Clip all positive rewards at 1 and all negative rewards at -1, leaving 0 rewards unchanged
        reward = np.clip(reward, -1, 1)

        # Store transition in replay memory
        self.replay_memory.append((state, action, reward, next_state, terminal))
        if len(self.replay_memory) > NUM_REPLAY_MEMORY:
            self.replay_memory.popleft()

        if self.t >= INITIAL_REPLAY_SIZE:
            # Train network
            if self.t % TRAIN_INTERVAL == 0:
                self.train_network()

            # Update target network
            if self.t % TARGET_UPDATE_INTERVAL == 0:
                self.sess.run(self.update_target_network)

            # Save network
            if self.t % SAVE_INTERVAL == 0:
                save_path = self.saver.save(self.sess, SAVE_NETWORK_PATH + '/' + ENV_NAME, global_step=self.t)
                print('Successfully saved: ' + save_path)

        self.total_reward += reward
        self.total_q_max += np.max(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))
        self.duration += 1

        if self.t % EPOCH_LENGTH == 0 and self.t != 0 and self.t > INITIAL_REPLAY_SIZE:
            # Write summary
                
            total_reward, avg_reward, total_q, avg_q =  self.test()

            if not (os.path.isfile('logs/'+self.args['env']+'.json')):
                with open('logs/'+self.args['env']+'.json','w') as f:
                    d = dict()
    
                    d['total_reward'] = [total_reward]
                    d['avg_reward'] = [avg_reward]
                    d['total_q'] = [total_q]
                    d['avg_q'] = [avg_q]
                    json.dump(d, f)    
            else:
                with open('logs/'+self.args['env']+'.json','r') as f:
                    d = json.load(f)
                    d['total_reward'].append(total_reward)
                    d['avg_reward'].append(avg_reward)
                    d['total_q'].append(total_q)
                    d['avg_q'].append(avg_q)
                os.remove('logs/'+self.args['env']+'.json')

                with open('logs/'+self.args['env']+'.json','w') as f:
                    json.dump(d, f)
            
                #stats = [self.total_reward, self.total_q_max / float(self.duration),
                #        self.duration, self.total_loss / (float(self.duration) / float(TRAIN_INTERVAL))]
                #for i in range(len(stats)):
                #    self.sess.run(self.update_ops[i], feed_dict={
                #        self.summary_placeholders[i]: float(stats[i])
                #    })
                #summary_str = self.sess.run(self.summary_op)
                #self.summary_writer.add_summary(summary_str, self.episode + 1)

            # Debug
            if self.t < INITIAL_REPLAY_SIZE:
                mode = 'random'
            elif INITIAL_REPLAY_SIZE <= self.t < INITIAL_REPLAY_SIZE + EXPLORATION_STEPS:
                mode = 'explore'
            else:
                mode = 'exploit'
            print self.epsilon
            print('EPOCH:',self.t/EPOCH_LENGTH,  '/ EPSILON:', self.epsilon, '/ TOTAL_REWARD:', total_reward, \
            '/ AVG_REWARD:', avg_reward, '/ TOTAL_Q:', total_q, '/ AVG_MAX_Q:', avg_q)

        if terminal:
           # Debug
           if self.t < INITIAL_REPLAY_SIZE:
              mode = 'random'
           elif INITIAL_REPLAY_SIZE <= self.t < INITIAL_REPLAY_SIZE + EXPLORATION_STEPS:
              mode = 'explore'
           else:
              mode = 'exploit'
           print('EPISODE: {0:6d} / TIMESTEP: {1:8d} / DURATION: {2:5d} / EPSILON: {3:.5f} / TOTAL_REWARD: {4:3.0f} / AVG_MAX_Q: {5:2.4f} / AVG_LOSS: {6:.5f} / MODE: {7}'.format(
           self.episode + 1, self.t, self.duration, self.epsilon,
           self.total_reward, self.total_q_max / float(self.duration),
           self.total_loss / (float(self.duration) / float(TRAIN_INTERVAL)), mode))

           self.total_reward = 0
           self.total_q_max = 0
           self.total_loss = 0
           self.duration = 0
           self.episode += 1

        self.t += 1

        return next_state

    def train_network(self):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminal_batch = []
        y_batch = []

        # Sample random minibatch of transition from replay memory
        minibatch = random.sample(self.replay_memory, BATCH_SIZE)
        for data in minibatch:
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(data[3])
            terminal_batch.append(data[4])

        # Convert True to 1, False to 0
        terminal_batch = np.array(terminal_batch) + 0

        next_action_batch = np.argmax(self.q_values.eval(feed_dict={self.s: np.array(next_state_batch) / 255.0}), axis=1)
        target_q_values_batch = self.target_q_values.eval(feed_dict={self.st:np.array (next_state_batch) / 255.0})
        for i in xrange(len(minibatch)):
            y_batch.append(reward_batch[i] + (1 - terminal_batch[i]) * GAMMA * target_q_values_batch[i][next_action_batch[i]])

        loss, _ = self.sess.run([self.loss, self.grads_update], feed_dict={
            self.s: np.float32(np.array(state_batch) / 255.0),
            self.a: action_batch,
            self.y: y_batch
        })

        self.total_loss += loss

    def load_network(self):
        checkpoint = tf.train.get_checkpoint_state(self.args['load'])
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print('Successfully loaded: ' + checkpoint.model_checkpoint_path)
        else:
            print('Training new network...')

    def get_action_at_test(self, state):
        if random.random() <= 0.05:
            action, max_q = random.randrange(self.num_actions), None
        else:
            values = self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]})
            action, max_q = np.argmax(values), np.max(values)

        self.t += 1

        return action, max_q
    def test(self):
        # env.monitor.start(ENV_NAME + '-test')
        ENV_NAME = self.args['env'] # Atari game to be played!
        
        env = gym.make(ENV_NAME)
        
        test_ep = 0 
        total_reward = 0
        total_q = 0

        cont_q = 0
        cont_ep = 0
        
        while test_ep < TEST_LENGTH:
            ep_reward = 0
            terminal = False
            observation = env.reset()
            for _ in range(random.randint(1, NO_OP_STEPS)):
                last_observation = observation
                observation, _, _, _ = env.step(0)  # Do nothing
            state = self.get_initial_state(observation, last_observation)
            while not terminal:
                last_observation = observation
                action, max_q = self.get_action_at_test(state)

                if max_q != None:
                    total_q += max_q
                    cont_q += 1
            
                observation, reward, terminal, _ = env.step(action)
                ep_reward+=reward

                if (self.args['render']):
                   env.render()

                processed_observation = preprocess(observation, last_observation)
                #print state.shape, processed_observation.shape
                state = np.append(state[:, :, 1:], processed_observation, axis=2)

                if(test_ep>TEST_LENGTH):
                    break
                
                test_ep+=1
                    
            total_reward += ep_reward
            cont_ep += 1
            
        # env.monitor.close()
        return total_reward, total_reward/cont_ep, total_q, total_q/cont_q #Returns totRew, avgRew, totQ, avgQ
        

def preprocess(observation, last_observation):
    processed_observation = np.maximum(observation, last_observation)
    processed_observation = np.uint8(resize(rgb2gray(processed_observation), (FRAME_WIDTH, FRAME_HEIGHT), mode='reflect') * 255)
    return np.reshape(processed_observation,  (FRAME_WIDTH, FRAME_HEIGHT, 1))


def main():

    parser = argparse.ArgumentParser(description='Train agent to play Atari games from raw input')
    parser.add_argument('-e', '--env', help='Atari game to play!', required=False, default='Pong-v0')
    parser.add_argument('-m', '--mode', help='Train / Test', required=True, default='Train')
    parser.add_argument('-r', '--render', help='Render the game', required=False, default=False)
    parser.add_argument('-l', '--load', help='Load weights the game', required=False, default=SAVE_NETWORK_PATH)
    args = vars(parser.parse_args())

    ENV_NAME = args['env'] # Atari game to be played!
    TRAIN = True if args['mode'] == 'Train' else False # Select either to train or not

    
    env = gym.make(ENV_NAME)
    agent = Agent(num_actions=env.action_space.n, args = args)

    ep = 0
    
    if TRAIN:  # Train mode
        while  (ep <= NUM_EPISODES):
            terminal = False
            observation = env.reset()
            for _ in range(random.randint(1, NO_OP_STEPS)):
                last_observation = observation
                observation, _, _, _ = env.step(0)  # Do nothing
            state = agent.get_initial_state(observation, last_observation)
            while not terminal:
                last_observation = observation
                action = agent.get_action(state)
                observation, reward, terminal, _ = env.step(action) 
                # env.render()
                processed_observation = preprocess(observation, last_observation)
                state = agent.run(state, action, reward, terminal, processed_observation)

                if agent.t > TOTAL_STEPS:
                    ep = NUM_EPISODES + 1
                    break
            ep+=1 #Increment episode counter  
                
    else:  # Test mode

        agent.test()

        # env.monitor.start(ENV_NAME + '-test')
        #for _ in range(NUM_EPISODES_AT_TEST):
        #    terminal = False
        #    observation = env.reset()
        #    for _ in range(random.randint(1, NO_OP_STEPS)):
        #        last_observation = observation
        #        observation, _, _, _ = env.step(0)  # Do nothing
        #    state = agent.get_initial_state(observation, last_observation)
        #    while not terminal:
        #        last_observation = observation
        #        action = agent.get_action_at_test(state)
        #        observation, _, terminal, _ = env.step(action)
        #        env.render()
        #        processed_observation = preprocess(observation, last_observation)
        #        state = np.append(state[1:, :, :], processed_observation, axis=0)
        # env.monitor.close()


if __name__ == '__main__':
    main()
