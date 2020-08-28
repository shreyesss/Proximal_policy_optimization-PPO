import tensorflow as tf
from tensorflow import keras as k
import numpy as np
import threading
import cv2
import random
from matplotlib import pyplot as plt
import time
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.enable_eager_execution()


class PPO_SuperMarioBros:
    def __init__(self, game, level, n_workers, NMaxEp):

        self.game_name = ('SuperMarioBros-' + str(game) + '-' + str(level) + '-v1')
        self.env = JoypadSpace(gym_super_mario_bros.make(self.game_name), SIMPLE_MOVEMENT)
        self.action_space = [0, 1, 2, 3, 4, 5, 6]
        self.n_actions = len(self.action_space)
        self.lr = 0.0001
        self.beta = 0.01
        self.E = 0.2
        self.alpha = 0.9
        self.n_workers = n_workers
        self.NMaxEp = NMaxEp
        self.gamma = 0.99
        self.lambda_ = 0.95
        self.horizon = 128
        self.EPOCHS = 3
        self.HEIGHT = 96
        self.WIDTH = 96
        self.FRAME_SKIP = 4
        self.STACK = 4
        self.EPISODE = 0
        self.avg_score = 0
        self.avg_x_pos = 0
        self.total_frames = 0
        self.max_score = 0
        self.max_x = 0
        self.max_action_repeat = 200
        self.score_tally = []
        self.loss_tally = []
        self.x_pos_tally = []
        self.sticky_action = False
        self.is_normalize_GAE = True
        self.model = self.Model(self.HEIGHT, self.WIDTH, self.STACK)
        self.old_model = self.Model(self.HEIGHT, self.WIDTH, self.STACK)
        self.old_model.set_weights(self.model.get_weights())

    def process_input(self, state):
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        state = cv2.resize(state, (self.HEIGHT, self.WIDTH), interpolation=cv2.INTER_CUBIC)
        state = state / 255.0
        return state

    def policy_loss(self, action, old_prediction):

        E = self.E
        beta = self.beta

        def loss_fn(advantage, policy):

            r = tf.expand_dims(tf.reduce_sum(policy * action,axis=1) / tf.reduce_sum(old_prediction * action,axis=1), axis=1)
            clip_r = tf.clip_by_value(r, clip_value_min=1 - E, clip_value_max=1 + E)
            p1 = r * advantage
            p2 = clip_r * advantage
            policy_loss = -tf.reduce_mean(tf.reduce_sum(tf.minimum(p1, p2), axis=1))
            entropy = -tf.reduce_mean(tf.reduce_sum(policy * tf.log(policy), axis=1))
            loss = policy_loss - beta * entropy
            return loss

        return loss_fn

    def value_loss(self):

        def loss_fn(Q, V):
            val_loss = 0.5 * tf.reduce_mean(tf.square(Q - V))
            return val_loss

        return loss_fn

    def Model(self, height, width, stack):

        state = k.layers.Input(shape=(height, width, stack))
        action = k.layers.Input(shape=(self.n_actions,))
        old_prediction = k.layers.Input(shape=(self.n_actions,))

        conv1 = k.layers.Conv2D(32, kernel_size=(8, 8), strides=(4, 4),
                                kernel_initializer=k.initializers.glorot_normal(),
                                activation=k.activations.relu, padding='valid')(state)

        conv2 = k.layers.Conv2D(64, kernel_size=(4, 4), strides=(2, 2),
                                kernel_initializer=k.initializers.glorot_normal(),
                                activation=k.activations.relu, padding='valid')(conv1)

        conv3 = k.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                                kernel_initializer=k.initializers.glorot_normal(),
                                activation=k.activations.relu, padding='valid')(conv2)

        dense1 = k.layers.Flatten()(conv3)

        dense2 = k.layers.Dense(512, activation=k.activations.relu,
                                kernel_initializer=k.initializers.glorot_normal(),
                                bias_initializer=k.initializers.glorot_normal())(dense1)

        policy = k.layers.Dense(self.n_actions, activation=k.activations.softmax,
                                kernel_initializer=k.initializers.glorot_normal(),
                                bias_initializer=k.initializers.glorot_normal())(dense2)

        value_state = k.layers.Dense(1, activation=None,
                                     kernel_initializer=k.initializers.glorot_normal(),
                                     bias_initializer=k.initializers.glorot_normal())(dense2)

        model = k.Model(inputs=[state, old_prediction, action], outputs=[policy, value_state])

        model.compile(optimizer=k.optimizers.Adam(lr=self.lr, epsilon=float(1e-5)),
                      loss=[self.policy_loss(action, old_prediction), self.value_loss()],
                      loss_weights=[1, 0.5])  
        model.summary()
        # model.load_weights('PPO_mario_2_1(7_actions).h5')

        return model

    def train(self):

        envs = [JoypadSpace(gym_super_mario_bros.make(self.game_name), SIMPLE_MOVEMENT) for i in range(self.n_workers)]
        lock = threading.Lock()
        workers = [threading.Thread(target=self.run_thread, daemon=True, args=(envs[i], i, lock)) for i in
                   range(self.n_workers)]
        for worker in workers:
            worker.start()
            time.sleep(0.1)
        [worker.join() for worker in workers]

        plt.plot(self.x_pos_tally)
        plt.show()
        plt.plot(self.score_tally)
        plt.show()
        print('<<<<< training complete >>>>>')

    def get_action(self, probability):

        action_one_hot = np.zeros([self.n_actions])
        action = np.random.choice(self.action_space, 1, p=probability)[0]
        action_one_hot[self.action_space.index(action)] = 1
        return action_one_hot, action

    def step(self, prev_action, action, env):

        s_rewards, states = [], []
        next_state, reward, done, info = np.empty([96, 96]), 0, False, {}

        for i in range(self.FRAME_SKIP):

            if i == 0:

                if np.random.uniform(0, 1) <= 0.25 and self.sticky_action == True:

                    next_state, reward, done, info = env.step(prev_action)
                    s_rewards.append(reward)
                    states.append(next_state)

                else:

                    next_state, reward, done, info = env.step(action)

                    s_rewards.append(reward)
                    states.append(next_state)

            else:

                next_state, reward, done, info = env.step(action)
                s_rewards.append(reward)
                states.append(next_state)

            if done and  not info['flag_get'] :
                s_rewards.append(-50)
                break

            if done and  info['flag_get'] :
                s_rewards.append(50)
                break

        s_rewards = np.sum(s_rewards) / 20.0

        if len(states) >= 2:
            next_state = np.maximum(states[-1], states[-2])

        return next_state, s_rewards, done, info

    def run_thread(self, env, i, lock):

        while self.EPISODE < self.NMaxEp:

            self.EPISODE += 1
            t_prev_update = 0
            t_step = 0
            score = 0
            done = 0
            x_pos = 0
            level_complete = False
            flag = False
            save = True
            prev_action = 0
            action_repeat = 0
            last_time = time.time()
            state = self.process_input(env.reset())
            state = np.stack([state] * self.STACK, axis=2)
            state_list, reward_list, action_list, done_list, policy_list = [], [], [], [], []

            while not done and not flag:

                lock.acquire()

                t_step += 1
                self.total_frames += 1
                policy = self.get_policy(np.expand_dims(state, axis=0), self.model)[0]

                lock.release()

                if np.isnan(policy).any():
                     print('EPISODE : {} terminated for NaN values...../////'.format(self.EPISODE))
                     save = False
                     break

                else:
                    action_one_hot, action = self.get_action(policy)

                next_state, reward, done, info = self.step(prev_action, action, env)

                next_state = self.process_input(next_state)
                next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, axis=2), axis=2)

                if t_step == 1000:
                    reward = -2.5
                    flag = True

                if action == prev_action:
                    action_repeat += 1
                    if action_repeat >= self.max_action_repeat:
                        reward = -2.5
                        print('EPISODE : ', self.EPISODE, 't_step : ', t_step,
                              'action : ', action, 'x_pos : ', x_pos, 'breaking......time_overflow...../////')
                        flag = True

                state_list.append(state)
                reward_list.append(reward)
                action_list.append(action_one_hot)
                done_list.append(done)
                policy_list.append(policy)
                score += reward
                state = next_state
                prev_action = action

                if info['x_pos'] > x_pos:
                    x_pos = info['x_pos']

                if info['flag_get']:
                    level_complete = True

                if done or t_step - t_prev_update == self.horizon:

                    state_list.append(state)                       # tail_state

                    lock.acquire()
                    if len(action_list):
                        self.update(state_list, action_list, reward_list, done_list)
                    lock.release()
                    state_list, action_list, reward_list, done_list = [], [], [], []
                    t_prev_update = t_step

            if save:

                lock.acquire()

                if self.max_score < score:
                    self.max_score = score

                if self.max_x < x_pos:
                    self.max_x = x_pos

                self.avg_score = 0.99 * self.avg_score + 0.01 * score
                self.avg_x_pos = 0.99 * self.avg_x_pos + 0.01 * x_pos

                self.score_tally.append(self.avg_score)
                self.x_pos_tally.append(self.avg_x_pos)

                if self.EPISODE % 100 == 0:
                    self.model.save_weights('PPO_mario_2_1(7_actions).h5')
                    print('model saved to disc')

                    if len(policy_list) >= 20:
                        print(random.sample(policy_list, 20))

                print('EPISODE : ', self.EPISODE, ' Total_frames : ', self.total_frames, ' avg_score : ', round(self.avg_score, 2),
                      ' score :', round(score, 2), ' t_step : ', t_step,
                      ' avg_x_pos : ', round(self.avg_x_pos, 2), ' x_pos :', x_pos, ' max_score : ',
                      round(self.max_score, 2),' max_x_pos : ', self.max_x,
                      ' time_elapsed :', round(time.time() - last_time, 2), ' level_complete : ', level_complete)

                lock.release()

    def GAE_and_TargetValues(self, V, rewards, done):

        deltas, GAE = [], []
        advantage = 0

        for i in range(len(rewards)):
            delta = rewards[i] + (1-done[i])*self.gamma * V[i + 1][0] - V[i][0]
            deltas.append(delta)
       
        for delta in reversed(deltas):
            advantage = self.gamma * self.lambda_ * advantage + delta
            GAE.append(advantage)

        GAE =np.expand_dims(np.flip(GAE),axis=-1)

        if self.is_normalize_GAE:
            if np.std(GAE) != 0 and len(GAE) > 1:
                GAE = GAE / (np.std(GAE))

        target_values = GAE + V[:-1]

        return GAE, target_values

    def V_predict(self, states):

        dummy = np.zeros([len(states), self.n_actions])
        V = self.model.predict(
            x=[states, dummy, dummy])[1]


        return V

    def get_policy(self, states, model):
        dummy = np.zeros([len(states), self.n_actions])
        policy = np.clip(model.predict(x=[states, dummy, dummy])[0], 0.00001, 0.99999)
        return policy

    def update(self, states, actions, rewards, done):

        global loss_tally
        states = np.stack(states, axis=0)
        actions = np.stack(actions, axis=0)

        V = self.V_predict(states)

        GAE, target_values = self.GAE_and_TargetValues(V, rewards, done)

        old_prediction = self.get_policy(states, self.old_model)

        sess_details = self.model.fit(x=[states[:-1], old_prediction[:-1], actions], y=[GAE, target_values],
                                      epochs=self.EPOCHS, verbose=0)

        old_model_weights = np.array(self.model.get_weights()) * self.alpha + np.array(self.old_model.get_weights()) * (1-self.alpha)

        self.old_model.set_weights(old_model_weights)

        if self.EPISODE % 500 == 0:
            self.print_values(states, rewards, V, target_values, GAE, old_prediction, done, actions)

    def print_values(self, states, rewards, V, target_values, GAE, old_prediction, done, actions):

        #print('states',(states.shape))
        print('rewards', (rewards))
        print('V',(V[:,0]))
        #print('target_values', (target_values[:,0]))
        print('gae',( GAE[:,0]))
        #print('actions :',(actions))
        #print('old',(old_prediction))
        # print('done',(done))

    def test(self):

        EPISODE = 0

        while EPISODE < 20:

            env = self.env
            t_step = 0
            score = 0
            x_pos = 0
            done = False
            flag = False
            action_repeat = 0
            prev_action = random.choice(self.action_space)

            EPISODE += 1
            state = self.process_input(env.reset())
            state = np.stack([state] * self.STACK, axis=2)

            while not done and not flag:

                env.render()
                t_step += 1
                probability = self.get_policy(np.expand_dims(state, axis=0), self.model)[0]
                #print(probability)

                if np.isnan(probability).any() == True:

                    print('terminated for NaN values')
                    break


                else:
                    action_one_hot, action = self.get_action(probability)

                next_state, reward, done, info,  = self.step(prev_action, action, env)

                if t_step == 1000:
                    reward = -2.5
                    flag = True

                if action == prev_action:
                    action_repeat += 1
                    if action_repeat >= self.max_action_repeat:
                        reward = -2.5
                        print('EPISODE : ', EPISODE, 't_step : ', t_step,
                              'action : ', action, 'x_pos : ', x_pos, 'error_breaking......stuck_at_a_place...../////')
                        flag = True

                next_state = self.process_input(next_state)
                next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, axis=2), axis=2)


                score += reward
                state = next_state
                prev_action = action
                time.sleep(0.05)
                x_pos = info['x_pos']
                print(reward)

            print('episode : ', EPISODE, 'score : ', score, 'x_pos  : ', x_pos, 't_step : ', t_step)
            cv2.imshow('terminal state', state)

            if cv2.waitKey(1) and 0xFF == ord('q'):
                cv2.destroyAllWindows()


agent = PPO_SuperMarioBros(game=2, level=1, n_workers=16, NMaxEp=6000)
agent.train()
#agent.test()
















