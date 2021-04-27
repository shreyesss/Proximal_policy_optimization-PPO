import tensorflow as tf
from tensorflow import keras as k
import numpy as np
import threading
import cv2
import gym
import random
import time
import os
import matplotlib.pyplot as plt
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.enable_eager_execution()



class PPO_lunarlander:

    def __init__(self, lr, n_workers, NMaxEp, frequency):
        self.lr = lr

        self.game_name = 'LunarLander-v2'
        self.env = gym.make(self.game_name).env
        self.gamma = 0.99
        self.lambda_ = 0.95
        self.EPISODE = 0
        self.running_score = -20.0
        self.G_t_step = 0
        self.max_score = -100.0
        self.running_score_tally = []
        self.score_tally = []
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        self.n_workers = n_workers
        self.model = self.Model()
        self.old_model = self.Model()
        self.NMaxEp = NMaxEp
        self.frequency = frequency
        self.old_model.set_weights(self.model.get_weights())
        self.dummy_action = np.zeros([1, self.n_actions])
        self.dummy_old_prediction = np.zeros([1, self.n_actions])
        self.is_normalize_GAE = True

    def Model(self):
        state = k.layers.Input(shape=(self.n_states,))
        action = k.layers.Input(shape=(self.n_actions,))
        old_prediction = k.layers.Input(shape=(self.n_actions,))

        dense1 = k.layers.Dense(64, activation=k.activations.relu,
                               kernel_initializer = k.initializers.glorot_normal(),
                               bias_initializer = k.initializers.glorot_normal())(state)

        dense3 = k.layers.Dense(32, activation=k.activations.relu,
                                kernel_initializer=k.initializers.glorot_normal(),
                                bias_initializer=k.initializers.glorot_normal())(dense1)

        actions = k.layers.Dense(self.n_actions, activation=k.activations.softmax,
                             kernel_initializer=k.initializers.glorot_normal(),
                             bias_initializer=k.initializers.glorot_normal())(dense3)

        value = k.layers.Dense(1, activation=None,
                           kernel_initializer=k.initializers.glorot_normal(),
                           bias_initializer=k.initializers.glorot_normal())(dense3)

        model = k.Model(inputs=[state, old_prediction, action], outputs=[actions, value])

        model.compile(optimizer=k.optimizers.Adam(lr=self.lr),
                  loss=[self.policy_loss(action, old_prediction), self.value_loss()],
                  loss_weights=[1, 0.5])

        model.summary()
        return model

    def value_loss(self):

        def loss_fn(y_true, y_pred):
            val_loss = 0.5 * tf.reduce_mean(tf.square(y_true - y_pred))
            return val_loss

        return loss_fn

    def policy_loss(self, action, old_prediction):

        def loss_fn(advantage, policy):

            E = 0.2
            beta = 0.01
            r = tf.expand_dims(tf.reduce_sum(policy * action, axis=1) / tf.reduce_sum(old_prediction * action, axis=1),
                               axis=1)
            clip_r = tf.clip_by_value(r, clip_value_min=1 - E, clip_value_max=1 + E)
            p1 = r * advantage
            p2 = clip_r * advantage
            policy_loss = -tf.reduce_mean(tf.reduce_sum(tf.minimum(p1, p2), axis=1))
            entropy = -tf.reduce_mean(tf.reduce_sum(policy * tf.log(policy), axis=1))
            loss = policy_loss - beta * entropy
            return loss

        return loss_fn

    def train(self):
        envs = [gym.make(self.game_name) for i in range(self.n_workers)]
        lock = threading.Lock()
        workers = [threading.Thread(target=self.run_thread, daemon=True, args=(envs[i], i, lock)) for i in
                   range(self.n_workers)]
        for worker in workers:
            worker.start()
            time.sleep(0.1)
        [worker.join() for worker in workers]

        plt.plot(self.running_score_tally)
        plt.show()

    def update(self, states, actions, rewards, done):
        deltas, GAE, target_values = [], [], []
        advantage = 0

        states = np.stack(states,axis=0)
        actions = np.stack(actions,axis=0)


        V = \
            self.model.predict(
                x=[states, np.zeros([len(states), self.n_actions]), np.zeros([len(states), self.n_actions])])[1]

        if done[-1] == True:
            V[-1][0] = 0

        for i in range(len(rewards)):
            delta = rewards[i] + self.gamma * V[i + 1][0] - V[i][0]
            deltas.append(delta)

        for delta in reversed(deltas):
            advantage = advantage * self.gamma * self.lambda_ + delta
            GAE.append(advantage)

        GAE = np.expand_dims(np.flip(GAE),axis=1)

        if self.is_normalize_GAE:
            if np.std(GAE) != 0 and len(GAE) > 1:
                GAE = GAE / np.std(GAE)

        target_values = GAE + V[:-1]

        old_prediction = self.old_model.predict(
            x=[states, np.zeros([len(states), self.n_actions]), np.zeros([len(states), self.n_actions])])[0]

        sess = self.model.fit(x=[states[:-1], old_prediction[:-1], actions], y=[GAE, target_values], epochs=3, verbose=0,shuffle=False)

        #print(sess.history['loss'])

        new_weights = np.array(self.model.get_weights()) * 0.9 + np.array(self.old_model.get_weights()) * 0.1

        self.old_model.set_weights(new_weights)


    def run_thread(self, env, i, lock):


        while self.EPISODE < self.NMaxEp and self.running_score < 100:

            self.EPISODE += 1
            done = False
            flag = False
            t_step, score, t = 0, 0, 0
            state = env.reset()
            last_time = time.time()
            state_list, reward_list, action_list, done_list, probability_list = [], [], [], [], []
            while not done and t_step < 600:
                lock.acquire()
                t_step += 1
                self.G_t_step += 1
                action = np.zeros([self.n_actions])
                probability = np.clip(
                    self.model.predict([np.expand_dims(state, axis=0), self.dummy_old_prediction, self.dummy_action])[
                        0][0],
                    0.00001, 0.99999)

                if np.isnan(probability).any():
                    print('EPISODE : {} terminated for NaN values...../////'.format(self.EPISODE))

                    flag = True
                    break

                else:
                    a = np.random.choice(self.n_actions, 1, p=probability)[0]
                    action[a] = 1

                lock.release()
                #if i == 1:
                 #   env.render()
                next_state, reward, done, info = env.step(a)
                next_state = next_state
                reward = reward / 10.0
                if done:
                    reward = -1

                state_list.append(state)
                reward_list.append(reward)
                action_list.append(action)
                done_list.append(done)
                probability_list.append(probability)
                state = next_state
                score += reward

                if t_step - t == self.frequency or done == True:
                    state_list.append(state)
                    lock.acquire()
                    self.update(state_list, action_list, reward_list, done_list)
                    lock.release()
                    state_list, action_list, reward_list, done_list = [], [], [], []
                    t = t_step

            if not flag:
                lock.acquire()

                if score > self.max_score:
                    self.max_score = score
                self.running_score = 0.99 * self.running_score + 0.01 * score
                self.running_score_tally.append(self.running_score)
                self.score_tally.append(score)
                print('EPISODE : ', self.EPISODE, ' total_frames : ', self.G_t_step, ' avg_score : ', round(self.running_score, 2),
                      ' ep_score :', round(score, 2), ' t_step : ', t_step, ' max_score : ',
                      round(self.max_score, 2),
                      ' time_elapsed :', round(time.time() - last_time, 2) )
                if self.EPISODE % 50 == 0:
                    #self.model.save_weights('PPO_LunarLander.h5')
                    #print('model saved to disc')
                    print(random.sample(probability_list, 20))
                lock.release()


    def test(self):
        EPISODE = 0
        self.model.load_weights('PPO_LunarLander.h5')
        env = self.env

        while EPISODE < 20:

            t_step = 0
            score = 0
            done = False
            EPISODE += 1
            state = env.reset()

            while not done and t_step < 400:
                env.render()

                t_step += 1
                probability = np.clip(self.model.predict(
                    [np.expand_dims(state, axis=0), self.dummy_old_prediction, self.dummy_action])[0][0], 0.00001,
                                      0.99999)

                action = np.random.choice(self.n_actions, 1, p=probability)[0]

                next_state, reward, done, info = env.step(action)

                #states.append(next_state)
                score += reward
                state = next_state


            print('episode : ', (EPISODE), 'score : ', (score))



agent1 = PPO_lunarlander(lr=0.0001, n_workers=16, NMaxEp=3000, frequency=64)
#agent1.train()
agent1.test()

























