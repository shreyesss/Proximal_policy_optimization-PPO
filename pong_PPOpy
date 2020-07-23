import tensorflow as tf
from tensorflow import keras as k
import numpy as np
import threading
import cv2
import gym
import random
import time


tf.enable_eager_execution()
EPISODE,running_score,G_t_step = 0,0,0

class PPO_breakout:

        def __init__(self, game_name,n_workers, NMaxEp):
            self.game_name = game_name
            self.env = gym.make(self.game_name).env
            self.action_space = [1, 2, 3]
            self.n_actions = len(self.action_space)
            self.lr = 0.0001
            self.beta = 0.01
            self.E = 0.2
            self.alpha = 0.1
            self.n_workers = int(n_workers)
            self.NMaxEp = NMaxEp
            self.gamma = 0.99
            self.lambda_ = 0.95
            self.horizon = 32
            self.EPOCHS = 4
            self.HEIGHT = 80
            self.WIDTH = 80
            self.FRAME_SKIP = 4
            self.STACK = 4
            self.sticky_action = False
            self.is_normalize_GAE = True
            self.model = self.Model(self.HEIGHT, self.WIDTH, self.STACK)
            self.old_model = self.Model(self.HEIGHT, self.WIDTH, self.STACK)
            self.old_model.set_weights(self.model.get_weights())

        def process_input(self, state):

            height = self.HEIGHT
            width = self.WIDTH
            state = state[35:195:2, 0:160:2, :]
            state = cv2.resize(state, (height, width), interpolation=cv2.INTER_CUBIC)
            state = 0.299 * state[:, :, 0] + 0.587 * state[:, :, 1] + 0.114 * state[:, :, 2]
            state = state / 255.0
            return state

        def policy_loss(self, action, old_prediction):

            beta = self.beta
            E = self.E

            def loss_fn(y_true, y_pred):
                new_prediction = tf.clip_by_value(y_pred, 0.00001, 0.99999)
                advantage = y_true
                entropy = -tf.reduce_mean(tf.reduce_sum(y_pred * tf.log(y_pred), axis=1))
                r = tf.exp(
                    tf.log(action * new_prediction + float(1e-10)) - tf.log(action * old_prediction + float(1e-10)))
                clip_r = tf.clip_by_value(r, clip_value_min=1 - E, clip_value_max=1 + E)
                p1 = r * advantage
                p2 = clip_r * advantage
                policy_loss = -tf.reduce_mean(tf.reduce_sum(tf.minimum(p1, p2), axis=1))
                loss = policy_loss - beta * entropy
                return loss

            return loss_fn

        def value_loss(self):

            def loss_fn(y_true, y_pred):
                val_loss = 0.5 * tf.reduce_mean(tf.square(y_true - y_pred))
                return val_loss
            return  loss_fn

        def Model(self, height ,width ,stack ):

            state = k.layers.Input(shape=(height, width, stack))
            action = k.layers.Input(shape=(self.n_actions,))
            old_prediction = k.layers.Input(shape=(self.n_actions,))

            conv1 = k.layers.Conv2D(32, kernel_size=(4, 4), strides=(4, 4),
                                    kernel_initializer=k.initializers.glorot_normal(),
                                    activation=k.activations.relu, padding='valid')(state)

            conv2 = k.layers.Conv2D(16, kernel_size=(4, 4), strides=(2, 2),
                                    kernel_initializer=k.initializers.glorot_normal(),
                                    activation=k.activations.relu, padding='valid')(conv1)

            dense1 = k.layers.Flatten()(conv2)

            dense2 = k.layers.Dense(64, activation=k.activations.relu,
                                    kernel_initializer=k.initializers.glorot_normal(),
                                    bias_initializer=k.initializers.glorot_normal())(dense1)

            policy = k.layers.Dense(self.n_actions, activation=k.activations.softmax,
                                     kernel_initializer=k.initializers.glorot_normal(),
                                     bias_initializer=k.initializers.glorot_normal())(dense2)

            value = k.layers.Dense(1, activation=None,
                                   kernel_initializer=k.initializers.glorot_normal(),
                                   bias_initializer=k.initializers.glorot_normal())(dense2)

            model = k.Model(inputs=[state,old_prediction,action], outputs=[policy, value])

            model.compile(optimizer=k.optimizers.Adam(lr=self.lr),
                          loss=[self.policy_loss(action,old_prediction),self.value_loss()],
                          loss_weights=[1, 0.5])

            model.summary()

            return model



        def train(self):

            envs = [gym.make(self.game_name) for _ in range(self.n_workers)]
            lock = threading.Lock()
            workers = [threading.Thread(target=self.run_thread, daemon=True, args=(envs[i], i, lock)) for i in
                       range(self.n_workers)]
            for worker in workers:
                worker.start()
                time.sleep(0.1)
            [worker.join() for worker in workers]

        def step(self,prev_action,action,env,lives):

                sum_rewards,states = [],[]

                next_state,reward,done,info = np.empty([self.HEIGHT,self.WIDTH]),0,False,{}

                for i in range(self.FRAME_SKIP):

                    if i == 0:

                        if np.random.uniform(0,1) <= 0.25 and self.sticky_action == True:
                            next_state, reward, done, info = env.step(prev_action)
                            sum_rewards.append(reward)
                            states.append(next_state)

                        else:

                            next_state, reward, done, info = env.step(action)
                            sum_rewards.append(reward)
                            states.append(next_state)

                    else:

                        next_state, reward, done, info = env.step(action)
                        sum_rewards.append(reward)
                        states.append(next_state)

                    if info['ale.lives'] - lives == -1:

                         done = True
                         lives -= 1
                         sum_rewards.append(-1.0)
                         break

                sum_rewards = np.sum(sum_rewards)

                if len(states)>=2:
                  next_state = np.maximum(states[-1],states[-2])

                return next_state, sum_rewards, done, info, lives



        def run_thread(self, env, i, lock):

            global EPISODE, running_score, G_t_step

            while EPISODE < self.NMaxEp and running_score < 300:

                t_prev_update = 0
                t_step = 0
                score = 0
                lives = 20
                flag =  False
                prev_action = random.choice(self.action_space)

                state_list, reward_list, action_list, done_list, probability_list = [], [], [], [], []

                EPISODE += 1
                state = self.process_input(env.reset())
                state = np.stack([state] * self.STACK, axis=2)

                while lives > 0 and t_step < 2000:

                    lock.acquire()

                    t_step += 1
                    G_t_step += 1
                    probability = self.get_policy(np.expand_dims(state, axis=0), self.model)[0]

                    if i == 1:
                       env.render()

                    lock.release()


                    if np.isnan(probability).any() == True:

                        print('terminated for NaN values')
                        flag = True
                        break

                    else:
                        action_one_hot, action = self.get_action(probability)


                    next_state, reward, done, info, lives = self.step(prev_action,action,env,lives)


                    next_state = self.process_input(next_state)
                    next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, axis=2), axis=2)

                    if reward == -1:
                        lives -= 1
                        done = True

                    state_list.append(state)
                    reward_list.append(reward)
                    action_list.append(action_one_hot)
                    done_list.append(done)
                    probability_list.append(probability)
                    score += reward
                    state = next_state
                    prev_action = action

                    if t_step - t_prev_update == self.horizon or done == True:
                        state_list.append(state)
                        lock.acquire()
                        if len(action_list):
                            self.update(state_list, action_list, reward_list, done_list)
                        lock.release()
                        state_list, action_list, reward_list, done_list = [], [], [], []
                        t_prev_update = t_step

                lock.acquire()

                if flag != True:

                  running_score = 0.95 * running_score + 0.05 * score
                  print('EPISODE : ', (EPISODE), 'G_tstep : ', (G_t_step), 'running score : ', (running_score),
                      'score :',
                      (score), 't_step : ', (t_step))

                  if EPISODE % 50 == 0:
                    self.model.save_weights('PPO_pong.h5')
                    print('model saved to disc')
                    print(random.sample(probability_list, 20))

                lock.release()

        def GAE_and_TargetValues(self, V, rewards):

            deltas, GAE = [], []
            advantage = 0

            for i in range(len(rewards)):
                delta = rewards[i] + self.gamma * V[i + 1][0] - V[i][0]
                deltas.append(delta)

            for delta in reversed(deltas):
                advantage = self.gamma * self.lambda_ * advantage + delta
                GAE.append(advantage)

            GAE = np.flip(GAE)

            if self.is_normalize_GAE:
                if np.std(GAE) != 0 and len(GAE) > 1:
                    GAE = (GAE - np.mean(GAE)) / (np.std(GAE))

            target_values = GAE + V[:-1]

            return target_values, GAE



        def V_predict(self,states,done):

            V = self.model.predict(
                x=[states, np.zeros([len(states), self.n_actions]), np.zeros([len(states), self.n_actions])])[1]

            if done[-1]:
                V[-1][0] = 0

            return V

        def get_policy(self,states,model):

            dummy = np.zeros([len(states), self.n_actions])
            policy = np.clip(model.predict(x=[states,dummy,dummy])[0],0.00001,0.99999)
            return policy

        def get_action(self, probability):

            action_one_hot = np.zeros([self.n_actions])
            action = np.random.choice(self.action_space, 1, p=probability)[0]
            action_one_hot[self.action_space.index(action)] = 1
            return action_one_hot, action

        def update(self, states, actions, rewards, done ):

            states = np.stack(states,axis=0)
            actions = np.vstack(actions)
            rewards=np.vstack(rewards)

            V = self.V_predict(states,done)

            GAE, target_values = self.GAE_and_TargetValues(V,rewards)

            old_prediction = self.get_policy(states,self.old_model)

            self.model.fit(x=[states[:-1], old_prediction[:-1],actions],y=[GAE,target_values],epochs=self.EPOCHS,verbose=0)

            old_model_weights = np.array(self.model.get_weights())*(1-self.alpha)+ np.array(self.old_model.get_weights())*self.alpha

            self.old_model.set_weights(old_model_weights)

            #self.print_values(states, rewards, V, target_values,GAE, old_prediction)


        def print_values(self,states,rewards,V,target_values,GAE,old_prediction):

            print('states', (states))
            print('rewards', (rewards))
            print('V', (V))
            print('target_values', (target_values))
            print('gae', (GAE))
            print('old', (old_prediction))

        def test(self):

            EPISODE = 0
            #self.model.load_weights('PPO_breakout.h5')

            while EPISODE < 20:

                env = self.env
                t_step = 0
                score = 5
                lives = 20
                prev_action = random.choice(self.action_space)


                EPISODE += 1
                state = self.process_input(env.reset())
                state = np.stack([state] * self.STACK, axis=2)

                while lives > 0 and t_step < 2000:

                    env.render()

                    t_step += 1
                    probability = self.get_policy(np.expand_dims(state, axis=0), self.model)[0]


                    if np.isnan(probability).any() == True:

                        print('terminated for NaN values')
                        flag = True
                        break

                    else:

                        action_one_hot, action = self.get_action(probability)

                    next_state, reward, done, info, lives = self.step(prev_action, action, env, lives)

                    next_state = self.process_input(next_state)
                    next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, axis=2), axis=2)


                    score += reward
                    state = next_state
                    prev_action = action

                print('episode', (EPISODE), 'score', (score))





agent = PPO_breakout(n_workers=8, NMaxEp=20000, game_name='PongNoFrameskip-v4')
agent.train()
#agent.test()




















