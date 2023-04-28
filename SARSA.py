import random
import numpy as np
import math
import matplotlib.pyplot as plt

from Environment import Environment
from Parameters import *

# set the constant random seed
np.random.seed(1)


class SARSA(object):
    def __init__(self, env, learning_rate, gamma, epsilon):
        self.env = env
        self.n_obs = self.env.n_states
        self.n_a = self.env.n_actions
        self.success = []
        self.fail = []

        # Hyper parameters
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

        self.Q = self.create_Q_table()

    # Create a Q table
    def create_Q_table(self):
        self.Q = {}
        for s in range(self.n_obs):
            for a in range(self.n_a):
                self.Q[(s, a)] = 0.0

        return self.Q

    # Choose actions based on epsilon greedy policy
    def epsilon_greedy(self, state):
        if np.random.uniform(0, 1) < self.epsilon*0.75:
            action = np.random.randint(0, 3) #generate a random integer 0<= x <=3
            while action == max(list(range(self.n_a)), key=lambda x: self.Q[(state, x)]):
                action = np.random.randint(0, 3)
            return action

        else:
            return max(list(range(self.n_a)), key=lambda x: self.Q[(state, x)])

    # Choose actions based on optimal greedy policy
    def optimal_policy(self, observation):
        return max(list(range(self.n_a)), key=lambda x: self.Q[(observation, x)])

    # Learning and updating the Q table using the SARSA update rules as :
    # Q(s,a) = Q(s,a) + alpha * (r + gamma * Q(s',a') - Q(s,a))
    def train(self, num_epoch):
        steps = []
        all_costs = []
        accuracy = []
        Reward_list = []
        Q_value = {}

        goal_count = 0
        rewards = 0
        positive_count = 0
        negative_count = 0

        # for each episode
        for i in range(num_epoch):
            observation = self.env.reset()

            action = self.epsilon_greedy(observation)

            step = 0
            cost = 0

            # Calculate the accuracy rate for every 50 steps
            if i != 0 and i % 50 == 0:
                goal_count = goal_count / 50
                accuracy += [goal_count]
                goal_count = 0

            if i != 0 and i % 1000 == 0:
                Q_value[i] = []
                for j in range(self.env.n_actions):
                    Q_value[i].append(self.Q[((self.env.n_states - 2), j)])

            while True:

                next_observation, reward, done, info = self.env.step(action)

                next_action = self.epsilon_greedy(next_observation)

                # Calculate the Q value of the state-action pair
                # SARSA specifies unique next action based on epsilon greedy policy(different from q-learning)
                k = step
                if step == 0: k =1
                self.Q[(observation, action)] += (self.lr/k) * (
                            reward + self.gamma * self.Q[(next_observation, next_action)] - self.Q[
                        (observation, action)])

                cost += self.Q[(observation, action)]

                step += 1

                if done:
                    if reward > 0:
                        positive_count += 1
                        self.success += [i]
                    else:
                        negative_count += 1
                        self.fail += [i]

                    steps += [step]
                    all_costs += [cost]
                    #4*4
                    # if reward == 1:
                    #10*10
                    if reward == 5:
                        goal_count += 1

                    rewards += reward
                    Reward_list += [rewards / (i + 1)]

                    break

                # Update observation and action
                observation = next_observation
                action = next_action

            print("episodes:{}".format(i))

        # See if converge
        print("Q_value:{}".format(Q_value))

        # Record the data to the list
        all_cost_bar = [positive_count, negative_count]

        self.plot_results(steps, all_costs, accuracy, all_cost_bar, Reward_list)

        return self.Q, steps, all_costs, accuracy, all_cost_bar, Reward_list, self.success, self.fail

    # Plotting the training results
    def plot_results(self, steps, cost, accuracy, all_cost_bar, Reward_list):
        
        plt.figure()
        plt.plot(np.arange(len(steps)), steps, color='rosybrown')
        plt.title('Episode via steps')
        plt.xlabel('Episode')
        plt.ylabel('Steps')

        
        plt.figure()
        plt.plot(np.arange(len(cost)), cost, color='rosybrown')
        plt.title('Episode via cost')
        plt.xlabel('Episode')
        plt.ylabel('Cost')

        
        plt.figure()
        plt.plot(np.arange(len(accuracy)), accuracy, color='rosybrown')
        plt.title('Episode via Accuracy')
        plt.xlabel('Episode')
        plt.ylabel('Accuracy')

        
        plt.figure()
        list = ['Success', 'Fail']
        color_list = ['steelblue', 'rosybrown']
        plt.bar(np.arange(len(all_cost_bar)), all_cost_bar, tick_label=list, color=color_list)
        plt.title('Bar/Success and Fail')
        plt.ylabel('Number')

        plt.figure()
        plt.plot(np.arange(len(Reward_list)), Reward_list, color='rosybrown')
        plt.title('Episode via Average rewards')
        plt.xlabel('Episode')
        plt.ylabel('Average rewards')

        # Showing the plots
        plt.show()

    # Test after training
    def test(self):
    
        num_test = 100

        f = {}

        num_find_goal = 0
        reward_list = []
        steps_list = []

        # run 100 episode to test the correctness of the method
        for i in range(num_test):
            observation = self.env.reset()

            for j in range(NUM_STEPS):
                # # render the environment
                # env.render()

                action = self.optimal_policy(observation)

                next_observation, reward, done, info = self.env.step(action)

                y = int(math.floor(next_observation / GRID_SIZE)) * PIXELS
                x = int(next_observation % GRID_SIZE) * PIXELS
                f[j] = [x, y]

                if done:
                    if reward == 1:
                        num_find_goal += 1
                    r = reward
                    step = j + 1
                    reward_list += [r]
                    steps_list += [step]

                    break

                observation = next_observation

        # Print final route
        self.env.f = f
        self.env.final()

        print("Accuracy:{}".format(num_find_goal / num_test))

        #
        plt.figure()
        plt.plot(np.arange(len(steps_list)), steps_list, color='rosybrown')
        plt.title('Episode via steps')
        plt.xlabel('Episode')
        plt.ylabel('Steps')

        #
        plt.figure()
        plt.plot(np.arange(len(reward_list)), reward_list, color='rosybrown')
        plt.title('Episode via Success Rate')
        plt.xlabel('Episode')
        plt.ylabel('Success Rate')

        # Showing the plots
        plt.show()


    # Store the final Q table values
    def write_Q_table(self):
        # open data file
        Q = self.Q
        file_name='./Q_table/SARSA'
        filename = open(file_name, 'w')
        # write data
        for k, v in Q.items():
            filename.write(str(k) + ':' + str(v))
            filename.write('\n')
        # close file
        filename.close()


if __name__ == '__main__':
    # create a FrozenLake environment
    env = Environment(grid_size=GRID_SIZE)

    # Create a SARSA agent
    SARSA = SARSA(env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON)

    # write_Q_table(file_name='./Q_table/SARSA', Q = Q)

    # Learning and updating
    SARSA.train(num_epoch=NUM_EPISODES)

    # Test after training
    SARSA.test()

    # Remain visualization
    env.mainloop()
