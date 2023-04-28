import random
import numpy as np
import math
import matplotlib.pyplot as plt

from Environment import Environment
from collections import defaultdict
from Parameters import *

np.random.seed(10)


class Monte_carlo(object):
    def __init__(self, env, epsilon, gamma):
        # Variable initialization
        self.env = env
        self.n_obs = self.env.n_states
        self.n_a = self.env.n_actions

        self.epsilon = epsilon
        self.gamma = gamma
        # Variables for metrics
        self.steps = []
        self.all_cost = []
        self.accuracy = []
        self.Rewards_list = []
        self.success = []
        self.fail = []
        self.rewards = 0
        self.positive_count = 0
        self.negative_count = 0
        self.goal_count = 0

        self.Q, self.Total_return, self.N, self.failpair = self.create_q_table()

    # Create a Q table
    def create_q_table(self):
        Q = defaultdict(float)
        N = defaultdict(int)
        Total_return = defaultdict(float)
        failpair = defaultdict(int)

        for s in range(self.n_obs):
            for a in range(self.n_a):
                Q[(s, a)] = 0.0
                Total_return[(s, a)] = 0
                N[(s, a)] = 0
                failpair[(s,a)] = 0
        return Q, Total_return, N, failpair

    # Choose actions based on epsilon greedy policy
    def epsilon_greedy_policy(self, observation):
        if np.random.uniform(0, 1) < 0.5*0.75:
            action = np.random.randint(0, 3) #generate a random integer 0<= x <=3
            while action == max(list(range(self.n_a)), key=lambda x: self.Q[(observation, x)]):
                action = np.random.randint(0, 3)
            return action

        else:
            return max(list(range(self.n_a)), key=lambda x: self.Q[(observation, x)])


    def optimal_policy(self, observation):
        return max(list(range(self.n_a)), key=lambda x: self.Q[(observation, x)])

    # Generate a list of data for a episode
    def generate_episode(self,epi):
        episode = []
        observation = self.env.reset()
        steps = 0

        # Loop for each time step
        for t in range(NUM_STEPS):
            
            action = self.epsilon_greedy_policy(observation)

            next_observation, reward, done, info = self.env.step(action)

            episode.append((observation, action, reward))

            steps += 1

            if done:

                if reward > 0:
                    self.positive_count += 1
                    self.goal_count += 1
                    self.success += [epi]
                else:
                    self.negative_count += 1
                    self.fail += [epi]

                # Record the step
                self.steps += [steps]

                self.rewards += reward

                print("Episode finished after {} steps".format(t + 1))
                break
            
            #4*4 env
            # elif steps == 100:
            #     self.steps += [100]
            #     self.fail += [epi]
            #     print("Episode has'n finished after {} steps".format(t + 1))

            #10*10 env
            elif steps == 5000:
                self.steps += [5000]
                self.fail += [epi]
                print("Episode has'n finished after {} steps".format(t + 1))
         
            observation = next_observation

        return episode

    # Learning and updating Q table based on the First visit Monte Carlo method
    def fv_mc_prediction(self, num_epoch):

        for i in range(num_epoch):
            cost = 0

            episode = self.generate_episode(i)

            state_action_pairs = [(observation, action) for (observation, action, reward) in episode]
            
            ## check if the final trap is the same and give the penalty
            # observation_f, action_f, reward_f = episode[len(episode) - 1]
            # if reward_f == -1 :
            #     self.failpair[(observation_f, action_f)] += 1
            #     if self.failpair[(observation_f, action_f)] != 1:
            #         G = -1*self.failpair[(observation_f, action_f)]
            #     else:
            #         G = 0
            # elif reward_f == 1:
            #     self.failpair[(observation_f, action_f)] += 1
            #     G = self.failpair[(observation_f, action_f)]
            # else:
            #     # Initialize the G value
            #     G = 0
            
            # Initialize the G value
            G = 0

            # Calculate the accuracy rate for every 50 steps
            if i != 0 and i % 50 == 0:
                self.goal_count = self.goal_count / 50
                self.accuracy += [self.goal_count]
                self.goal_count = 0

            self.Rewards_list += [self.rewards / (i + 1)]

            for i in range(len(episode)):
                # Calculate the return G from the end, T-1, T-2...... by G = gamma* G + R(t+1)
                observation, action, reward = episode[len(episode) - (i + 1)]
                
                if i == 0:
                    G = reward + self.gamma * G
                else:
                    if (observation, action, reward) == episode[len(episode) - i] :
                        G = reward + self.gamma * G - 3
                    elif action == 1 or action == 2 : 
                        G = G + 4
                    else:
                        G = reward + self.gamma * G 
                
                if 5 <= observation/10 <=9 | 5 <= observation%10 <= 9:
                    G = G + 3
                
                
                # Check if the state-action pair is occurring for the first time in the episode
                if not (observation, action) in state_action_pairs[:len(episode) - (i + 2)]:
                    self.Total_return[(observation, action)] += G
                    self.N[(observation, action)] += 1
                    self.Q[(observation, action)] = self.Total_return[(observation, action)] / self.N[
                        (observation, action)]
                    
                cost += self.Q[(observation, action)]
                
            self.all_cost += [cost]

        all_cost_bar = [self.positive_count, self.negative_count]
        self.plot_results(self.steps, self.all_cost, self.accuracy, all_cost_bar, self.Rewards_list)

        return self.Q, self.steps, self.all_cost, self.accuracy, all_cost_bar, self.Rewards_list, self.success, self.fail

    # Plot training results
    @ staticmethod
    def plot_results(steps, all_cost, accuracy, all_cost_bar, Reward_list):
        # Plot Episodes vis steps
        plt.figure()
        plt.plot(np.arange(len(steps)), steps, 'rosybrown')
        plt.title('Episode via steps')
        plt.xlabel('Episode')
        plt.ylabel('Steps')

        # Plot Episodes via Cost
        plt.figure()
        plt.plot(np.arange(len(all_cost)), all_cost,'rosybrown')
        plt.title('Episode via Cost')
        plt.xlabel('Episode')
        plt.ylabel('Cost')

        # Plot Episodes via Accuracy
        plt.figure()
        plt.plot(np.arange(len(accuracy)), accuracy, 'rosybrown')
        plt.title('Episode via Accuracy')
        plt.xlabel('Episode')
        plt.ylabel('Accuracy')

        # Plot Bar of Success and failure rate
        plt.figure()
        list = ['Success', 'Fail']
        color_list = ['steelblue', 'rosybrown']
        plt.bar(np.arange(len(all_cost_bar)), all_cost_bar, tick_label=list, color=color_list)
        plt.title('Bar/Success and Fail')
        plt.ylabel('Number')

        # Plot Episode via Average rewards
        plt.figure()
        plt.plot(np.arange(len(Reward_list)), Reward_list, 'rosybrown')
        plt.title('Episode via Average rewards')
        plt.xlabel('Episode')
        plt.ylabel('Average rewards')

        
        plt.show()

    # Test after training
    def test(self):
        num_test = 100

        f = {}

        num_reach_goal = 0
        reward_list = []
        steps_list = []

        for i in range(num_test):
            observation = self.env.reset()

            # render the environment
            # env.render()

            for j in range(NUM_STEPS):
                # # render the environment
                # self.env.render()

                action = self.optimal_policy(observation)

                next_observation, reward, done, info = self.env.step(action)

                y = int(math.floor(next_observation / GRID_SIZE)) * PIXELS
                x = int(next_observation % GRID_SIZE) * PIXELS
                f[j] = [x, y]

                if done:
                    if reward == 1:
                        num_reach_goal += 1
         
                    r = reward
                    step = j + 1
                    reward_list += [r]
                    steps_list += [step]

                    break

                observation = next_observation

        # Print final route
        self.env.f = f
        self.env.final()

        print("Accuracy:{}".format(num_reach_goal / num_test))

        # Plot the test results
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
        file_name='./Q_table/monte_carlo'
        filename = open(file_name, 'w')
        # write data
        for k, v in Q.items():
            filename.write(str(k) + ':' + str(v))
            filename.write('\n')
        # close file
        filename.close()


if __name__ == "__main__":
    # create a FrozenLake environment
    env = Environment(grid_size=GRID_SIZE)

    # Create a monte carlo agent
    monte_carlo = Monte_carlo(env, epsilon=EPSILON, gamma=GAMMA)

    # Learning and updating Q table
    Q = monte_carlo.fv_mc_prediction(num_epoch=NUM_EPISODES)

    # write_Q_table(file_name='./Q_table/monte_carlo', Q = Q)

    # Test after training
    monte_carlo.test()

    # Remain visualization
    env.mainloop()
