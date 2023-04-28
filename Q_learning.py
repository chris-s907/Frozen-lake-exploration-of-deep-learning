import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from Environment import Environment
from Parameters import *

np.random.seed(1)


class Q_learning(object):
    def __init__(self, env, learning_rate, gamma, epsilon):
        
        self.env = env
        self.actions = list(range(self.env.n_actions))
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.success = []
        self.fail = []
        self.q_table = pd.DataFrame(columns=self.actions)
        self.q_table_final = pd.DataFrame(columns=self.actions)

    # Adding to the Q-table new states
    def check_state_validation(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )


    # Choose valid actoins
    def epsilon_greedy_policy(self, observation):

        self.check_state_validation(observation)
        
        if np.random.uniform(0, 1) < self.epsilon*0.75:
            action = np.random.randint(0, 3) 
            state_action = self.q_table.loc[observation, :]
            while action == np.random.choice(state_action[state_action == np.max(state_action)].index):
                action = np.random.randint(0, 3)
            return action

        else:
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        return action

    # Learning and updating the Q table using the Q learning update rules as :
    # Q(s,a) = Q(s,a) + alpha *(r + gamma * max[Q(s',a)] - Q(s,a))
    def learn(self, state, action, reward, next_state,step):
        self.check_state_validation(next_state)

        q_predict = self.q_table.loc[state, action]

        q_target = reward + self.gamma * self.q_table.loc[next_state, :].max()

        k = step
        if step == 0: k =1
        
        self.q_table.loc[state, action] += (self.lr/k) * (q_target - q_predict)

        return self.q_table.loc[state, action]

    # Train for updating the Q table
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

        for i in range(num_epoch):
            observation = self.env.reset()
            step = 0
            cost = 0

            # Calculate the accuracy for every 50 steps
            if i != 0 and i % 50 == 0:
                goal_count = goal_count / 50
                accuracy += [goal_count]
                goal_count = 0

            # Record Q value for specific grid for checking converging
            if i != 0 and i % 1000 == 0:
                Q_value[i] = []
                for j in range(self.env.n_actions):
                    Q_value[i].append(self.q_table.loc[str(14), j])

            while True:
                # Render environment
                # self.env.render()

                action = self.epsilon_greedy_policy(str(observation))
                observation_, reward, done, info = self.env.step(action)
                cost += self.learn(str(observation), action, reward, str(observation_),step)
                observation = observation_
                step += 1

                # Break while loop when it is the end of current Episode
                # When agent reached the goal or obstacle
                if done:
                    # Record the positive cost and negative cost
                    if reward > 0:
                        positive_count += 1
                        self.success += [i]
                    else:
                        negative_count += 1
                        self.fail += [i]
                    # Record the step
                    steps += [step]

                    # Record the cost
                    all_costs += [cost]

                    # goal count +1, if reaching the goal
                    #4*4
                    # if reward == 1:
                    #10*10
                    if reward == 5:
                        goal_count += 1

                    # Record total rewards to calculate average rewards
                    rewards += reward
                    Reward_list += [rewards / (i + 1)]

                    break

            print('episode:{}'.format(i))

        # See if converge
        print("Q_value:{}".format(Q_value))

        # Record the data to the list
        all_cost_bar = [positive_count, negative_count]

        self.plot_results(steps, all_costs, accuracy, all_cost_bar, Reward_list)

        return self.q_table, steps, all_costs, accuracy, all_cost_bar, Reward_list, self.success, self.fail

    def write_Q_table(self):
        # open data file
        Q = self.q_table
        Q.to_csv('./Q_table/Q-learning.csv')

    # plot training results
    def plot_results(self, steps, cost, accuracy, all_cost_bar, Reward_list):

        #
        f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
        #
        ax1.plot(np.arange(len(steps)), steps, color='rosybrown')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Steps')
        ax1.set_title('Episode via steps')

        #
        ax2.plot(np.arange(len(cost)), cost, color='rosybrown')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Cost')
        ax2.set_title('Episode via cost')

        #
        ax3.plot(np.arange(len(accuracy)), accuracy, color='rosybrown')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Episode via Accuracy')

        plt.tight_layout()  # Function to make distance between figures

        
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

        for i in range(num_test):

            observation = self.env.reset()

            for j in range(NUM_STEPS):
                # render the environment
                # self.env.render()

                state_action = self.q_table.loc[str(observation), :]
                action = np.random.choice(state_action[state_action == np.max(state_action)].index)

                next_observation, reward, done, info = self.env.step(action)

                y = int(math.floor(next_observation / GRID_SIZE)) * PIXELS
                x = int(next_observation % GRID_SIZE) * PIXELS
                f[j] = [x, y]

                if done:
                    # Record the number of goal reaching
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

        # Plot results
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


# Commands to be implemented after running this file
if __name__ == "__main__":
    # Create an environment
    env = Environment(grid_size=GRID_SIZE)

    # Create a q learning agent
    Q_learning = Q_learning(env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON)

    # Learning and updating
    Q_table = Q_learning.train(num_epoch=NUM_EPISODES)

    Q_learning.test()

    # Remain visualization
    env.mainloop()
