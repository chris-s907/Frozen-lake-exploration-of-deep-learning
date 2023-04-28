import numpy as np
import matplotlib.pyplot as plt
import argparse

from Environment import Environment
from Parameters import *
from Monte_carlo_control import Monte_carlo
from SARSA import SARSA
from Q_learning import Q_learning


# Define line plot functions
# Episodes via Costs
def plot_all_cost(all_cost, label):
    plt.figure()
    plt.plot(np.arange(len(all_cost[0])), all_cost[0], alpha=0.8, c='coral', label=label[0], linewidth=1)
    plt.plot(np.arange(len(all_cost[1])), all_cost[1], alpha=0.8, c='#6B8E23', label=label[1], linewidth=1)
    # plt.plot(np.arange(len(all_cost[2])), all_cost[2], alpha=0.8, c='dodgerblue', label=label[2], linewidth=1)
    plt.title('Episode via Cost')
    plt.xlabel('Episode')
    plt.ylabel('Cost')
    plt.legend(loc='best')
    plt.show()


# Episodes via Accuracy
def plot_accuracy(accuracy, label):
    plt.figure()
    plt.plot(np.arange(len(accuracy[0])), accuracy[0], alpha=0.8, c='coral', label=label[0], linewidth=1)
    plt.plot(np.arange(len(accuracy[1])), accuracy[1], alpha=0.8, c='#6B8E23', label=label[1], linewidth=1)
    # plt.plot(np.arange(len(accuracy[2])), accuracy[2], alpha=0.8, c='dodgerblue', label=label[2], linewidth=1)
    plt.title('Episode via Accuracy')
    plt.xlabel('Episode')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.show()



# Define scatter plot functions
# Monte Carlo
def plot_steps_scatter(steps, success, fail, name):
    plt.figure()
    label = ['success', 'fail','optimal policy']
    plt.scatter(success, [steps[i] for i in success], alpha=1, s=0.1, c='coral', label=label[0])
    plt.scatter(fail, [steps[i] for i in fail], alpha=1, s=0.1, c='dodgerblue', label=label[1])
    plt.title('Training process via Steps {}'.format(name))
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.legend(loc='best')
    plt.show()



if __name__ == '__main__':
    np.random.seed(1)

    parser = argparse.ArgumentParser(description="Parameters need to be input for testing")
    parser.add_argument('--job', default=2)
    parser.add_argument('--grid_size', default=10)
    parser.add_argument('--num_epoch', default=100000)
    args = parser.parse_args()

    # Job 0, 4x4 frozen lake environment training, correctness test, and comparison test
    if args.job == 0:
        env = Environment(grid_size=4)
        # Create three agents corresponding to three algorithms
        Monte_carlo = Monte_carlo(env, epsilon=EPSILON, gamma=GAMMA)

        SARSA = SARSA(env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON)

        Q_learning = Q_learning(env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON)

        label_1 = ['Monte_carlo', 'SARSA', 'Q_learning']

        Q_1, steps_1, all_cost_1, accuracy_1, all_cost_bar_1, Rewards_list_1 , success1, fail1= Monte_carlo.fv_mc_prediction(num_epoch=args.num_epoch)
        Monte_carlo.write_Q_table()
        Q_2, steps_2, all_cost_2, accuracy_2, all_cost_bar_2, Rewards_list_2 , success2, fail2= SARSA.train(num_epoch=args.num_epoch)
        SARSA.write_Q_table()
        Q_3, steps_3, all_cost_3, accuracy_3, all_cost_bar_3, Rewards_list_3 , success3, fail3= Q_learning.train(num_epoch=args.num_epoch)
        Q_learning.write_Q_table()

        all_cost = [all_cost_1, all_cost_2, all_cost_3]

        accuracy = [accuracy_1, accuracy_2, accuracy_3]
        
        plot_steps_scatter(steps_1, success1, fail1, name = 'Monte Carlo')
        plot_steps_scatter(steps_2, success2, fail2, name = 'SARSA')
        plot_steps_scatter(steps_3, success3, fail3, name = 'Q_learning')
        
        plot_all_cost(all_cost, label_1)

        plot_accuracy(accuracy, label_1)

    # Job 1, 10X10 frozen lake environment training, correctness test, and comparison test
    elif args.job == 1:
        NUM_EPISODES = 20000

        GRID_SIZE = 10

        env = Environment(grid_size=10)
        # Create three agents corresponding to three algorithms
        SARSA = SARSA(env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON)

        Q_learning = Q_learning(env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON)

        label_1 = ['SARSA', 'Q_learning']

        Q_2, steps_2, all_cost_2, accuracy_2, all_cost_bar_2, Rewards_list_2, success2, fail2 = SARSA.train(num_epoch=args.num_epoch)

        Q_3, steps_3, all_cost_3, accuracy_3, all_cost_bar_3, Rewards_list_3, success3, fail3 = Q_learning.train(num_epoch=args.num_epoch)

        plot_steps_scatter(steps_2, success2, fail2, name = 'SARSA')
        plot_steps_scatter(steps_3, success3, fail3, name = 'Q_learning')

        all_cost = [all_cost_2, all_cost_3]

        accuracy = [accuracy_2, accuracy_3]

        plot_all_cost(all_cost, label_1)

        plot_accuracy(accuracy, label_1)

    # Job 2 Monte Carlo learning of 10*10 grid
    elif args.job == 2:
        NUM_EPISODES = 100000

        GRID_SIZE = 10

        env = Environment(grid_size=10)
        # Create three agents corresponding to three algorithms
        Monte_carlo = Monte_carlo(env, epsilon=EPSILON, gamma=GAMMA)
        Q_1, steps_1, all_cost_1, accuracy_1, all_cost_bar_1, Rewards_list_1, success1, fail1 = Monte_carlo.fv_mc_prediction(num_epoch=args.num_epoch)

        plot_steps_scatter(steps_1, success1, fail1, name = 'Monte Carlo')
        
    