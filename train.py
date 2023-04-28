import argparse

from Environment import Environment
from Parameters import *
from Monte_carlo_control import Monte_carlo
from SARSA import SARSA
from Q_learning import Q_learning


parser = argparse.ArgumentParser(description="Parameters need to be input for training")
parser.add_argument('--agent', type = str, default='sarsa')
parser.add_argument('--grid_size', type = int, default=10)
parser.add_argument('--num_epoch', type = int, default=10000)
args = parser.parse_args()

env = Environment(grid_size=args.grid_size)

if args.agent == 'mc':
    # Create a monte carlo agent
    monte_carlo = Monte_carlo(env, epsilon=EPSILON, gamma=GAMMA)

    # Learning and updating Q table
    Q = monte_carlo.fv_mc_prediction(num_epoch=NUM_EPISODES)

    monte_carlo.write_Q_table()

    # Test after training
    monte_carlo.test()

    # Remain visualization
    env.mainloop()

elif args.agent == 'sarsa':
    # Create a SARSA agent
    SARSA = SARSA(env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON)
    
    # Learning and updating
    Q = SARSA.train(num_epoch=NUM_EPISODES)

    SARSA.write_Q_table()

    # Test after training
    SARSA.test()

    # Remain visualization
    env.mainloop()

elif args.agent == 'ql':
    # Create a q learning agent
    Q_learning = Q_learning(env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON)

    # Learning and updating
    Q_table = Q_learning.train(num_epoch=NUM_EPISODES)

    Q_learning.write_Q_table()
    # Test after training
    Q_learning.test()

    # Remain visualization
    env.mainloop()
