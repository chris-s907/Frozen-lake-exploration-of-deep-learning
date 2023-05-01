## Dependencies
* This repo depends on the following standard ROS pkgs:
  * `numpy`
  * `matplotlib`
  * `tkinter`
  * `math`
  * `argparse`
  * `PIL`
## Files
* `Q table` includes the results of Q table
	  * 4*4 : Q table of the 4*4 environment of three algorithms
	  * 10*10 : Q table of the 10*10 environment of three algorithms

* `Environment.py`  Build the environment
* `Parameters.py`  Set the parameters
* `Monte_Carlo_control.py`  Perform Monte Carlo algorithm
* `Q_learning`  Perform Q learning algorithm
* `SARSA`  Perform SARSA algorithm
* `train.py`  The main file to run all the algorithm by the input (can change the default value to get the results as expected)
* `test.py`  Run the three algorithms simutaneously and get the figure for analysis

