# Reimplementation and Comparative Analysis of Deep Q-Network Algorithms for Atari 2600 Games
Implementing and comparing different Deep Reinforcement Learning algorithms for playing Atari games for a University Project

# Requirements

To run the following project, you need to have the following installed on your system:
- Python 3.x
- Tensorflow 2.x
- Gymnasium 0.29
- Numpy
- Wandb (Optional)

# Usage

To start training the algorithms, modify the file "DQN_Training.ipynb" by changing the name of the environment and uncommenting the Model you want to use.

## Procedure:

1. Open the file "DQN_Training.ipynb"
2. modify "env.make("Name of the environment",render_mode="human"(optional))
3. Uncomment the agent desired, use DQNAgent for the DQN algorithm, DoubleDQNAgent for DoubleDQN, etc...
4. Run the notebook

## How to visualize results
This project uses Wandb for the visualization of the algorithms' performance.
To use it, you must be registered to the Wandb website.
Then modify "wandb.init(name="Name of the Observation", project="name of the project")

## Folder Structure
The folder structure of the project is as follows:
```csharp
DRLAtari/
  ├── DQN_family_algorithms/
  |    ├── saved_models/
  │    ├── DoubleDQN.py
  │    ├── DQN_Trainin.ipynb
  │    ├── DQNAgent.py
  │    ├── DuelingDoubleDQN.py
  │    └── DuelingDQNAgent.py
  ├── losses/
  │    ├── Hubert.py
  │    └── MSE.py
  ├── optimizers/
  │    └── Adam.py
  ├── NeuralNetwork.py
  └── README.md
```
* DQN_family_algorithms/: Contains the models of each algorithm plus the notebook to run the training.
* losses/: Contains the implemented losses functions.
* optimizers/: Contains the implemented optimizers.

