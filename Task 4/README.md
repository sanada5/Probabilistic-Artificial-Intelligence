The goal of this project is to develop an off-policy reinforcement learning algorithm to control and swing up an inverted pendulum, starting from an angle of $\pi$ and aiming for $0$. 
This repository contains the code that implements an off-policy reinforcement learning algorithm using a neural network architecture to control and swing up an inverted pendulum. It defines an actor-critic framework where 
the actor network learns the policy, and the critic network evaluates the actions taken by the actor. The agent utilizes an experience replay buffer to store and sample experiences for training, and it employs target 
networks and a soft update mechanism to stabilize training. The agent is trained through multiple episodes, alternating between exploration and exploitation phases, and tested for performance by running episodes in the 
environment and calculating the average return.
