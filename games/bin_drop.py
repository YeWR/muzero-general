import datetime
import os

import gym
import numpy
import torch
import argparse
import robosuite as suite

from .abstract_game import AbstractGame
from robosuite.scripts.bin_packing_baselines2 import build_env
import numpy as np
from gym import spaces


class MuZeroConfig:
    def __init__(self):
        # More help is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game

        ### Game
        self.observation_shape = (4, 64, 64)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = [i for i in range(2)]  # Fixed list of all possible actions. You should only edit the length
        # low, high = (np.array([0.5, 0.3]), np.array([0.7, 0.5]))
        # self.action_space = spaces.Box(low=low, high=high)
        self.players = [i for i in range(1)]  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = None  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

        ### Self-Play
        self.num_actors = 1  # Number of simultaneous threads self-playing to feed the replay buffer
        self.max_moves = 6  # Maximum number of moves if game is not finished before
        self.num_simulations = 4  # Number of future moves self-simulated
        self.discount = 0.997  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping temperature to 0 (ie playing according to the max)

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 1  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size

        # Residual Network
        self.downsample = False  # Downsample observations before representation network (See paper appendix Network Architecture)
        self.blocks = 1  # Number of blocks in the ResNet
        self.channels = 16  # Number of channels in the ResNet
        self.reduced_channels_reward = 16  # Number of channels in reward head
        self.reduced_channels_value = 16  # Number of channels in value head
        self.reduced_channels_policy = 16  # Number of channels in policy head
        self.resnet_fc_reward_layers = [8]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [8]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [8]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [16]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [16]  # Define the hidden layers in the reward network
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network

        ### Training
        self.results_path = os.path.join(os.path.dirname(__file__), "../results", os.path.basename(__file__)[:-3],
                                         datetime.datetime.now().strftime(
                                             "%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.training_steps = 30000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 64  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.training_device = "cuda" if torch.cuda.is_available() else "cpu"  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.0003  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000

        ### Replay Buffer
        self.window_size = 3000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 20  # Number of game moves to keep for every batch element
        self.td_steps = 20  # Number of steps in the future to take into account for calculating the target value
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)

        # Prioritized Replay (See paper appendix Training)
        self.PER = True  # Select in priority the elements in the replay buffer which are unexpected for the network
        self.use_max_priority = False  # If False, use the n-step TD error as initial priority. Better for large replay buffer
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1
        self.PER_beta = 1.0

        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired self played games per training step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        super().__init__(seed)
        # args = self.get_args()
        # self.env = build_env(args)
        self.env = suite.make(
            'BinPackPlace',
            has_renderer=False,
            has_offscreen_renderer=True,
            ignore_done=True,
            use_camera_obs=True,
            control_freq=1,
            render_drop_freq=20,
            camera_height=64,
            camera_width=64,
            video_height=64,
            video_width=64,

            random_take=True
        )
        # self.env = gym.make('BinPack-v0')

    def get_args(self):
        ## params
        parser = argparse.ArgumentParser(description='Baseline Training...')

        ## env args
        parser.add_argument('--has_renderer', type=bool, default=False)
        parser.add_argument('--use_camera_obs', type=bool, default=True)
        parser.add_argument('--use_object_obs', type=bool, default=True)
        parser.add_argument('--has_offscreen_renderer', type=bool, default=True)
        parser.add_argument('--camera_type', type=str, default='image+depth')
        parser.add_argument('--random_take', type=bool, default=True)
        parser.add_argument('--use_typeVector', type=bool, default=False)
        parser.add_argument('--make_dataset', type=bool, default=False)

        parser.add_argument('--control_freq', type=int, default=1)
        parser.add_argument('--render_drop_freq', type=int, default=0)
        parser.add_argument('--camera_height', type=int, default=64)
        parser.add_argument('--camera_width', type=int, default=64)
        parser.add_argument('--take_nums', type=int, default=6)

        parser.add_argument('--keys', type=str, default='image', choices=['state', 'image'])
        parser.add_argument('--dataset_path', type=str, default='data/temp/')

        ## alg args
        parser.add_argument('--env_id', type=str, default='BinPack-v0')
        parser.add_argument('--alg', type=str, default='ppo2')
        parser.add_argument('--num_env', type=int, default=1)
        parser.add_argument('--load_path', type=str, default='gg')

        ## others
        parser.add_argument('--seed', default=None)

        args = parser.parse_args()
        args.env_id, args.env_type = 'BinPack-v0', 'mujoco'
        return args

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        # observation, reward, done, _ = self.env.step_discrete(action)
        observation, reward, done, _ = self.env.step(action)
        return observation, reward, done, _

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return [i for i in range(8 * 8)]

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def close(self):
        """
        Properly close the game.
        """
        self.env.close()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        actions = {
            0: "Push cart to the left",
            1: "Push cart to the right",
        }
        return "{}. {}".format(action_number, actions[action_number])