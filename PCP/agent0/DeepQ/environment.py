"""In this file we create the environment in which we train our agent

The class Connect4Env contains 3 main functions:

__init__: initializes the environment
_reset: restarts the game
_step: applies the player action

Environment can be set up so that the agent either plays itself or against a minimax agent
For simplicity the agent always plays with int8(1)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from connectn.common import initialize_game_state, apply_player_action, check_end_state, check_valid_action, GameState
from connectn.agent_minimax import generate_move
from DeepQ.agent_DQN import generate_move_DQN

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()


class Connect4Env(py_environment.PyEnvironment):

    def __init__(self):
        """Class initialization

        _action_spec defines what kind of actions are expected, an int in range [0, 6]
        _observation_spec defines what the agent can see. In our case the agent can see the whole board, a (6,7) array
        _reward_spec defines the shape and dtype of the reward
        _state initializes as a board of zeros using function initialize_game_state.
        With a 50% chance the second player moves first
        """
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int8, minimum=0, maximum=6, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(6, 7), dtype=np.int8, minimum=0, name='observation')
        self._reward_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.float32, name='reward')
        self._state = initialize_game_state()
        if 0.5 > np.random.rand(1)[0]:
            self._state = apply_player_action(self._state, 3, player=np.int8(-1))

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        """Resets the game

        resets _state to an empty board using function initialize_game_state.
        With a 50% chance the second player moves first
        returns the TimeStep class
        """
        self._state = initialize_game_state()
        if 0.5 > np.random.rand(1)[0]:
            self._state = apply_player_action(self._state, 3, player=np.int8(-1))
        return ts.restart(np.array(self._state, dtype=np.int8))

    def _step(self, action):
        """Applies player action

        Applies the player action and if the game isn't finished generates and applies the second player's action

        action: an int in range [0,6], the action the agent is to take
        returns the TimeStep that contains the new state and reward of the action
        """
        if check_valid_action(self._state, action):  # Makes sure we don't make illegal moves
            self._state = apply_player_action(self._state, action, player=np.int8(1))
            playing = check_end_state(self._state, player=np.int8(1))

            if playing == GameState.STILL_PLAYING:  # If the last move didn't finish the game
                board = np.where(self._state == -1, 2, self._state)
                second_player_action, _ = generate_move(board, player=np.int8(2), depth=2)
                self._state = apply_player_action(self._state, second_player_action, player=np.int8(-1))
                playing = check_end_state(self._state, player=np.int8(-1))

                if playing == GameState.STILL_PLAYING:  # If still playing
                    return ts.transition(np.array(self._state, dtype=np.int8), reward=10, discount=1)

                elif playing == GameState.IS_DRAW:  # If a draw
                    return ts.termination(np.array(self._state, dtype=np.int8), reward=50)

                elif playing == GameState.IS_WIN:  # If second player won
                    return ts.termination(np.array(self._state, dtype=np.int8), reward=-500)

            elif playing == GameState.IS_DRAW:  # If a draw
                return ts.termination(np.array(self._state, dtype=np.int8), reward=50)

            elif playing == GameState.IS_WIN:  # If first player one
                return ts.termination(np.array(self._state, dtype=np.int8), reward=500)

        else:
            return ts.transition(np.array(self._state, dtype=np.int8), reward=-10000, discount=1)

