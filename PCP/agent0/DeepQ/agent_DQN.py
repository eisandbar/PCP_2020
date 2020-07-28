"""In this file we create our generate_move function

The function uses a DQN to approximate Q-values and use that to choose an action
The saved DQN model is used by default, but can be changed when calling
"""

import numpy as np
from typing import Optional
import tensorflow as tf
import os

from DeepQ.model import num_actions
from connectn.common import SavedState


os.chdir("C:/Users/Polaris/Documents/GitHub/PCP_2020/PCP/agent0/models")
my_model = tf.keras.models.load_model('my_model.h5')
PlayerAction = np.int8
BoardPiece = np.int8


def generate_move_DQN(board: np.ndarray, player: BoardPiece, model=my_model,
                      saved_state: Optional[SavedState] = None) -> (np.int8, Optional[SavedState]):
    """Function uses the model to approximate Q-values for each action and chooses the best one

    :param board: The current state of the board
    :param player: The pieces the player is playing with
    :param model: The model used to predict Q-values, defaults to the one saved in file
    :param saved_state: Optional, to match the minimax genmove
    :return: a player action (int8) and saved_state (SavedState)
    """

    # The environment had the agent always play with player==1, so this inverts the board if necessary
    board = np.where(board == 2, -1, board)
    if player == 2:
        board = -board

    # Transforms the state so that it can be fed to the model
    state_tensor = tf.convert_to_tensor(board)
    state_tensor = tf.expand_dims(state_tensor, 0)

    # Calculate Q-values
    action_probs = model(state_tensor, training=False)

    # Take best action of those allowed
    possible_actions = (board[5, :] == 0)
    max_idx = np.argmax(action_probs.numpy()[0, possible_actions])
    action = np.arange(num_actions)[possible_actions][max_idx]
    return np.int8(action), saved_state
