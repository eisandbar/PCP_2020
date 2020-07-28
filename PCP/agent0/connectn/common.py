import numpy as np
from enum import Enum
from typing import Optional, Callable, Tuple

BoardPiece = np.int8  # The data type (dtype) of the board
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 has a piece

PlayerAction = np.int8  # The column to be played

class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0

def initialize_game_state() -> np.ndarray:
    """Function creates a (6, 7) array of zeroes, dtype = BoardPiece"""
    return np.zeros((6, 7), dtype=BoardPiece)


def pretty_print_board(board: np.ndarray) -> str:
    """Function converts the board array to a string"""

    pp_board = ""  # Starts empty
    for i in range(6):  # For all rows
        for j in range(7):  # For all columns
            if board[5-i, j] == 0:
                pp_board += ' '  # Zeroes become spaces
            elif board[5-i, j] == 1:
                pp_board += 'X'  # Ones become Xs
            elif board[5-i, j] == 2:
                pp_board += 'O'  # Twos become Os
        pp_board += "\n"  # after every row we add a new line
    pp_board += "_______\n"  # Decorative line
    pp_board += "0123456\n"  # Column indexes
    return pp_board


def apply_player_action(
    board: np.ndarray, action: PlayerAction, player: BoardPiece, copy: bool=False,
) -> np.ndarray:
    """Function changes the lowest empty slot in column 'action' to 'player'

    Does not check for availability, agent has to do it itself
    Does not check that player = {1, 2}
    """
    row = np.argwhere(board[:, action] == 0)  # Finds all the empty slots in column 'action'
    board[row[0, 0], action] = player  # Changes slot to player
    return board


def string_to_board(pp_board: str) -> np.ndarray:
    """Function takes as input a string and returns its board equivalent"""
    board = np.zeros((6, 7), dtype=BoardPiece)
    for i in range(6):  # For all rows
        for j in range(7):  # For all columns
            if pp_board[8 * (5 - i) + j] == ' ':
                board[i, j] = 0  # Spaces are 0
            elif pp_board[8 * (5 - i) + j] == 'X':
                board[i, j] = 1  # X is 1
            elif pp_board[8 * (5 - i) + j] == 'O':
                board[i, j] = 2  # O is 2
    return board

from numba import njit

@njit()
def connected_four_iter(
    board: np.ndarray, player: BoardPiece, CONNECT_N,_last_action: Optional[PlayerAction] = None
) -> bool:
    rows, cols = board.shape
    rows_edge = rows - CONNECT_N + 1
    cols_edge = cols - CONNECT_N + 1
    for i in range(rows):
        for j in range(cols_edge):
            if np.all(board[i, j:j+CONNECT_N] == player):
                return True
    for i in range(rows_edge):
        for j in range(cols):
            if np.all(board[i:i+CONNECT_N, j] == player):
                return True
    for i in range(rows_edge):
        for j in range(cols_edge):
            block = board[i:i+CONNECT_N, j:j+CONNECT_N]
            if np.all(np.diag(block) == player):
                return True
            if np.all(np.diag(block[::-1, :]) == player):
                return True
    return False


@njit()
def connected_N_iter(
    board: np.ndarray, player: BoardPiece, CONNECT_N,_last_action: Optional[PlayerAction] = None
) -> int:
    """Function is a copy of connected_four_iter with the exception that instead of a bool it
    returns an int, count = the amount of times N pieces in a row can be found
    """
    rows, cols = board.shape
    rows_edge = rows - CONNECT_N + 1
    cols_edge = cols - CONNECT_N + 1
    count = 0
    for i in range(rows):
        for j in range(cols_edge):
            if np.all(board[i, j:j+CONNECT_N] == player):
                count +=1
    for i in range(rows_edge):
        for j in range(cols):
            if np.all(board[i:i+CONNECT_N, j] == player):
                count +=1
    for i in range(rows_edge):
        for j in range(cols_edge):
            block = board[i:i+CONNECT_N, j:j+CONNECT_N]
            if np.all(np.diag(block) == player):
                count +=1
            if np.all(np.diag(block[::-1, :]) == player):
                count +=1
    return count

def check_end_state(
    board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> GameState:
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING)?
    """
    if connected_four_iter(board, player, 4):
        return GameState.IS_WIN
    elif not np.any(board == 0):  # If no more empty slots
        return GameState.IS_DRAW
    else:
        return GameState.STILL_PLAYING

def check_valid_action(
    board: np.ndarray, action: PlayerAction,
) -> bool:
    """Function checks whether the action is available"""
    if board[5, action] == 0:  # If move is possible
        return True
    else:
        return False


class SavedState:
    pass


GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    Tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]

