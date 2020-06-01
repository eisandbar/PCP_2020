import numpy as np
from typing import Optional
from connectn.common import apply_player_action, connected_four_iter, connected_N_iter, SavedState

PlayerAction = np.int8
BoardPiece = np.int8


def generate_move(board: np.ndarray, player: BoardPiece,
                  saved_state: Optional[SavedState] = None) -> (np.int8, Optional[SavedState]):
    """Function implements an alpha-beta pruning minimax search algorithm and returns the best move

    For all possible child nodes checks their value using function alpha_beta
    If the value is larger than the current highest changes to that move and updates max_value
    Returns the best move and saved_sate
    """
    depth = 4  # The depth of our minimax tree
    order = np.asarray([3, 2, 4, 1, 5, 0, 6])  # The order in which we check nodes
    move: PlayerAction = np.argwhere(board == 0)[0, 1]  # Default move when no moves are good
    max_value = np.NINF  # Starting value of -inf
    alpha = np.NINF  # Staring alpha
    beta = np.inf  # Starting beta

    for i in order:
        if board[5, i] == 0:  # If move is possible
            value = alpha_beta(board.copy(), i, depth, alpha, beta, player, self=True)  # Find value
            alpha = np.maximum(alpha, value)  # Updates alpha
            if value > max_value:  # If new highest value
                move = i  # Update move
                max_value = value  # Update max_value
    return np.int8(move), saved_state


def alpha_beta(board: np.ndarray, node: PlayerAction, depth: int, alpha: int, beta: int,
               player: BoardPiece, self: bool) -> float:
    """Function implements an alpha-beta pruning minimax algorithm to find the value of the node

    First applies the last action and checks to make sure it wasn't a winning move.
    Next checks whether the depth is zero, in which case using the heuristic function returns the value of the node
    Otherwise goes over all the children of node and checks their value
    Returns the calculated value of node
    """
    order = np.asarray([3, 2, 4, 1, 5, 0, 6])  # The order in which we check nodes

    # Board update
    if self:  # Self keeps track from whose perspective we are currently playing
        new_board = apply_player_action(board.copy(), node, player)  # Updates the board
        if connected_four_iter(board.copy(), player, 4):  # Checks for a win
            return np.inf
    else:
        new_board = apply_player_action(board.copy(), node, 3-player)  # Updates the board
        if connected_four_iter(board.copy(), 3-player, 4):  # Checks for a loss
            return np.NINF

    # Terminal node check
    if depth == 0:  # Checks if terminal node
        value = heuristic(new_board, player)  # Heuristic
        return value

    # Going deeper into minimax tree
    if self:  # If playing as self
        value = np.inf  # Starting value of inf
        for i in order:  # Going over all children
            if new_board[5, i] == 0:  # Checking that move is possible
                value = np.minimum(value, alpha_beta(new_board.copy(), i, depth-1,
                                                     alpha, beta, player, not self))  # Updating value
                beta = np.minimum(beta, value)  # Updating beta
                if beta <= alpha:  # Prune condition
                    return value
        return value

    else:  # If playing as opponent
        value = np.NINF  # Starting value of -inf
        for i in order:  # Going over all children
            if new_board[5, i] == 0:  # If move is possible
                value = np.maximum(value, alpha_beta(new_board.copy(), i, depth-1,
                                                     alpha, beta, player, not self))  # Updating value
                alpha = np.maximum(alpha, value)  # Updating alpha
                if alpha >= beta:  # Prune condition
                    return value
        return value


def heuristic(board: np.ndarray, player: BoardPiece) -> float:
    """Function uses a variation of connected_four_iter to calculate the value of a terminal node

    plus_value are the points gained for having 2,3 or 4 pieces in a row
    minus_value are the points lost for your opponent having 2,3 or 4 pieces in a row
    The heuristic considers the opponents achievements with a greater weight
    Returns the sum of plus_value and minus_value
    """
    plus_value = connected_N_iter(board, player, 4)*10000 + connected_N_iter(board, player, 3)*100\
                 + connected_N_iter(board, player, 2)*1.0

    minus_value = - connected_N_iter(board, 3-player, 4)*500000 - connected_N_iter(board, 3-player, 3)*300\
                  - connected_N_iter(board, 3-player, 2)*2.0

    return plus_value + minus_value
