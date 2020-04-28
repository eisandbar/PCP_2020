import numpy as np
from typing import Optional
PlayerAction = np.int8
BoardPiece = np.int8

def initialize_game_state() -> np.ndarray:
    return np.zeros((6, 7), dtype=BoardPiece)


def pretty_print_board(board: np.ndarray) -> str:
    pp_board = ""
    for i in range(6):
        for j in range(7):
            if board[5-i, j] == 0:
                pp_board += ' '
            elif board[5-i, j] == 1:
                pp_board += 'X'
            elif board[5-i, j] == 2:
                pp_board += 'O'
        pp_board += "\n"
    pp_board += "_______\n"
    pp_board += "0123456\n"
    return pp_board


def apply_player_action(
    board: np.ndarray, action: PlayerAction, player: BoardPiece, copy: bool=False,
) -> np.ndarray:
    row = np.where(board[:, action]==0)
    board[row[0][0], action] = player;
    return board;


def string_to_board(pp_board: str) -> np.ndarray:
    board = np.zeros((6, 7), dtype=BoardPiece)
    for i in range(6):
        for j in range(7):
            if pp_board[8 * (5 - i) + j] == ' ':
                board[i, j] = 0
            elif pp_board[8 * (5 - i) + j] == 'X':
                board[i, j] = 1
            elif pp_board[8 * (5 - i) + j] == 'O':
                board[i, j] = 2
    return board

def connect_four(
    board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> bool:
    state = False
    for i in range(6):
        if i<3:
            for j in range(4):
                if board[i, j] == board[i + 1, j + 1] == board[i + 2, j + 2] == board[i + 3, j + 3] == player:
                    state = True;
            for j in range(7):
                if board[i, j] == board[i + 1, j] == board[i + 2, j] == board[i + 3, j] == player:
                    state = True;
        elif i>=3:
            for j in range(4):
                if board[i, j] == board[i - 1, j - 1] == board[i - 2, j - 2] == board[i - 3, j - 3] == player:
                    state = True;
        for j in range(4):
            if board[i, j] == board[i, j + 1] == board[i, j + 2] == board[i, j + 3] == player:
                state = True;
    return state;

board = initialize_game_state()
board[0, 0] = 1 # lower left corner of board
print(pretty_print_board(board))
board = apply_player_action(board, 0, 2)
print(pretty_print_board(board))
board = apply_player_action(board, 5, 1)
print(pretty_print_board(board))

