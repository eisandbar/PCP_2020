import numpy as np

def test_intialize_game_state():
    from connectn.common import initialize_game_state

    ret = initialize_game_state()

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == np.int8
    assert ret.shape == (6, 7)
    assert np.all(ret == 0)


def test_pretty_print_board():
    from connectn.common import initialize_game_state
    from connectn.common import pretty_print_board

    board = initialize_game_state()
    board[0, 0] = 1
    board[0, 1] = 2
    ret = pretty_print_board(board)

    assert isinstance(ret, str)
    assert len(ret) == 8 * 8 # extra char for '\n'
    assert ret[5 * 8] == 'X'
    assert ret[5 * 8 + 1] == 'O'
    assert ret[5 * 8 + 2] == ' '


def test_apply_player_action():
    from connectn.common import initialize_game_state
    from connectn.common import apply_player_action

    board = initialize_game_state()
    ret = apply_player_action(board, 0, 1)

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == np.int8
    assert ret.shape == (6, 7)
    assert ret[0, 0] == 1


def test_string_to_board():
    from connectn.common import initialize_game_state
    from connectn.common import pretty_print_board
    from connectn.common import string_to_board

    board = initialize_game_state()
    pp_board = pretty_print_board(board)
    ret = string_to_board(pp_board)

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == np.int8
    assert ret.shape == (6, 7)
    assert np.all(ret == 0)

    board[0, 0] = 1
    board[0, 1] = 2
    pp_board = pretty_print_board(board)
    ret = string_to_board(pp_board)

    assert ret[0, 0] == 1
    assert ret[0, 1] == 2


def test_connect_four():
    from connectn.common import connect_four



test_intialize_game_state();
test_apply_player_action();
test_pretty_print_board();
test_string_to_board();