import numpy as np


def test_generate_move():
    from connectn.agent_minimax import generate_move

    board1 = np.zeros((6, 7), dtype=np.int8)
    ret, saved_state = generate_move(board1.copy(), player=1)

    assert isinstance(ret, np.int8)
    assert 0 <= ret <= 6  # Move is within range

    board2 = np.ones((6, 7), dtype=np.int8)
    board2[5, 5] = 0
    ret, saved_state = generate_move(board2.copy(), player=1)

    assert isinstance(ret, np.int8)
    assert ret == 5  # Only empty slot

    board1[0, 4] = board1[0, 6] = 2
    ret, saved_state = generate_move(board1.copy(), player=1)

    assert isinstance(ret, np.int8)
    assert ret == 5  # Best move


def test_alpha_beta():
    from connectn.agent_minimax import alpha_beta

    board = np.zeros((6, 7), dtype=np.int8)
    ret = alpha_beta(board, 3, 0, np.NINF, np.inf, 1, True)

    assert isinstance(ret, float)
    assert ret == 0

    ret = alpha_beta(board, 3, 2, np.NINF, np.inf, 1, True)

    assert isinstance(ret, float)
    assert ret > 0  # With 3 pieces on the board best value is always positive

    ret = alpha_beta(board, 3, 3, np.NINF, np.inf, 1, True)

    assert isinstance(ret, float)
    assert ret < 0  # With 4 pieces on the board best value is always negative





def test_heuristic():
    from connectn.agent_minimax import heuristic

    board = np.zeros((6, 7), dtype=np.int8)
    ret = heuristic(board, 1)

    assert isinstance(ret, float)
    assert ret == 0

    board[0:2, 0:2] = 1
    board[0, 2] = 1
    ret = heuristic(board, 1)

    assert isinstance(ret, float)
    assert ret == 108  # 8 pairs and one triple

    board[1, 2] = board[2, 1] = board[0, 3] = 2
    ret = heuristic(board, 1)

    assert isinstance(ret, float)
    assert ret == 108 - 304  # Opponent has 2 pairs and one triple


test_generate_move()
test_alpha_beta()
test_heuristic()