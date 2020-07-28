import numpy as np
BoardPiece = np.int8


def test_generate_move_DQN():
    from DeepQ.agent_DQN import generate_move_DQN
    from connectn.common import SavedState
    for player in np.array([1, 2], dtype=BoardPiece):
        board1 = np.zeros((6, 7), dtype=np.int8)
        ret, saved_state = generate_move_DQN(board1.copy(), player=player)

        # Make sure that the action is of the proper type
        assert isinstance(ret, np.int8)
        assert 0 <= ret <= 6  # Move is within range

        # Makes sure that the saved state is of the proper type
        assert isinstance(saved_state, SavedState) or saved_state == None

        # No matter the model if there is only one move available, the agent has to pick it
        for i in range(7):
            board2 = np.ones((6, 7), dtype=np.int8)
            board2[5, i] = 0
            ret, saved_state = generate_move_DQN(board2.copy(), player=player)

            assert isinstance(ret, np.int8)
            assert ret == i  # Only empty slot


test_generate_move_DQN()
