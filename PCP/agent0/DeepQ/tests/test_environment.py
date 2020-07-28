import numpy as np


def test_Connect4Env():
    from DeepQ.environment import Connect4Env
    environment = Connect4Env()
    for i in range(7):
        time_step = environment.reset()
        state = time_step.observation
        # Make sure the reset function returns a proper state
        assert isinstance(state, np.ndarray)
        assert state.dtype == np.int8
        assert state.shape == (6, 7)

        action = np.array(i, dtype=np.int8)
        time_step = environment.step(action)
        state = time_step.observation
        reward = time_step.reward

        # Make sure the step function returns a proper state
        assert isinstance(state, np.ndarray)
        assert state.dtype == np.int8
        assert state.shape == (6, 7)

        # Make sure the step function applies the player action
        assert state[0, i] == 1 or i == 3

        # Make sure the step function returns a proper reward
        assert isinstance(reward, np.ndarray)
        assert reward.dtype == np.float32
        assert reward == 10  # Transition move


test_Connect4Env()
