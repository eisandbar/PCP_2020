import numpy as np
import tensorflow as tf


def test_create_q_model():
    from DeepQ.model import create_q_model
    model = create_q_model()

    # Creating a tensor of our board to feed the model
    state = np.zeros((6, 7), dtype=np.int8)
    state_tensor = tf.convert_to_tensor(state)
    state_tensor = tf.expand_dims(state_tensor, 0)

    # Calculating probabilities
    action_probs = model(state_tensor, training=False)

    # Make sure probabilities have proper type and shape
    assert isinstance(action_probs, tf.Tensor)
    assert action_probs.dtype == np.float32
    assert action_probs.shape == (1, 7)


test_create_q_model()

