import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hides TF info and warnings
import numpy as np
import tensorflow as tf
from tensorflow import keras

from DeepQ.environment import Connect4Env
from DeepQ.model import num_actions, create_q_model
from DeepQ.agent_DQN import generate_move_DQN
from connectn.agent_minimax import generate_move

env = Connect4Env()

os.chdir("..")
os.chdir("models")
# model = create_q_model()
# model_target = create_q_model()
model = tf.keras.models.load_model('my_model.h5')
model_target = tf.keras.models.load_model('my_model.h5')

# Using Adam optimizer
optimizer = keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)
# Using huber loss for stability
loss_function = keras.losses.Huber()

# Configuration parameters for the whole setup
gamma = 1  # Discount factor for past rewards
epsilon = 0.5  # Epsilon greedy parameter
epsilon_min = 0.2  # Minimum epsilon greedy parameter
epsilon_interval = (epsilon_min/epsilon)  # Rate at which to reduce chance of random action being taken

# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
episode_reward_history = []
batch_size = 128  # Size of batch taken from replay buffer
max_memory_length = 10000  # Maximum replay length

# Initialize counters with 0
running_reward = 0
episode_count = 0
frame_count = 0
best_reward = -200

minimax_frames = 0  # Number of frames explored with minimax
epsilon_greedy_frames = 20000.0  # Number of frames for exploration
update_after_actions = 4  # Train the model after 4 actions
update_target_network = 1000  # How often to update the target network
save_model = 5000  # How often to save the model
episode_test = 1000  # How often to test model on mininmax


while True:  # Run until solved
    time_step = env.reset()  # Start new game
    state = time_step.observation
    episode_reward = 0

    while not time_step.is_last():  # Until game ends
        frame_count += 1

        # Use epsilon-greedy for exploration
        if epsilon > np.random.rand(1)[0]:
            # Take random action
            possible_actions = (state[5, :] == 0)
            action = np.random.choice(np.arange(num_actions)[possible_actions])

        # Sometimes we want to explore with our minimax agent
        elif frame_count < minimax_frames:
            board = np.where(state == -1, 2, state)
            action, _ = generate_move(board, player=np.int8(1))

        # Use DQN agent
        else:
            # Predict action Q-values from environment state
            state_tensor = tf.convert_to_tensor(time_step.observation)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            # Take best action
            possible_actions = (state[5, :] == 0)
            max_idx = np.argmax(action_probs.numpy()[0, possible_actions])
            action = np.arange(num_actions)[possible_actions][max_idx]

        # Decay probability of taking random action
        epsilon *= epsilon_interval ** float(1 / epsilon_greedy_frames)
        epsilon = max(epsilon, epsilon_min)

        # Apply the sampled action in our environment
        time_step = env.step(action)
        state_next = time_step.observation
        episode_reward += time_step.reward

        # Save actions and states in replay buffer
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        rewards_history.append(time_step.reward)
        state = state_next

        # Update every fourth frame and once batch size is over 128
        if frame_count % update_after_actions == 0 and len(action_history) > batch_size:

            # Get indices of samples for replay buffers
            indices = np.random.choice(range(len(action_history)), size=batch_size)

            # Using list comprehension to sample from replay buffer
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            rewards_sample = [rewards_history[i] for i in indices]
            action_sample = [action_history[i] for i in indices]

            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            future_rewards = model_target.predict(state_next_sample)
            # Q value = reward + discount factor * expected future reward
            updated_q_values = rewards_sample + gamma * tf.reduce_max(future_rewards, axis=1)

            # Create a mask so we only calculate loss on the updated Q-values
            masks = tf.one_hot(action_sample, num_actions)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = model(state_sample)

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if frame_count % update_target_network == 0:
            # update the the target network with new weights
            model_target.set_weights(model.get_weights())
            # Log details
            template = "running reward: {:.2f} at episode {}, frame count {}, epsilon {:.4f}"
            print(template.format(running_reward, episode_count, frame_count, epsilon))

        if frame_count % save_model == 0:
            # Save model
            model.save('my_model.h5')

        # Limit the state and reward history
        if len(rewards_history) > max_memory_length:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]

    episode_count += 1

    # Compare agent to minimax
    if episode_count % episode_test == 0:
        time_step = env.reset()
        state = time_step.observation
        episode_reward = 0
        while not time_step.is_last():
            action, _ = generate_move_DQN(state, player=np.int8(1))
            time_step = env.step(action)
            state = time_step.observation
            episode_reward += time_step.reward
        template = "episode reward: {:.2f} at episode {}"
        print(template.format(episode_reward, episode_count))
        if episode_reward > 0:  # Condition to consider the task solved
            # Save model
            model.save('solved_model.h5')
        if running_reward > best_reward:
            best_reward = running_reward
            model.save('best_model.h5')

    # Update running reward to check condition for solving
    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)

    # if running_reward > 0 and frame_count > minimax_frames:  # Condition to consider the task solved
    #     # Save model
    #     model.save('solved_model.h5')
    #
    #     print("Solved at episode {}!".format(episode_count))
    #     break

