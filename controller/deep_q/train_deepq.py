
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
from controller.deep_q.ReplayBuffer import ReplayBuffer
from controller.deep_q.deepq_modal import DQN

learning_rate = 0.001
gamma = 0.99
batch_size = 64
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
target_update = 10


# Training function
def train_dqn(env, num_episodes=1000, input_dim=4, hidden_dim=64, output_dim=2, buffer_size=10000):
    # Initialize networks and optimizer
    policy_net = DQN(input_dim, hidden_dim, output_dim)
    target_net = DQN(input_dim, hidden_dim, output_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

    replay_buffer = ReplayBuffer(buffer_size)
    epsilon = epsilon_start

    for episode in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        total_reward = 0

        while True:
            # Epsilon-greedy action selection
            if random.random() > epsilon:
                with torch.no_grad():
                    action = policy_net(state).argmax(dim=1).item()
            else:
                action = env.action_space.sample()

            # Execute action
            next_state, reward, done, _ = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # Sample a batch from replay buffer and train
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                states = torch.cat(states)
                actions = torch.tensor(actions).unsqueeze(1)
                rewards = torch.tensor(rewards)
                next_states = torch.cat(next_states)
                dones = torch.tensor(dones, dtype=torch.float32)

                # Compute Q-values and targets
                q_values = policy_net(states).gather(1, actions).squeeze()
                next_q_values = target_net(next_states).max(1)[0]
                target_q_values = rewards + gamma * next_q_values * (1 - dones)

                # Compute loss and update weights
                loss = nn.MSELoss()(q_values, target_q_values.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        # Update epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Update target network
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

    return policy_net
