import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt

# ------------------ Maze Environment ------------------
class MazeEnvironment:
    def __init__(self, N, M):
        self.N, self.M = N, M
        self.grid = np.zeros((N, M))
        self.start = (0,0)
        self.goal = (N-1, M-1)
        self.grid[self.goal] = 2
        num_walls = int(0.1 * N * M)
        for _ in range(num_walls):
            i,j = np.random.randint(0,N), np.random.randint(0,M)
            if (i,j) != self.start and (i,j) != self.goal:
                self.grid[i,j] = 1

    def step(self, state, action):
        x,y = state
        if action==0: nx,ny = x-1,y
        elif action==1: nx,ny = x+1,y
        elif action==2: nx,ny = x,y-1
        elif action==3: nx,ny = x,y+1

        # Out of bounds
        if nx<0 or nx>=self.N or ny<0 or ny>=self.M:
            return state, -10, False
        # Wall
        if self.grid[nx,ny]==1:
            return state, -10, False
        # Goal
        if self.grid[nx,ny]==2:
            return (nx,ny), 100, True
        return (nx,ny), -1, False

    def reset(self):
        return self.start

# ------------------ Deep Q-Network ------------------
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# ------------------ DQN Agent ------------------
class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.input_size = env.N * env.M
        self.output_size = 4
        self.model = DQN(self.input_size, self.output_size)
        self.target_model = DQN(self.input_size, self.output_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = deque(maxlen=1000)
        self.gamma = 0.9
        self.epsilon = 0.9

    def state_to_tensor(self, state):
        x = np.zeros(self.input_size, dtype=np.float32)
        idx = state[0]*self.env.M + state[1]
        x[idx] = 1.0
        return torch.from_numpy(x)

    def choose_action(self, state, episode):
        if episode % 10 == 0:  # custom exploration: freeze target
            return random.randint(0,3)
        if random.random() < self.epsilon:
            return random.randint(0,3)
        with torch.no_grad():
            q_vals = self.model(self.state_to_tensor(state))
            return torch.argmax(q_vals).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self, batch_size):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_tensor = torch.stack([self.state_to_tensor(s) for s in states])
        next_states_tensor = torch.stack([self.state_to_tensor(s) for s in next_states])
        actions_tensor = torch.tensor(actions)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        dones_tensor = torch.tensor(dones, dtype=torch.float32)

        q_values = self.model(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q = self.target_model(next_states_tensor).max(1)[0]
        target = rewards_tensor + self.gamma * next_q * (1 - dones_tensor)

        loss = nn.SmoothL1Loss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# ------------------ Training ------------------
def train():
    N, M = 6, 6
    env = MazeEnvironment(N,M)
    agent = DQNAgent(env)
    episodes = 1000
    batch_size = 32
    rewards_history = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step_count = 0

        while not done and step_count < 1000:
            action = agent.choose_action(state, ep)
            next_state, reward, done = env.step(state, action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update(batch_size)
            state = next_state
            total_reward += reward
            step_count += 1

        # Decay epsilon
        agent.epsilon = max(0.1, agent.epsilon - 0.8/episodes)
        rewards_history.append(total_reward)

        # Freeze target network every 10th episode
        if ep % 10 == 0:
            agent.target_model.load_state_dict(agent.model.state_dict())

        if ep % 100 == 0:
            print(f"Episode {ep}, Total reward: {total_reward}")

    # Plot rewards
    plt.plot(rewards_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Milestone 3 DQN Training Rewards")
    plt.savefig("milestone3_rewards.png")
    plt.close()
    print("Training plot saved as milestone3_rewards.png")

    # ------------------ Evaluation ------------------
    success = 0
    steps_list = []
    for _ in range(100):
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < 1000:
            with torch.no_grad():
                q_vals = agent.model(agent.state_to_tensor(state))
                action = torch.argmax(q_vals).item()
            state, reward, done = env.step(state, action)
            steps += 1
        if done:
            success += 1
            steps_list.append(steps)
    print("Success rate:", success/100)
    print("Average steps:", np.mean(steps_list))

if __name__=="__main__":
    train()
