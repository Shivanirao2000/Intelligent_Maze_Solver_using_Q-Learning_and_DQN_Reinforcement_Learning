import numpy as np
import matplotlib.pyplot as plt
import pickle
import random

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
        if action==0: nx,ny = x-1,y      # Up
        elif action==1: nx,ny = x+1,y    # Down
        elif action==2: nx,ny = x,y-1    # Left
        elif action==3: nx,ny = x,y+1    # Right

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

# ------------------ Q-Learning Agent ------------------
class QLearningAgent:
    def __init__(self, env):
        self.env = env
        self.q_table = np.random.uniform(-0.1,0.1,(env.N*env.M,4))
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.5  # start lower to leverage historical data

    def state_to_index(self,state):
        x,y = state
        return x*self.env.M + y

    def choose_action(self, state, episode):
        if episode % 5 == 0:  # custom exploration rule
            return np.random.randint(0,4)
        if np.random.rand() < self.epsilon:
            return np.random.randint(0,4)
        idx = self.state_to_index(state)
        return np.argmax(self.q_table[idx])

    def update(self, state, action, reward, next_state):
        s_idx = self.state_to_index(state)
        ns_idx = self.state_to_index(next_state)
        self.q_table[s_idx, action] += self.alpha * (reward + self.gamma*np.max(self.q_table[ns_idx]) - self.q_table[s_idx, action])

# ------------------ Load Historical Data ------------------
with open("trajectories.pkl","rb") as f:
    trajectories = pickle.load(f)
print(f"Loaded {len(trajectories)} historical trajectories.")

# ------------------ Training ------------------
def train():
    N, M = 6, 6
    env = MazeEnvironment(N,M)
    agent = QLearningAgent(env)
    rewards = []

    # Step 1: Initialize Q-table using historical data
    for traj in trajectories:
        agent.alpha = 0.15  # higher learning rate for historical states
        for state, action, reward, next_state in traj:
            agent.update(state, action, reward, next_state)
    agent.alpha = 0.1  # reset for self-play

    # Step 2: Continue training with self-play
    episodes = 500
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        for step in range(1000):
            action = agent.choose_action(state, ep)
            next_state, reward, done = env.step(state, action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            if done:
                break
        rewards.append(total_reward)
        agent.epsilon = max(0.1, agent.epsilon - 0.8/episodes)
        if ep % 100 == 0:
            print(f"Episode {ep}, Total reward: {total_reward}")

    # Step 3: Plot rewards
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Milestone 2 Training Rewards')
    plt.savefig("milestone2_rewards.png")
    plt.close()
    print("Training plot saved as milestone2_rewards.png")

    # Step 4: Evaluation
    success = 0
    steps_list = []
    for _ in range(100):
        state = env.reset()
        steps = 0
        for _ in range(1000):
            action = np.argmax(agent.q_table[agent.state_to_index(state)])
            state, reward, done = env.step(state, action)
            steps += 1
            if done:
                success += 1
                steps_list.append(steps)
                break
    print("Success rate:", success/100)
    print("Average steps:", np.mean(steps_list))

if __name__=="__main__":
    train()
