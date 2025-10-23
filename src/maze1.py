import numpy as np
import matplotlib.pyplot as plt
import random
import pickle

# ------------------ Maze Environment ------------------
class MazeEnvironment:
    def __init__(self, N, M):
        self.N, self.M = N, M
        self.grid = np.zeros((N, M))   # 0: empty, 1: wall, 2: goal
        self.start = (0,0)
        self.goal = (N-1, M-1)
        self.grid[self.goal] = 2

        # Randomly place walls (10% of cells)
        num_walls = int(0.1 * N * M)
        for _ in range(num_walls):
            i, j = np.random.randint(0,N), np.random.randint(0,M)
            if (i,j) != self.start and (i,j) != self.goal:
                self.grid[i,j] = 1

    def step(self, state, action):
        x, y = state
        if action == 0: nx, ny = x-1, y   # Up
        elif action == 1: nx, ny = x+1, y # Down
        elif action == 2: nx, ny = x, y-1 # Left
        elif action == 3: nx, ny = x, y+1 # Right

        # Check bounds
        if nx<0 or nx>=self.N or ny<0 or ny>=self.M:
            return state, -10, False

        # Check wall
        if self.grid[nx,ny] == 1:
            return state, -10, False

        # Check goal
        if self.grid[nx,ny] == 2:
            return (nx,ny), 100, True

        return (nx,ny), -1, False

    def reset(self):
        return self.start

# ------------------ Q-Learning Agent ------------------
class QLearningAgent:
    def __init__(self, env):
        self.env = env
        self.q_table = np.random.uniform(-0.1,0.1, (env.N*env.M, 4))
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.9

    def state_to_index(self, state):
        x,y = state
        return x*self.env.M + y

    def choose_action(self, state, episode):
        # Custom exploration rule: every 5th episode, take 3 random actions
        if episode % 5 == 0:
            return np.random.randint(0,4)

        if np.random.rand() < self.epsilon:
            return np.random.randint(0,4)
        idx = self.state_to_index(state)
        return np.argmax(self.q_table[idx])

    def update(self, state, action, reward, next_state):
        s_idx = self.state_to_index(state)
        ns_idx = self.state_to_index(next_state)
        self.q_table[s_idx, action] = self.q_table[s_idx, action] + \
            self.alpha * (reward + self.gamma * np.max(self.q_table[ns_idx]) - self.q_table[s_idx, action])

# ------------------ Training ------------------
def train():
    N, M = 6, 6
    env = MazeEnvironment(N,M)
    agent = QLearningAgent(env)
    episodes = 1000
    rewards = []
    trajectories = []  # list to store good trajectories

    for ep in range(episodes):
        state = env.reset()
        traj = []  # store this episode
        total_reward = 0
        for step in range(1000):
            action = agent.choose_action(state, ep)
            next_state, reward, done = env.step(state, action)
            traj.append((state, action, reward, next_state))
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            if done:
                if len(traj) < 200:  # only keep fast/efficient episodes
                    trajectories.append(traj)
                break
        rewards.append(total_reward)
        # Linear epsilon decay
        agent.epsilon = max(0.1, 0.9 - 0.8*(ep/episodes))

        if ep % 100 == 0:
            print(f"Episode {ep}, Total reward: {total_reward}")

    # Save trajectories to file
    with open("trajectories.pkl", "wb") as f:
        pickle.dump(trajectories, f)

    print(f"Saved {len(trajectories)} good trajectories.")
    
    # Plot rewards without blocking
    import matplotlib.pyplot as plt
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Milestone 1 Training Rewards')
    plt.savefig("milestone1_rewards.png")  # save the plot
    plt.close()                             # close the figure to prevent blocking
    print("Training plot saved as milestone1_rewards.png")


    # Evaluation
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
