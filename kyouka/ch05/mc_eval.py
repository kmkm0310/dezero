if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from collections import defaultdict
from common.gridworld import GridWorld

class RandomAgent:
    def __init__(self) :
        self.gamma = 0.9
        self.action_size = 4
        random_actions = {0:0.25, 1:0.25, 2:0.25, 3:0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.V = defaultdict(lambda: 0)
        self.cnts = defaultdict(lambda: 0)
        self.memory = []
        
        
    def get_action(self, state):
        action_prob = self.pi[state]
        actions = list(action_prob.keys())
        probs = list(action_prob.values())
        return np.random.choice(actions, p=probs)
    
    
    def add(self, state, action, reward):
        data = (state, action, reward)
        self.memory.append(data)
        
        
    def reset(self):
        self.memory.clear()
        
        
    def eval(self):
        G = 0
        for data in reversed(self.memory): # 逆向きにたどる
            state, action, reward = data
            G = self.gamma * G + reward
            self.cnts[state] += 1
            self.V[state] += (G - self.V[state]) / self.cnts[state]
        

env = GridWorld()
agent = RandomAgent()
episodes = 1000
cnts = []
for episode in range(episodes):
    state = env.reset()
    agent.reset()
    cnt = 0
    while True:
        cnt += 1
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        
        agent.add(state, action, reward)
        if done:
            agent.eval()
            cnts.append(cnt)
            break
        state = next_state
        
print(sum(cnts)/len(cnts))
env.render_v(agent.V)
