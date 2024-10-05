import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, arms=10):
        self.rates = np.random.rand(arms) #各マシンの勝率

    def play(self, arm):
        rate = self.rates[arm]
        if rate > np.random.rand():
            return 1
        else:
            return 0
        
class Agent:
    def __init__(self, epsilon, action_size=10):
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)

    def update(self, action, reward):
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self):
        #基本はQsが一番大きい(期待値が一番高い)マシンを選ぶが、一定の確率(epsilon)で他のマシンを選ぶ
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)

steps = 1000
epsilon = 0.1

bandit = Bandit()
agent = Agent(epsilon)
total_reward = 0
total_rewards = []
rates = []

for step in range(steps):
    action = agent.get_action()
    reward = bandit.play(action)
    agent.update(action, reward)
    total_reward += reward

    total_rewards.append(total_reward)
    rates.append(total_reward / (step+1))

print(total_reward)

plt.ylabel('Total reward')
plt.xlabel('Steps')
plt.plot(total_rewards)
plt.show()

plt.ylabel('Rates')
plt.xlabel('Steps')
plt.plot(rates)
plt.show()


'''   
bandit = Bandit()
Qs = np.zeros(10)
ns = np.zeros(10)

for n in range(1000):
    action = np.random.randint(0, 10) #ランダムにマシンを選ぶ
    reward = bandit.play(action) #選んだマシンでプレイ

    #選んだマシンの評価
    ns[action] += 1
    Qs[action] += (reward - Qs[action]) / ns[action]
    #print(Qs)

print(Qs)
print(f'\n\n{bandit.rates}')

diff = sum((Qs-bandit.rates)**2)
print(diff)
'''

'''
for n in range(1, 1001):
    reward = bandit.play(0) #0番目のマシンをプレイ
    Q += (reward - Q) / n
    #print(Q)


print(Q)
print(f'\n\n{bandit.rates[0]}')
'''