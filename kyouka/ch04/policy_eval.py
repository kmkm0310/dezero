if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from collections import defaultdict
from common.gridworld import GridWorld

def eval_onestep(pi, V, env, gamma=0.9):
    for state in env.states(): # 1 各状態へのアクセス
        if state == env.goal_state: # 2 ゴールの価値観数は常に0
            V[state] = 0
            continue

        action_probs = pi[state]
        new_V = 0

        # 3 各行動へのアクセス
        for action, action_prob in action_probs.items():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            new_V += action_prob * (r + gamma * V[next_state])
        V[state] = new_V
    return V

def policy_eval(pi, V, env, gamma, threshold=0.001):
    while True:
        old_V = V.copy() # 更新前の価値数
        V = eval_onestep(pi, V, env, gamma)

        # 更新された量の最大値を求める
        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t

        # 閾値との比較
        if delta < threshold:
            break
    return V
        

if __name__ == '__main__':
    env = GridWorld()
    gamma = 0.9

    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    V = defaultdict(lambda: 0)

    V = policy_eval(pi, V, env, gamma)
    env.render_v(V, pi)