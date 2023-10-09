import numpy as np

np.random.seed(0)

# 定义状态转移概率矩阵 P
arr = [
    [0.9, 0.1, 0., 0., 0., 0.],
    [0.5, 0., 0.5, 0., 0., 0.],
    [0., 0., 0., 0.6, 0., 0.4],
    [0., 0., 0., 0., 0.3, 0.7],
    [0., 0.2, 0.3, 0.5, 0., 0.],
    [0., 0., 0., 0., 0., 1.],
]
P = np.array(arr, dtype=np.float32)
rewards = [-1, -2, -2, 10, 1, 0]
gamma = 0.5

# 给定一条序列， 计算从某个起始状态开始到序列终止状态得到的回报
def compute_return(start_index, chain, gamma):
    G = 0
    for i in reversed(range(start_index, len(chain))):
        G = gamma * G + rewards[chain[i] - 1]
    return G

# 一个状态序列 s1-s2-s3-s6
chain = [1, 2, 3, 6]
start_index = 0
G = compute_return(start_index, chain, gamma)
print("根据本次计算得到的回报为: %s."%G)

def compute(P, rewards, gamma, state_num):
    """求贝尔曼方程的解析解,state_num 是 MRP 的状态数"""
    rewards = np.array(rewards, dtype=np.float32).reshape((-1, 1))
    values = np.dot(np.linalg.inv(np.eye(state_num, state_num) - gamma * P), rewards)
    return values

V = compute(P, rewards, gamma, 6)
print("MRP中每个状态价值分别为\n", V)