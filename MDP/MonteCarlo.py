import numpy as np
import matplotlib.pyplot as plt

S = ["s1", "s2", "s3", "s4", "s5"] # 状态集合
A = ["保持s1", "前往s1", "前往s2", "前往s3", "前往s4", "前往s5", "概率前往"] # 动作集合
#状态转移函数
P = {
    "s1-保持s1-s1": 1.0, "s1-前往s2-s2":1.0,
    "s2-前往s1-s1": 1.0, "s2-前往s3-s3":1.0,
    "s3-前往s4-s4": 1.0, "s3-前往s5-s5":1.0,
    "s4-前往s5-s5": 1.0, "s4-概率前往-s2":0.2,
    "s4-概率前往-s3":0.4, "s4-概率前往-s4":0.4,
}
# 奖励函数
R = {
    "s1-保持s1": -1, "s1-前往s2": 0,
    "s2-前往s1": -1, "s2-前往s3": -2,
    "s3-前往s4": -2, "s3-前往s5": 0,
    "s4-前往s5": 10, "s4-概率前往":1
}
gamma = 0.5
MDP = (S, A, P, R, gamma)


# 策略1, 随机策略
Pi_1 = {
    "s1-保持s1": 0.5, "s1-前往s2": 0.5,
    "s2-前往s1": 0.5, "s2-前往s3": 0.5,
    "s3-前往s4": 0.5, "s3-前往s5": 0.5,
    "s4-前往s5": 0.5, "s4-概率前往": 0.5
}

Pi_2 = {
    "s1-保持s1": 0.6, "s1-前往s2": 0.4,
    "s2-前往s1": 0.3, "s2-前往s3": 0.7,
    "s3-前往s4": 0.5, "s3-前往s5": 0.5,
    "s4-前往s5": 0.1, "s4-概率前往": 0.9
}

# 把输入的两个字符串通过 "-" 连接, 便于使用上述定义的P、R变量
def join(str1, str2):
    return str1 + "-" + str2

def compute(P, rewards, gamma, state_num):
    """求贝尔曼方程的解析解,state_num 是 MRP 的状态数"""
    rewards = np.array(rewards, dtype=np.float32).reshape((-1, 1))
    values = np.dot(np.linalg.inv(np.eye(state_num, state_num) - gamma * P), rewards)
    return values


def sample(MDP, Pi, timestep_max, number):
    """采样策略, 策略Pi, 限制最长时间步,总共采样序列数number"""
    S, A, P, R, gamma = MDP
    # 采样序列集合
    episodes = []
    for _ in range(number):
        episode = []
        timestep = 0
        s = S[np.random.randint(4)] # 随机选一个除 s5 以外的状态 s 作为起点
        # 当前状态为终止状态或者时间步太长时, 一次采样结束
        while s != "s5" and timestep <= timestep_max:
            timestep += 1
            rand, temp = np.random.rand(), 0
            # 在状态 s 下根据策略选择动作
            for a_opt in A:
                temp += Pi.get(join(s, a_opt), 0)
                if temp > rand:
                    a = a_opt
                    r = R.get(join(s, a), 0)
                    break
            rand, temp = np.random.rand(), 0
            # 根据状态转移概率得到下一个状态 s_next
            for s_opt in S:
                temp += P.get(join(join(s, a), s_opt), 0)
                if temp > rand:
                    s_next = s_opt
                    break
            episode.append((s, a, r, s_next)) # 把 (s,a,r,s_next) 加入采样序列
            s = s_next
        episodes.append(episode)
    return episodes

# 对所有采样序列计算所有状态的价值
def MC(episodes, V:dict, N, gamma, V_label):
    loss = []
    for episode in episodes:
        G = 0
        for i in range(len(episode)-1, -1, -1): # 一个序列从后往前计算
            (s, a, r, s_next) = episode[i]
            G = r + gamma * G
            N[s] = N[s] + 1
            V[s] = V[s] + (G - V[s]) / N[s]
            
        # 计算差距
        list = []
        list.extend(V.values())
        list = np.array(list)
        list = list.reshape(V_label.shape)
        
        l = np.sqrt(np.sum((list - V_label) ** 2))
        loss.append(l)
    return loss


"""MRP解析解来求解MDP的状态价值函数"""
P_from_mdp_to_mrp = [
    [0.5, 0.5, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.5, 0.5],
    [0.0, 0.1, 0.2, 0.2, 0.5],
    [0.0, 0.0, 0.0, 0.0, 1.0]
]
P_from_mdp_to_mrp = np.array(P_from_mdp_to_mrp)
R_from_mdp_to_mrp = [-0.5, -1.5, -1.0, 5.5, 0]
V_label = compute(P_from_mdp_to_mrp, R_from_mdp_to_mrp, gamma, 5) # MRP解析解得到的真实的状态价值函数
V_label = V_label.reshape(-1)

timestep_max = 20 # 限制最长时间步
# 采样1000次
episodes = sample(MDP, Pi_1, timestep_max, 2000)
gamma = 0.5
V = {"s1":0, "s2":0, "s3":0, "s4":0, "s5":0} # 状态价值
N = {"s1":0, "s2":0, "s3":0, "s4":0, "s5":0} # 状态出现次数
loss = MC(episodes, V, N, gamma, V_label)


print("使用蒙特卡洛方法计算状态价值为:\n", V)
print("真实值为", V_label)



# 画出每次采样后的状态价值与真实状态价值的差距
time_list = range(len(episodes))
plt.plot(time_list, loss)
plt.xlabel("Sample steps")
plt.ylabel("Loss")
plt.title("Monte Carlo Loss")
plt.show()