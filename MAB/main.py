import numpy as np
import matplotlib.pyplot as plt

class BernoulliBandit:
    """多臂老虎机问题, 输入k代表拉杆的个数"""
    def __init__(self, K) -> None:
        self.probs = np.random.uniform(size=K) # 随机生成 K 个0~1的数，作为拉动每根拉杆的获奖概率
        self.best_idx = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_idx]
        self.K = K
    
    def step(self, k):
        """当玩家选择了k号拉杆后,根据该老虎机的k号拉杆获得奖励的概率，返回0或1"""
        if self.probs[k] > np.random.rand():
            return 1
        return 0
    
class Solver:
    """解决多臂老虎机问题算法的基本框架
    功能:
        1. 根据策略选择动作
        2. 根据动作获得奖励
        3. 更新期望奖励估计值
        4. 更新累计懊悔和计数
    """
    def __init__(self, bandit:BernoulliBandit) -> None:
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K) # 记录每根拉杆被拉动的次数
        self.regret = 0 # 记录当前步的累计懊悔值
        self.actions = [] # 维护一个列表, 记录每一步的动作
        self.regrets = [] # 维护一个列表, 记录每一步的累计懊悔值
    
    def update_regret(self, k):
        """更新累计懊悔值, k 为玩家选择的拉杆号码"""
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)
    
    def run_one_step(self) -> int:
        """返回当前策略选择的拉杆编号, 由每个具体的策略实现
        功能:
            1. 根据策略选择动作
            2. 根据动作获得奖励
            3. 更新期望奖励估计值

        例如, 一个随机策略的实现如下:
            return np.random.randint(self.bandit.K)
        """
        raise NotImplementedError
    
    def run(self, num_steps):
        """运行算法, num_steps 为运行的步数"""
        for _ in range(num_steps):
            k = self.run_one_step() # 得到当前步策略选择的拉杆编号
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)

class EpsilonGreedy(Solver):
    """Epsilon-Greedy 算法, 继承自 Solver 类
    
    属性:
        epsilon: 选择探索动作的概率
        init_prob: 初始化每根拉杆的期望奖励估计值
    
    """
    def __init__(self, bandit: BernoulliBandit, epsilon=0.01, init_prob=1.0) -> None:
        super().__init__(bandit)
        self.epsilon = epsilon
        self.estimates = np.ones(self.bandit.K) * init_prob # 初始化每根拉杆的期望奖励估计值

    def run_one_step(self) -> int:
        """返回当前策略选择的拉杆编号, 由每个具体的策略实现
        功能:
            1. 根据策略选择动作
            2. 根据动作获得奖励
            3. 更新期望奖励估计值
        """
        if np.random.rand() < self.epsilon: # 以 epsilon 的概率进行探索
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates) # 选择期望奖励最大的拉杆

        r = self.bandit.step(k) # 获得奖励
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k]) # 更新期望奖励估计值
        return k

class DecayingEpsilonGreedy(Solver):
    """ epsilon 随时间衰减的 Epsilon-Greedy 算法, 继承自 Solver 类"""
    def __init__(self, bandit: BernoulliBandit, init_prob=1.0) -> None:
        super().__init__(bandit)
        self.total_count = 0 # 记录总的步数
        self.estimates = np.ones(self.bandit.K) * init_prob # 初始化每根拉杆的期望奖励估计值

    def run_one_step(self) -> int:
        """返回当前策略选择的拉杆编号, 由每个具体的策略实现
        功能:
            1. 根据策略选择动作
            2. 根据动作获得奖励
            3. 更新期望奖励估计值
        """
        self.total_count += 1
        if np.random.rand() < 1.0 / self.total_count: #  epsilon 随时间衰减
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates) # 选择期望奖励最大的拉杆

        r = self.bandit.step(k) # 获得奖励
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k]) # 更新期望奖励估计值
        return k


def plot_results(solvers: list, solver_names):
    """生成累计懊悔随时间变化的图像,输入solvers是一个列表,每个元素是一种特定的策略
    solver_name 也是一个列表，存储每种策略的名称
    """
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel("Time steps")
    plt.ylabel("Cumulative Regret")
    plt.title("%d-armed bandit" % solvers[0].bandit.K)
    plt.legend()
    plt.show()

def BanditAnalyse():
    K = 10
    bandit = BernoulliBandit(K)
    print("随机生成了一个%d臂老虎机"% K)
    print("获奖概率最大的拉杆为%d号, 获奖概率为%.2f" % (bandit.best_idx, bandit.best_prob))



def EpsilonGreedyAnalyse():
    """分析 Epsilon-Greedy 算法的性能"""
    K = 10
    bandit_10_arm = BernoulliBandit(K)
    epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.01)
    epsilon_greedy_solver.run(5000)
    print('epsilon-greedy 的累积懊悔为: ',epsilon_greedy_solver.regret)
    plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])

    epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
    epsilon_greedy_solver_list = [EpsilonGreedy(bandit_10_arm, epsilon=eps) for eps in epsilons]
    epsilon_greedy_solver_names = ["epsilon=%.4f)" % eps for eps in epsilons]
    for solver in epsilon_greedy_solver_list:
        solver.run(5000)
    plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)

def DecayingEpsilonGreedyAnalyse():
    K = 10
    bandit_10_arm = BernoulliBandit(K)
    decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
    decaying_epsilon_greedy_solver.run(5000)
    print('decaying-epsilon-greedy 的累积懊悔为: ',decaying_epsilon_greedy_solver.regret)
    plot_results([decaying_epsilon_greedy_solver], ["DecayingEpsilonGreedy"])


if __name__ == "__main__":
    np.random.seed(0)
    #BanditAnalyse()
    #EpsilonGreedyAnalyse()
    DecayingEpsilonGreedyAnalyse()