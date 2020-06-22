import numpy as np
import matplotlib.pyplot as plt

class Garnet:
    def __init__(self, num_states=10, num_actions=5, b_factor=2):
        self.P = np.zeros((num_states, num_actions, num_states))# transition matrix
        self.reward = np.random.uniform(size=(num_states, num_actions))
        self.state = 0
        self.num_states = num_states
        self.num_actions = num_actions

        # assign the probabilities to next_state(limited by the branching_factor)
        for s, a in np.ndindex((num_states, num_actions)):
            possible_states = np.random.choice(range(num_states), b_factor, replace=False)
            cumulative = np.concatenate(([0], np.sort(np.random.rand(b_factor - 1)), [1]), axis=0)
            for k in range(b_factor):
                self.P[s, a, possible_states[k]] += cumulative[k + 1] - cumulative[k]

    def step(self, action):
        # feedback of the env when the agent makes an action
        reward = self.reward[self.state, action]
        self.state = self._next_states(self.state, action)
        return reward, self.state

    def _next_states(self, state, action):
        # choose the next states of the givn (state, action)
        probs = self.P[state, action, :]
        return np.random.choice(np.arange(self.num_states), p=probs)

    def generate_seq(self, time):
        # generate a sequence according to the action ~ Uniform[0,1]
        seq = {'states':[], 'actions':[], 'rewards':[]}
        seq['states'].append(self.state)

        for i in range(time):
            a = np.random.choice(np.arange(self.num_actions))
            r, next_s = self.step(a)
            seq['states'].append(next_s)
            seq['actions'].append(a)
            seq['rewards'].append(r)
        
        seq['states'] = seq['states'][:-1]
        self.state = 0  
        return seq     


def monte_carlo(epoch, seq_len, df, env, V_bel):
    # First-visit MC prediction
    mse_mc = [] # mean square error in every epoch
    V_mc = np.random.rand(env.num_states) # initialize the V(s) randomly
    returns = [[] for _ in range(env.num_states)] 

    for _ in range(epoch):
        seq = env.generate_seq(seq_len) #generate sequence by choosing a~U[0,1]
        G = 0
        for i, (r, s) in enumerate(zip(reversed(seq['rewards']), reversed(seq['states']))):
            G = df * G + r # accmulative Gain
            if s not in seq['states'][:seq_len - 1 - i]:
                returns[s].append(G)
                V_mc[s] = np.average(returns[s])
        mse_mc.append(((V_bel - V_mc) ** 2).mean()) # MSE
    return mse_mc


def TD(epoch, seq_len, df, alpha, env, V_bel):
    env.state = 0
    mse_td = [] # mean square error in every epoch
    V_td = np.random.rand(env.num_states) # initialize the V(s) randomly
    for _ in range(epoch):
        s = 0
        for t in range(seq_len):
            a = np.random.choice(np.arange(env.num_actions))
            r, next_s = env.step(a)
            V_td[s] += alpha * (r + df * V_td[next_s] - V_td[s])
            s = next_s
        mse_td.append(((V_bel - V_td) ** 2).mean()) # MSE
    return mse_td


def TD_linear(epoch, seq_len, df, alpha, env, dim, V_bel):
    env.state = 0
    mse_td_linear = [] # mean square error in every epoch
    W = np.random.rand(dim)
    features = np.random.normal(0, 1, (env.num_states, dim)) 
    
    for _ in range(epoch):
        s = 0
        for t in range(seq_len):
            a = np.random.choice(np.arange(env.num_actions))
            r, next_s = env.step(a)
            grad = (r + df * np.dot(features[next_s].T, W) - np.dot(features[s].T, W)) * features[s]
            #grad = np.clip(grad, -1000000000000000000, 1000000000000000000)
            W += alpha * grad     
            s = next_s
        mse_td_linear.append(((V_bel - np.dot(features, W)) ** 2).mean()) # MSE
    return mse_td_linear


def main():
    num_states = 500
    num_actions = 5
    epoch=150
    df = 0.9
    alpha = 0.0005
    seq_len = 1000
    np.random.seed(1234)
    env = Garnet(num_states, num_actions, b_factor=100)

    # V(s) of bellman equation
    R_ = np.sum(env.reward * 1 / num_actions, axis=1)
    P_ = np.sum(env.P * 1 / num_actions, axis=1)
    
    V_bel = np.linalg.inv((np.identity(num_states) - df * P_)).dot(R_)

    """     # MC
    mse_mc = monte_carlo(epoch=epoch, seq_len=seq_len, df=df, env=env, V_bel=V_bel)

    # TD
    mse_td = TD(epoch=epoch, seq_len=seq_len, df=df, alpha=alpha, env=env, V_bel=V_bel)

    plt.plot(range(epoch), mse_td, label="TD")
    plt.plot(range(epoch), mse_mc, label="MC")
    plt.legend()
    plt.title("Compare V(s) ")
    plt.ylabel("Mean Square Error")
    plt.xlabel("Number of epoches")
    plt.show() 
    """

    # TD linear
    mse_td_linear10 = TD_linear(epoch=epoch, seq_len=seq_len, df=df, alpha=alpha, env=env, dim=100, V_bel=V_bel)
    mse_td_linear20 = TD_linear(epoch=epoch, seq_len=seq_len, df=df, alpha=alpha, env=env, dim=200, V_bel=V_bel)
    mse_td_linear30 = TD_linear(epoch=epoch, seq_len=seq_len, df=df, alpha=alpha, env=env, dim=300, V_bel=V_bel)
    mse_td_linear40 = TD_linear(epoch=epoch, seq_len=seq_len, df=df, alpha=alpha, env=env, dim=400, V_bel=V_bel)
    mse_td_linear50 = TD_linear(epoch=epoch, seq_len=seq_len, df=df, alpha=alpha, env=env, dim=500, V_bel=V_bel)
    mse_td_linear55 = TD_linear(epoch=epoch, seq_len=seq_len, df=df, alpha=alpha, env=env, dim=550, V_bel=V_bel)
    

    plt.plot(range(epoch), mse_td_linear10, label="TD_linear100")
    plt.plot(range(epoch), mse_td_linear20, label="TD_linear200")
    plt.plot(range(epoch), mse_td_linear30, label="TD_linear300")
    plt.plot(range(epoch), mse_td_linear40, label="TD_linear400")
    plt.plot(range(epoch), mse_td_linear50, label="TD_linear500")
    plt.plot(range(epoch), mse_td_linear50, label="TD_linear550")
    plt.legend()
    plt.title("V(s)(bellman) vs V(s) TD_linear")
    plt.ylabel("Mean Square Error")
    plt.xlabel("Number of epoches(seq_length=1000)")
    plt.show() 


if __name__ == "__main__":
    main()
    