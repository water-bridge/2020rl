import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

class Garnet:
    def __init__(self, num_states=10, num_actions=5, b_factor=2):
        self.P = np.zeros((num_states, num_actions, num_states))  # transition matrix
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
        seq = {'states': [], 'actions': [], 'rewards': []}
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
    mse_mc = []  # mean square error in every epoch
    V_mc = np.random.rand(env.num_states)  # initialize the V(s) randomly
    returns = [[] for _ in range(env.num_states)] 

    for _ in range(epoch):
        seq = env.generate_seq(seq_len)  # generate sequence by choosing a~U[0,1]
        G = 0
        for i, (r, s) in enumerate(zip(reversed(seq['rewards']), reversed(seq['states']))):
            G = df * G + r  # accmulative Gain
            if s not in seq['states'][:seq_len - 1 - i]:
                returns[s].append(G)
                V_mc[s] = np.average(returns[s])
        mse_mc.append(((V_bel - V_mc) ** 2).mean())  # MSE
    return mse_mc


def TD(epoch, seq_len, df, alpha, env):
    V_bel = V_bellman()
    env.state = 0
    mse_td = []  # mean square error in every epoch
    V_td = np.random.rand(env.num_states)  # initialize the V(s) randomly
    for _ in range(epoch):
        s = 0
        for t in range(seq_len):
            a = np.random.choice(np.arange(env.num_actions))
            r, next_s = env.step(a)
            V_td[s] += alpha * (r + df * V_td[next_s] - V_td[s])
            s = next_s
        mse_td.append(((V_bel - V_td) ** 2).mean())  # MSE
    return mse_td


def TD_linear(epoch, seq_len, df, alpha, env, dim):
    V_bel = V_bellman(env)
    env.state = 0
    mse_td_linear = []  # mean square error in every epoch
    W = np.random.rand(dim)
    features = np.random.normal(0, 1, (env.num_states, dim)) 
    
    for _ in range(epoch):
        s = 0
        for t in range(seq_len):
            a = np.random.choice(np.arange(env.num_actions))
            r, next_s = env.step(a)
            grad = (r + df * np.dot(features[next_s].T, W) - np.dot(features[s].T, W)) * features[s]
            # grad = np.clip(grad, -1000000000000000000, 1000000000000000000)
            W += alpha * grad     
            s = next_s
        mse_td_linear.append(((V_bel - np.dot(features, W)) ** 2).mean())  # MSE
    return mse_td_linear


def TD_neural(seq_len, df, learning_rate, env, net, averaging=True, B=1000):
    # only one epoch
    env.state = 0
    running_loss = 0
    if averaging:
        path = "proj and avg"
    else:
        path = "proj"
    # initialize parameters
    s = torch.zeros(env.num_states)
    s[0] = 1
    w = nn.Parameter(torch.rand(net.m, net.dim)) 
    w_ = net.w0.detach().clone()  # barW for averaging

    # initialize optimizer and logger
    mse = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    optimizer_proj = optim.SGD([w], lr=learning_rate)
    writer = SummaryWriter(log_dir='runs/{}'.format(path))
    
    for i in range(seq_len):
        # sample (s, a, r, s')
        a = np.random.choice(np.arange(env.num_actions))
        r, next_s = env.step(a)
        next_s_t = torch.zeros(env.num_states)
        next_s_t[next_s] = 1
        expected = r + df * net(next_s_t).detach()  # detach the variable from computational graph

        # TD update
        optimizer.zero_grad()
        loss = mse(net(s), expected)
        loss.backward()
        optimizer.step()

        for _ in range(10000):
            # minimize ||W-bar/W(t+1)||
            optimizer_proj.zero_grad()
            loss_proj = torch.norm(w - net.wr.weight.data.detach(), p="fro")
            loss_proj.backward()
            optimizer_proj.step()
            if loss_proj <= 0.001:
                break

        # projection 
        w.clamp(max=B)
        net.wr.weight.data = w.data.detach()  # data attribute does not affect the computational graph

        # averaging
        if averaging:
            w_ = ((i + 1) / (i + 2)) * w_ + (1 / (i + 2)) * w.data.detach()
            
        s = next_s_t

        # record trainnig loss  
        running_loss += loss.item()
        if i % 1000 == 999:
            # ...log the running loss
            print("loss:", running_loss / 1000, " seq:", i + 1)
            writer.add_scalar('training loss with {}(m = {})'.format(path, net.m), running_loss / 1000, i + 1)
            running_loss = 0.0

    if averaging:
        net.wr.weight.data = w_.data
    torch.save(net.state_dict(), "weights/m({})={}".format(path, net.m))
    writer.close()


class Net(nn.Module):
    def __init__(self, dim, m):
        super(Net, self).__init__()
        self.dim = dim
        self.m = m
        self.wr = nn.Linear(dim, m)
        self.br = nn.Linear(m, 1)
        self._initialize_weights()
        self.w0 = self.wr.weight.data.clone().detach()

    def forward(self, x):
        x = F.relu(self.wr(x))
        x = self.br(x) / np.sqrt(self.m)
        return x

    def _initialize_weights(self):
        self.wr.weight.data.normal_(0.0, 1 / self.dim)
        self.br.weight.data.uniform_(-1, 1)


def train(num_states, num_actions, seq_len, df, learning_rate, m, env, averaging=True):
    net = Net(num_states, m)
    TD_neural(seq_len, df, learning_rate, env, net, averaging)
    print("finish")

def V_bellman(env):
    # calculate V(s) of bellman equation
    R_ = np.sum(env.reward * 1 / env.num_actions, axis=1)
    P_ = np.sum(env.P * 1 / env.num_actions, axis=1)
    V_bel = np.linalg.inv((np.identity(env.num_states) - df * P_)).dot(R_)
    return V_bel

def evaluate(num_states, num_actions, M, df, env, path):
    # V(s) of bellman equation
    V_bel = V_bellman(env)

    mse = []
    mse_linear = []
    for m in M:
        # load weights
        net = Net(num_states, m)
        net.load_state_dict(torch.load("weights/m({})={}".format(path, m)))
        with torch.no_grad():
            all_s = torch.zeros(num_states, num_states)
            for i in range(num_states):
                all_s[i][i] = 1
            V_neural = net(all_s).squeeze().numpy()
        
        assert V_bel.shape == V_neural.shape
        mse.append(((V_bel - V_neural) ** 2).mean())
        #mse_linear.append(TD_linear(epoch, seq_len, df, alpha, env, m))

    plt.plot(M, mse, label='Neural')
    #plt.plot(M, mse_linear, label='Linear')
    plt.title("MSE of the value function")
    plt.ylabel("Mean Square Error")
    plt.xlabel("m")
    plt.show() 

if __name__ == "__main__":
    np.random.seed(4321)
    torch.manual_seed(4321)
    epoch=1
    alpha = 0.0005
    num_states = 500
    num_actions = 5
    seq_len = 350000
    df = 0.9
    learning_rate = 0.005
    M = [5, 10, 50, 100, 200, 300, 400, 500, 600, 700]
    #M = [400, 500]
    env = Garnet(num_states, num_actions, b_factor=100)
    """ for m in M:
        train(num_states, num_actions, seq_len, df, learning_rate, m, env, averaging=False) """

    evaluate(num_states, num_actions, M, df, env, path="proj")