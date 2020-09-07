import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    def stationary(self):
        Q = np.sum(self.P, axis=1)/5

        evals, evecs = np.linalg.eig(Q.T)
        evec1 = evecs[:,np.isclose(evals, 1)]

        #Since np.isclose will return an array, we've indexed with an array
        #so we still have our 2nd axis.  Get rid of it, since it's only size 1.
        evec1 = evec1[:,0]

        stationary = evec1 / evec1.sum()

        #eigs finds complex eigenvalues and eigenvectors, so you'll want the real part.
        stationary = stationary.real
        return stationary


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
    V_bel = V_bellman(env)
    pi = env.stationary()

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
    pi = env.stationary()

    env.state = 0
    msbe = []  # mean square error in every epoch
    W = np.random.rand(dim)
    features = np.random.normal(1, 1, (env.num_states, dim)) 
    
    for _ in range(epoch):
        s = 0
        for t in range(seq_len):
            a = np.random.choice(np.arange(env.num_actions))
            r, next_s = env.step(a)
            grad = (r + df * np.dot(features[next_s], W) - np.dot(features[s], W)) * features[s]
            # grad = np.clip(grad, -1000000000000000000, 1000000000000000000)
            W += alpha * grad     
            s = next_s
        print(np.squeeze(((V_bel - np.dot(features, W)) ** 2)).mean())
        msbe.append(np.sum(pi * ((V_bel - np.dot(features, W)) ** 2)))
        #mse_td_linear.append(((V_bel - np.dot(features, W)) ** 2).mean())  # MSE
    return msbe


def TD_linear_off_note(epoch, seq_len, df, alpha, env, dim):
    V_bel = V_bellman(env)
    pi = env.stationary()

    env.state = 0
    mse_td_linear = []  # mean square error in every epoch
    W = np.random.rand(dim)
    features = np.random.normal(1, 1, (env.num_states, dim)) 
    pi_b = 1 / env.num_actions # behavioral policy
    
    for _ in range(epoch):
        s = 0
        for t in range(seq_len):
            a = np.random.choice(np.arange(env.num_actions))
            r, next_s = env.step(a)
            pi_t = a + 1 / np.sum(range(env.num_actions + 1)) # target policy
            p = pi_t / pi_b 
            W += alpha * (p * features[s] * (r + df * np.dot(features[next_s].T, W)) 
                -  features[s] * np.dot(features[s].T, W)) 
            s = next_s

        mse_td_linear.append(((V_bel - np.dot(features, W)) ** 2).mean())  # MSE
    return mse_td_linear

def TD_linear_off_book(epoch, seq_len, df, alpha, env, dim):
    V_bel = V_bellman(env)
    pi = env.stationary()

    env.state = 0
    msbe = []  # mean square error in every epoch
    W = np.random.rand(dim)
    features = np.random.normal(1, 1, (env.num_states, dim)) 
    pi_b = 1 / env.num_actions # behavioral policy 
    
    for _ in range(epoch):
        s = 0
        for t in range(seq_len):
            a = np.random.choice(np.arange(env.num_actions))
            r, next_s = env.step(a)
            pi_t = a + 1 / np.sum(range(env.num_actions + 1)) # target policy
            p = pi_t / pi_b 
            W += alpha * p * ( features[s] * (r + df * np.dot(features[next_s].T, W)) 
                -  features[s] * np.dot(features[s].T, W)) 
            s = next_s

        print(((V_bel - np.dot(features, W)) ** 2).mean())
        msbe.append(np.sum(pi * ((V_bel - np.dot(features, W)) ** 2)))
        #mse_td_linear.append(((V_bel - np.dot(features, W)) ** 2).mean())  # MSE
    return msbe

def TDC(epoch, seq_len, df, alpha, env, dim, beta):
    # TD with gradient correction
    V_bel = V_bellman(env)
    pi = env.stationary()

    env.state = 0
    msbe = []  # mean square error in every epoch
    theta = np.random.rand(dim)
    W = np.random.rand(dim)
    features = np.random.normal(1, 1, (env.num_states, dim)) 
    pi_b = 1 / env.num_actions # behavioral policy
    
    for _ in range(epoch):
        s = 0
        for t in range(seq_len):
            a = np.random.choice(np.arange(env.num_actions))
            r, next_s = env.step(a)
            pi_t = a + 1 / np.sum(range(env.num_actions + 1)) # target policy
            p = pi_t / pi_b
            TD_error = r + df * np.dot(theta.T, features[next_s]) - np.dot(theta.T, features[s])
            theta += alpha * p * (TD_error * features[s] -  df * features[next_s] * np.dot(features[s].T, W))
            W += beta * p * (TD_error - np.dot(features[s].T, W)) * features[s]
            s = next_s
        print(((V_bel - np.dot(features, theta)) ** 2).mean())
        msbe.append(np.sum(pi * ((V_bel - np.dot(features, theta)) ** 2)))
        #mse_td_linear.append(((V_bel - np.dot(features, theta)) ** 2).mean())  # MSE
    return msbe

def GTD_note(epoch, seq_len, df, alpha, env, dim, beta):
    V_bel = V_bellman(env)
    pi = env.stationary()

    env.state = 0
    mse_td_linear = []  # mean square error in every epoch
    theta = np.random.rand(dim)
    W = np.random.rand(dim)
    features = np.random.normal(1, 1, (env.num_states, dim)) 
    pi_b = 1 / env.num_actions # behavioral policy
    
    for _ in range(epoch):
        s = 0
        for t in range(seq_len):
            a = np.random.choice(np.arange(env.num_actions))
            r, next_s = env.step(a)
            pi_t = a + 1 / np.sum(range(env.num_actions + 1)) # target policy
            p = pi_t / pi_b
            W += beta * (features[s] * np.dot(features[s].T, theta) - W
                - p * features[s] * (df * np.dot(features[next_s].T, theta) + r))
            theta -= alpha * (features[s] * np.dot(features[s].T, W) 
                - p * df * features[next_s] * np.dot(features[s].T, W))
            s = next_s
        print(((V_bel - np.dot(features, theta)) ** 2).mean())
        mse_td_linear.append(((V_bel - np.dot(features, theta)) ** 2).mean())  # MSE
    return mse_td_linear

def GTD0(epoch, seq_len, df, alpha, env, dim, beta):
    V_bel = V_bellman(env)
    pi = env.stationary()

    env.state = 0
    msbe = []  # mean square error in every epoch
    theta = np.random.rand(dim)
    W = np.random.rand(dim)
    features = np.random.normal(1, 1, (env.num_states, dim)) 
    pi_b = 1 / env.num_actions # behavioral policy
    
    for _ in range(epoch):
        s = 0
        for t in range(seq_len):
            a = np.random.choice(np.arange(env.num_actions))
            r, next_s = env.step(a)
            pi_t = a + 1 / np.sum(range(env.num_actions + 1)) # target policy
            p = pi_t / pi_b
            TD_error = r + df * np.dot(features[next_s].T, theta) - np.dot(features[s].T, theta)
            W += beta * p * (TD_error * features[s] - W)
            theta += alpha * p * (features[s] - df * features[next_s]) * np.dot(features[s].T, W)
            s = next_s
        print(((V_bel - np.dot(features, theta)) ** 2).mean())
        msbe.append(np.sum(pi * ((V_bel - np.dot(features, theta)) ** 2)))
        #mse_td_linear.append(((V_bel - np.dot(features, theta)) ** 2).mean())  # MSE
    return msbe

def TD_neural(seq_len, df, learning_rate, env, net, averaging=False, off_policy=False, B=1000):
    # only one epoch
    env.state = 0
    running_loss = 0
    # initialize saving path
    if averaging and off_policy:
        path = "avg and off"
    elif averaging:
        path = "avg"
    elif off_policy:
        path = "off_policy"
    else:
        path = "proj"

    # initialize parameters
    s = torch.zeros(env.num_states)
    s[0] = 1
    w_ = net.w0.detach().clone()  # barW for averaging

    # initialize optimizer and logger
    # Note!!!!!!!!! Since the input of the MSEloss is a scalar, 
    # MSELoss(divide by 1) is actually the same as MSELoss(reduction='sum')
    sq_loss = nn.MSELoss(reduction='sum') 
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    writer = SummaryWriter(log_dir='runs/{}'.format(path))
    
    for i in range(seq_len):
        # sample (s, a, r, s')
        a = np.random.choice(np.arange(env.num_actions))
        r, next_s = env.step(a)
        next_s_t = torch.zeros(env.num_states)
        next_s_t = next_s_t.to(device)
        next_s_t[next_s] = 1
        expected = r + df * net(next_s_t).detach()  # detach the variable from computational graph
        expected = expected.to(device)

        if off_policy == True:
            pi_t = a + 1 / np.sum(range(env.num_actions + 1)) # target policy
            pi_b = 1 / env.num_actions # behavioral policy
            p = pi_t / pi_b 
        else:
            p = 1

        # TD update
        optimizer.zero_grad()
        loss = p * sq_loss(net(s), expected)
        loss.backward()
        optimizer.step()

        # projection 
        net.wr.weight.clamp(min=-B,max=B)  

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

def GTD_neural(seq_len, df, learning_rate, env, num_states, m, v=0.01, B=1000):
    # only one epoch
    env.state = 0
    running_loss = 0
    # initialize saving path
    path = "GTD"

    # initialize parameters
    primal_net = Net(num_states, m).to(device)
    dual_net = Net(num_states, m).to(device)
    pi_b = 1 / env.num_actions # behavioral policy
    s = torch.zeros(env.num_states)
    s[0] = 1

    # initialize optimizer and logger
    optimizer_primal = optim.SGD(primal_net.parameters(), lr=learning_rate)
    optimizer_dual = optim.SGD(dual_net.parameters(), lr=learning_rate)
    writer = SummaryWriter(log_dir='runs/{}'.format(path))
    
    for i in range(seq_len):
        # sample (s, a, r, s')
        a = np.random.choice(np.arange(env.num_actions))
        r, next_s = env.step(a)
        next_s_t = torch.zeros(env.num_states).to(device)
        next_s_t[next_s] = 1

        #p ratio
        pi_t = a + 1 / np.sum(range(env.num_actions + 1)) # target policy
        p = pi_t / pi_b 
       
        #p * td_error
        delta = p * (primal_net(s) - df * primal_net(next_s_t) - r)
        #primal loss
        loss_primal = dual_net(s).detach() * delta - 0.5 * dual_net(s).detach()  ** 2 + 0.5 * v * primal_net(s) ** 2
      
        #dual loss
        loss_dual = dual_net(s) * delta.detach() - 0.5 * dual_net(s) ** 2 + 0.5 * v * primal_net(s).detach() ** 2

        # update
        optimizer_primal.zero_grad()
        optimizer_dual.zero_grad()
        loss = loss_primal - loss_dual
        loss.backward()
        optimizer_primal.step()
        optimizer_dual.step()
        learning_rate = 100 / (1000 +(i + 1))
        optimizer_primal = scheduler(optimizer_primal,learning_rate)
        optimizer_dual = scheduler(optimizer_dual,learning_rate)

        # projection 
        primal_net.wr.weight.clamp(min=-B,max=B)
        dual_net.wr.weight.clamp(min=-B,max=B)
      
        s = next_s_t

        # record trainnig loss  
        running_loss += loss_primal.item()
        if i % 1000 == 999:
            # ...log the running loss
            print("loss:", running_loss / 1000, " seq:", i + 1)
            writer.add_scalar('training loss with {}(m = {})'.format(path, primal_net.m), running_loss / 1000, i + 1)
            running_loss = 0.0

    torch.save(primal_net.state_dict(), "weights/m({})={}".format(path, primal_net.m))
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

def scheduler(optimizer,lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def train(num_states, num_actions, seq_len, df, learning_rate, m, env, averaging, off_policy):
    net = Net(num_states, m).to(device)
    TD_neural(seq_len, df, learning_rate, env, net, averaging=averaging, off_policy=off_policy)
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
    pi = env.stationary()
    
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
        mse.append(np.sum(pi * ((V_bel - V_neural) ** 2)))
        #mse_linear.append(TD_linear(epoch, seq_len, df, alpha, env, m))
    print(mse)
    plt.plot(M, mse, label='Neural')
    #plt.plot(M, mse_linear, label='Linear')
    plt.title("MSBE of the value function")
    plt.ylabel("Mean Square Error")
    plt.xlabel("m")
    plt.savefig("neural.png")
    plt.show() 

def evaluate2(num_states, num_actions, M, df, env, path, path2):
    # V(s) of bellman equation
    V_bel = V_bellman(env)
    pi = env.stationary()
    
    msbe = []
    msbe2 = []
    for m in M:
        # load weights
        net = Net(num_states, m)
        net.load_state_dict(torch.load("weights/m({})={}".format(path, m)))
        net2 = Net(num_states, m)
        net2.load_state_dict(torch.load("weights/m({})={}".format(path2, m)))
        with torch.no_grad():
            all_s = torch.zeros(num_states, num_states)
            for i in range(num_states):
                all_s[i][i] = 1
            V_neural = net(all_s).squeeze().numpy()
            V_neural2 = net2(all_s).squeeze().numpy()
        
        assert V_bel.shape == V_neural.shape
        msbe.append(np.sum(pi * ((V_bel - V_neural) ** 2)))
        msbe2.append(np.sum(pi * ((V_bel - V_neural2) ** 2)))
    print(msbe)
    plt.plot(M, msbe, label='neural GTD')
    plt.plot(M, msbe2, label='neural TD')
    plt.title("MSBE of the value function")
    plt.ylabel("Mean Square Bellman Error")
    plt.xlabel("m")
    plt.legend()
    plt.savefig("neural.png")
    plt.show() 

if __name__ == "__main__":
    np.random.seed(4321)
    torch.manual_seed(4321)
    epoch = 1
    alpha = 0.00005
    beta = alpha
    num_states = 500
    num_actions = 5
    seq_len = 300000
    df = 0.9
    learning_rate = 0.05
    #M = [*range(10, 46, 10)]
    M = [*range(50, 501, 50)]
    #M = [400]
    env = Garnet(num_states, num_actions, b_factor=100)
    mse_td = []
    mse_td_linear = []
    mse_td_linear_off = []
    mse_TDC = []
    mse_GTD = []
    for m in  M:
        mse_td_linear.append(TD_linear(epoch, seq_len, df, alpha, env, m))
        mse_td_linear_off.append(TD_linear_off_book(epoch, seq_len, df, alpha, env, m))
        mse_TDC.append(TDC(epoch, seq_len, df, alpha, env, m, beta))
        mse_GTD.append(GTD0(epoch, seq_len, df, 0.0000002, env, m, 0.001))

    plt.plot(M, mse_td_linear, label='TD_linear')
    plt.plot(M, mse_td_linear_off, label='TD_linear(off policy)')
    plt.plot(M, mse_TDC, label='TDC(off policy)')
    plt.plot(M, mse_GTD, label='GTD0(off policy)')
    plt.title("MSBE of the value function")
    plt.ylabel("Mean Square Bellman Error")
    plt.xlabel("m")
    plt.legend()
    plt.savefig("tdtd.png")
    plt.show() 
