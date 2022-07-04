import torch
from torch.distributions.multivariate_normal import MultivariateNormal

class LinearSystem:
    def __init__(self, F, G, H, Q, R):
        self.F = F
        self.G = G
        self.H = H
        self.Q = Q
        self.meanQ = torch.zeros(self.Q.shape[0])
        self.R = R
        self.meanR = torch.zeros(self.R.shape[0])
        self.m = F.shape[0]
        self.p = G.shape[1]
        self.n = H.shape[0]

    
    def InitSimulation(self, x0):
        self.x_sim = x0

    def simulate(self, u, q_noise=True, r_noise=True):
        if q_noise:
            q_distr = MultivariateNormal(loc=self.meanQ, covariance_matrix=self.Q)
            x = self.F.matmul(self.x_sim) + self.G.matmul(u) + q_distr.rsample()
        else:
            x = self.F.matmul(self.x_sim) + self.G.matmul(u)
        
        if r_noise:
            r_distr = MultivariateNormal(loc=self.meanR, covariance_matrix=self.R)
            y = self.H.matmul(x) + r_distr.rsample()
        else:
            y = self.H.matmul(x)
        
        self.x_sim = x
        return y, x

    def simulate_with_my_noise(self, u, q_noise, r_noise):
        x = self.F.matmul(self.x_sim) + self.G.matmul(u) + q_noise
        y = self.H.matmul(x) + r_noise
        self.x_sim = x
        return y, x


    def generate_sequence_with_my_input(self, T, x0, u, q_noise, r_noise):
        x = torch.empty(self.F.shape[0], T)
        y = torch.empty(self.H.shape[0], T)
        self.InitSimulation(x0)
        for t in range(T):
            y[:,t], x[:,t] = self.simulate_with_my_noise(u[:,t], q_noise[:,t], r_noise[:,t])

        return y, x


class System:
    def __init__(self, f, h, Q, R, dim_x, dim_u, dim_y):
        self.f = f
        self.h = h
        self.Q = Q
        self.meanQ = torch.zeros(self.Q.shape[0])
        self.R = R
        self.meanR = torch.zeros(self.R.shape[0])
        self.m = dim_x
        self.p = dim_u
        self.n = dim_y

    
    def InitSimulation(self, x0):
        self.x_sim = x0

    
    def simulate(self, u, q_noise=True, r_noise=True):
        if q_noise:
            q_distr = MultivariateNormal(loc=self.meanQ, covariance_matrix=self.Q)
            x = self.f(self.x_sim, u) + q_distr.rsample()
        else:
            x = self.f(self.x_sim, u)
        
        if r_noise:
            r_distr = MultivariateNormal(loc=self.meanR, covariance_matrix=self.R)
            y = self.h(x) + r_distr.rsample()
        else:
            y = self.h(x)

        self.x_sim = x
        return y, x


    def simulate_with_my_noise(self, u, q_noise, r_noise):
        x = self.f(self.x_sim, u) + q_noise
        y = self.h(x) + r_noise
        self.x_sim = x
        return y, x


    def generate_sequence_with_my_input(self, T, x0, u, q_noise, r_noise, zero_input=False):
        x = torch.empty(self.F.shape[0], T)
        y = torch.empty(self.H.shape[0], T)
        if zero_input:
            u = torch.zeros(u.shape[0], T)
        self.InitSimulation(x0)
        for t in range(T):
            y[:,t], x[:,t] = self.simulate_with_my_noise(u[:,t], q_noise[:,t], r_noise[:,t])

        return y, x


class Cartpole:
    def __init__(self, env, H, R, seed=1):
        self.env = env
        # Enlarge ranges of theta and x 
        # -> include ranges where linear approximation is bad
        self.env.env.theta_threshold_radians *= 2
        self.env.env.x_threshold *= 2
        self.H = H
        self.R = R
        self.meanR = torch.zeros(self.R.shape[0])
        self.done = False
        self.seed = seed
        
    
    def InitSimulation(self, x0):
        # Always start the same -> stabilize training
        self.env.env.seed(self.seed)
        self.env.reset()


    def InitSimulation2(self, x0):
        self.env.reset()


    def simulate(self, u, q_noise, r_noise):
        self.env.env.force_mag = torch.abs(u).item()
        a = 1 if u >= 0 else 0
        obs, r, done, info = self.env.step(a)

        self.done = done
        x = torch.FloatTensor(obs)
        if r_noise:
            r_distr = MultivariateNormal(loc=self.meanR, covariance_matrix=self.R)
            y = torch.matmul(self.H, x) + r_distr.rsample()
        else:
            y = torch.matmul(self.H, x)

        if done:
            self.env.reset()
        
        return y, x

    def simulate_with_my_noise(self, u, q_noise, r_noise):
        self.env.env.force_mag = torch.abs(u).item()
        a = 1 if u >= 0 else 0
        obs, r, done, info = self.env.step(a)

        self.done = done
        x = torch.FloatTensor(obs)

        y = torch.matmul(self.H, x) + r_noise

        if done:
            self.env.reset()
        
        return y, x
