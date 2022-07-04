from os import system
import torch

class ModelFinder:
    def __init__(self, system=None):
        self.system = system


    def generate_trajectory(self, x0, u):
        '''
        Generate a trajectory of the system.

        Parameters
        ----------
        x0: tensor of shape (m,)
        u: tensor of shape (p,T) 
        '''
        if self.system is None:
            print("No system available. Initialize ModelFinder with a system!")
            return None

        T = u.shape[1]

        x = torch.empty(self.system.m, T+1)
        x[:,0] = x0
        y = torch.empty(self.system.n, T)

        self.system.InitSimulation(x0)
        for t in range(1, T+1):
            yt, xt = self.system.simulate(u[:,t-1])
            y[:,t-1] = yt
            x[:,t] = xt

        return y, x


    def least_squares_for_trajectory(self, x, u, y):
        '''
        Computes the least squares estimate of the system dynamics. The estimate is 
        a linear time invariant state space model.

        Parameters
        ----------
        x: tensor of shape (m, T+1)
        u: tensor of shape (p, T)
        y: tensor of shape (n, T) or (n, T+1)

        Returns
        -------
        F, G, H
        '''
        # State and input dimension
        m = x.shape[0]
        p, T = u.shape

        # X = [x1, ... , xT]'
        X = x[:,1:].transpose(0,1)

        # zt = [xt', u1']'
        # Z = [z1, ... , zT]'
        Z = torch.concat((x[:,:-1], u)).transpose(0, 1)

        # FG = [F, G]
        FG = torch.linalg.lstsq(Z,X).solution.transpose(0, 1)

        F = FG[:,:m]
        G = FG[:,m:]

        # H
        if y.shape[1] > T:
            y = y[:,1:]
        H = torch.linalg.lstsq(X, y.transpose(0, 1)).solution.transpose(0, 1)

        return F, G, H


    def least_squares_for_batch_of_trajectories(self, X, U, Y):
        '''
        Computes the least squares estimate of the system dynamics. The estimate is 
        a linear time invariant state space model.

        Parameters
        ----------
        X: tensor of shape (N, m, T+1)
        U: tensor of shape (N, p, T)
        Y: tensor of shape (N, n, T) or (N, n, T+1)

        Returns
        -------
        F, G, H
        '''
        N, m, _ = X.shape
        p = U.shape[1]
        n = Y.shape[1]

        Fs = torch.empty([m, m, N])
        Gs = torch.empty([m, p, N])
        Hs = torch.empty([n, m, N])
        
        for k in range(N):
            Fs[:,:,k], Gs[:,:,k], Hs[:,:,k] = self.least_squares_for_trajectory(X[k], U[k], Y[k])
        
        # Average all estimates
        F = Fs.mean(dim=2)
        G = Gs.mean(dim=2)
        H = Hs.mean(dim=2)
        
        return F, G, H


    # # Not working 
    # def least_squares_online(self, x0, u):
    #     '''
    #     Computes the least squares estimate of the system dynamics online. The estimate is 
    #     a linear time invariant state space model.

    #     https://courses.cs.washington.edu/courses/cse599i/18wi/resources/lecture20/lecture20.pdf

    #     Parameters
    #     ----------
    #     x0: tensor of shape (m,)
    #     u: tensor of shape (p, T)
    #     '''
    #     m = x0.shape[0]
    #     p, T = u.shape

    #     self.system.InitSimulation(x0)

    #     alpha = 1e8
    #     phi_inv = alpha * torch.eye(m+p)
    #     theta = torch.zeros(m+p, m)
    #     delta = torch.zeros(m+p, m)

    #     x = torch.empty(m, T+1)
    #     x[:,0] = x0

    #     for t in range(T):
    #         ut = u[:,t]
    #         xt = x[:,t]
    #         yt, x[:,t+1] = self.system.simulate(ut)

    #         zt = torch.concat((xt,ut)).unsqueeze(1)
    #         ztT = zt.transpose(0, 1)
    #         xt = xt.unsqueeze(1)
    #         xtT = xt.transpose(0, 1)

    #         phi_inv = phi_inv - phi_inv.matmul(zt).matmul(ztT).matmul(phi_inv) \
    #                         / (1 + ztT.matmul(phi_inv).matmul(zt))

    #         delta = delta + zt.matmul(xtT)

    #         theta = phi_inv.matmul(delta)

    #     FG = theta.transpose(0, 1)
    #     F = FG[:,:m]
    #     G = FG[:,m:]

    #     return F, G, x
        