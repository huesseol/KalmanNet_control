"""# **Class: Kalman Filter**
Theoretical Linear Kalman
"""
import torch

if torch.cuda.is_available():
    dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
   dev = torch.device("cpu")
   print("Running on the CPU")

class KalmanFilter:

    def __init__(self, SystemModel):
        self.F = SystemModel.F
        self.F_T = torch.transpose(self.F, 0, 1)
        self.m = SystemModel.m

        self.G = SystemModel.G
        self.p = SystemModel.p

        self.Q = SystemModel.Q

        self.H = SystemModel.H
        self.H_T = torch.transpose(self.H, 0, 1)
        self.n = SystemModel.n

        self.R = SystemModel.R

        # self.T = SystemModel.T
        # self.T_test = SystemModel.T_test
   
    # Predict

    def Predict(self, u):
        # Predict the 1-st moment of x
        self.m1x_prior = torch.matmul(self.F, self.m1x_posterior) + torch.matmul(self.G, u)

        # Predict the 2-nd moment of x
        self.m2x_prior = torch.matmul(self.F, self.m2x_posterior)
        self.m2x_prior = torch.matmul(self.m2x_prior, self.F_T) + self.Q

        # Predict the 1-st moment of y
        self.m1y = torch.matmul(self.H, self.m1x_prior)

        # Predict the 2-nd moment of y
        self.m2y = torch.matmul(self.H, self.m2x_prior)
        self.m2y = torch.matmul(self.m2y, self.H_T) + self.R

    # Compute the Kalman Gain
    def KGain(self):
        self.KG = torch.matmul(self.m2x_prior, self.H_T)
        self.KG = torch.matmul(self.KG, torch.inverse(self.m2y))

    # Innovation
    def Innovation(self, y):
        self.dy = y - self.m1y

    # Compute Posterior
    def Correct(self):
        # Compute the 1-st posterior moment
        self.m1x_posterior = self.m1x_prior + torch.matmul(self.KG, self.dy)

        # Compute the 2-nd posterior moment
        self.m2x_posterior = torch.matmul(self.m2y, torch.transpose(self.KG, 0, 1))
        self.m2x_posterior = self.m2x_prior - torch.matmul(self.KG, self.m2x_posterior)

    def Update(self, y, u):
        self.Predict(u)
        self.KGain()
        self.Innovation(y)
        self.Correct()

        return self.m1x_posterior,self.m2x_posterior

    def InitSequence(self, m1x_0, m2x_0):
        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0
        self.m1x_posterior = self.m1x_0
        self.m2x_posterior = self.m2x_0

    #########################
    ### Generate Sequence ###
    #########################
    def GenerateSequence(self, y, u):
        # Assume T > dim y
        T = torch.max(torch.tensor(y.shape))
        # Pre allocate an array for predicted state and variance
        self.x = torch.empty(size=[self.m, T]).to(dev)
        self.sigma = torch.empty(size=[self.m, self.m, T]).to(dev)

        self.m1x_posterior = self.m1x_0
        self.m2x_posterior = self.m2x_0

        for t in range(0, T):
            yt = y[:,t] #torch.unsqueeze(y[:, t], 1)
            # Note that ut is actually the input that was applied at t-1 to reach xt
            ut = u[:, t]
            xt,sigmat = self.Update(yt, ut)
            self.x[:, t] = torch.squeeze(xt)
            self.sigma[:, :, t] = torch.squeeze(sigmat)