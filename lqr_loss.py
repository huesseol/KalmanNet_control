import torch
import torch.nn as nn

class LQRLoss(nn.Module):
    def __init__(self, QT, Qx, Qu, T=1):
        super(LQRLoss, self).__init__()
        self.QT = QT
        self.Qx = Qx
        self.Qu = Qu
        self.T = T

    def forward(self, inputs, target):
        '''
        Calculates the LQR loss over a horizon T

        Parameters
        ----------
        inputs : tuple of 2 tensors (x, u)
            States x (dim_x, T+1) and inputs u (dim_u, T)
        target : tensor
            Target state at time T (usually denoted as "xT")

        Returns
        -------
        cost : float
            LQR cost
        '''
        # TODO make this fast without loop
        x = inputs[0]
        u = inputs[1]

        T = max(u.shape)

        # Scale the total cost by the time horizon
        scale = 1 / T

        x_tilde = x - target.reshape(-1,1)

        terminal_cost = scale * torch.matmul(x_tilde[:,-1], torch.matmul(self.QT, x_tilde[:,-1]))

        stage_costs = 0
        for t in range(T):
            # x'Qx
            state_cost = torch.matmul(x_tilde[:,t], torch.matmul(self.Qx, x_tilde[:,t])) 
            # u'Ru
            control_cost = torch.matmul(u[:,t], torch.matmul(self.Qu, u[:,t]))
            stage_costs += scale * (state_cost + control_cost) 

        cost = terminal_cost + stage_costs
        return cost 


class ControlLoss(nn.Module):
    def __init__(self, alpha, beta, gamma, x_dim, u_dim, R_est=None, R_terminal=None, R_u=None):
        super(ControlLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.R_est = 1.0*torch.eye(x_dim) if R_est is None else R_est 
        self.R_terminal = 1.0*torch.eye(x_dim) if R_terminal is None else R_terminal
        self.R_u = 1.0*torch.eye(u_dim) if R_u is None else R_u

    def forward(self, inputs, targets):
        '''
        Computes the control loss over a trajectory

        Parameters
        ----------
        inputs : tuple of 2 tensors (x, u)
            Actual final state x of shape (dim_x,) and inputs u of shape (dim_u, T)
        target : tensor
            Target state at time T (usually denoted as "xT")
        
        Returns
        -------
        cost : FloatTensor
        '''
        xT, xT_hat, u = inputs
        x_target = targets

        ### Estimation Cost ###
        # Difference between true final state and its estimate
        dx_est = xT.squeeze() - xT_hat.squeeze()
        estimation_cost = dx_est.matmul(self.R_est).matmul(dx_est)

        ### Terminal Cost ###
        # Difference between true final state and target state
        dx_terminal = xT.squeeze() - x_target.squeeze()
        terminal_cost = dx_terminal.matmul(self.R_terminal).matmul(dx_terminal)

        # U = u.flatten()
        # QQu = torch.diag(torch.kron(torch.eye(T), self.Qu))
        # control_cost = U.matmul(QQu * U)

        ### Control Cost ###
        T = max(u.shape)
        control_cost = 0
        for t in range(T):
            control_cost += u[:,t].matmul(self.R_u).matmul(u[:,t]) 

        cost = self.alpha*estimation_cost + self.beta*terminal_cost + self.gamma*control_cost
        return cost


