import torch
import time
from Linear_KF import KalmanFilter
from lqr_loss import LQRLoss
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from support_functions import mean_and_std_linear_and_dB

class Controller:
    def __init__(self, ssModel, KNet=None):
        self.ssModel = ssModel
        self.KNet = KNet
        # self.lqr_loss_fn = LQRLoss(self.ssModel.QT, self.ssModel.Qx, self.ssModel.Qu, self.ssModel.T)
        self.mse_loss_fn = nn.MSELoss(reduction='mean')


    def run_simulation(self, x0, xT, T=100, noise=None, estimator='KNet', print_=True, steady_state=False, true_system=False, limit_u=None):
        '''
        Simulates a trajectory with an estimator (KalmanNet, Kalman Filter, or none) in 
        the control loop. 

        Parameters:
        - x0: A tensor of shape (m,) containing the initial state 
        - xT: A tensor of shape (m,) containing the target state
        - estimator: A string ('KNet' or 'KF') specifying which estimator should be used to 
        compute the control input (default = 'KNet)
        - print_: boolean to enable or disable printing of the error (default = True)
        - noise: None or a tuple of two tensors corresponding to the noise to be used. If None then
        new noise is sampled.
        - true_system: boolean, if true the controller derived from the true system is used. Use this 
        if a reference is needed (default = False).
        - steady_state: boolean, if true the steady state LQR gain is used (default = False).
        - limit_u: None or float, can be used to limit the magnitude of the control input (default = None)

        Returns a tuple of tuples:
        - estimates: (x_hat_knet, x_hat_kf) are the estimates of the state with KalmanNet and Kalman Filter
        - x_y_u: (x, y, u) are the true trajectories resulting from the simulation
        - cost: (LQR_cost, MSE_KNet, MSE_KF) are the different costs
        '''
        self.lqr_loss_fn = LQRLoss(self.ssModel.QT, self.ssModel.Qx, self.ssModel.Qu, T)

        self.KNet.eval()

        # Use a Kalman filter to compare state estimates
        KF = KalmanFilter(self.ssModel)

        with torch.no_grad():
        
            start = time.time()

            # distrib = MultivariateNormal(loc=torch.zeros(self.ssModel.n), covariance_matrix=self.ssModel.R)

            # Initialize simulation, KalmanNet, and Kalman filter
            self.KNet.InitSequence(x0, T)
            self.ssModel.system.InitSimulation(x0)
            KF.InitSequence(x0, self.ssModel.m2x_0)

            # Tensors for state estimates and inputs
            x_hat_knet = torch.empty(self.ssModel.m, T + 1)
            x_hat_knet[:,0] = x0
            x_hat_kf = torch.empty_like(x_hat_knet)
            x_hat_kf[:,0] = x0
            x_true = torch.empty_like(x_hat_knet)
            x_true[:,0] = x0
            y = torch.empty(self.ssModel.n, T + 1)
            y[:,0] = torch.matmul(self.ssModel.H, x0) 
            u = torch.empty(self.ssModel.p, T)

            for t in range(1, T + 1):
                # Calculate LQR input
                if estimator == 'KNet':
                    dx = x_hat_knet[:,t-1] - xT
                elif estimator == 'KF': 
                    dx = x_hat_kf[:,t-1] - xT
                elif estimator == 'obs':
                    dx = y[:,t-1] - xT
                else: # LQR with true state
                    dx = x_true[:,t-1] - xT

                if not steady_state:
                    if not true_system:
                        u[:,t-1] = - torch.matmul(self.ssModel.L[t-1], dx)
                    else:
                        u[:,t-1] = - torch.matmul(self.ssModel.L_true[t-1], dx)
                else:
                    if not true_system:
                        u[:,t-1] = - torch.matmul(self.ssModel.L_infinite, dx)
                    else:
                        u[:,t-1] = - torch.matmul(self.ssModel.L_infinite_true, dx)

                if limit_u is not None:
                    u[:,t-1] = torch.clip(u[:,t-1], -limit_u, limit_u)
                
                # Simulate one step with LQR input
                if noise:
                    y_sim, x_sim = self.ssModel.system.simulate_with_my_noise(u[:,t-1], noise[0][:,t-1], noise[1][:,t-1])
                else:
                    y_sim, x_sim = self.ssModel.system.simulate(u[:,t-1])

                # Obtain state estimates from KalmanNet and Kalman filter
                x_hat_knet[:,t] = self.KNet(y_sim, u[:,t-1])
                x_hat_kf[:,t], _ = KF.Update(y_sim, u[:,t-1])

                # Get true state from simulator
                x_true[:,t] = x_sim
                y[:,t] = y_sim


            # Compute cost for the trajectory
            LQR_cost = self.lqr_loss_fn((x_true, u), xT)
            LQR_cost_dB = 10 * torch.log10(LQR_cost)

            MSE_KNet = self.mse_loss_fn(x_hat_knet[:,1:], x_true[:,1:])
            MSE_KNet_dB = 10 * torch.log10(MSE_KNet)

            MSE_KNet_x1 = self.mse_loss_fn(x_hat_knet[0,1:], x_true[0,1:])
            MSE_KNet_x1_dB = 10 * torch.log10(MSE_KNet_x1)

            MSE_KF = self.mse_loss_fn(x_hat_kf[:,1:], x_true[:,1:])
            MSE_KF_dB = 10 * torch.log10(MSE_KF)

            MSE_KF_x1 = self.mse_loss_fn(x_hat_kf[0,1:], x_true[0,1:])
            MSE_KF_x1_dB = 10 * torch.log10(MSE_KF_x1)


            end = time.time()
            t = end - start
       
        if print_:
            info = f"Estimator: {estimator}, \n" \
                    f"LQR loss: {LQR_cost_dB: .5f} [dB], \n" \
                    f"MSE KNet: {MSE_KNet_dB: .5f} [dB], x1 MSE KNet: {MSE_KNet_x1_dB: .5f} [dB] \n" \
                    f"MSE KF: {MSE_KF_dB: .5f} [dB], x1 MSE KF: {MSE_KF_x1_dB: .5f} [dB]" 

            print(info)
            # Print Run Time
            print("Inference Time:", t)

        estimates = (x_hat_knet, x_hat_kf)
        x_y_u = (x_true, y, u)
        cost = (LQR_cost, MSE_KNet, MSE_KF)

        return estimates, x_y_u, cost


    def run_simulation_batch(self, X0, XT, T=100, noise=None, estimator='KNet', print_=True, steady_state=False):
        '''
        Simulates a batch of trajectories with an estimator (KalmanNet, Kalman Filter, or none) in 
        the control loop.

        Parameters:
        - X0: A tensor of shape (N, m) containing the initial states 
        - XT: A tensor of shape (N, m) containing the target states
        - estimator: A string ('KNet', 'KF', 'LQR', 'obs') specifying which estimator should be used to 
        compute the control input (default = 'KNet). 'LQR' means that the true state is used. 'obs' means 
        that the observation is used as state estimate and works only when the two dimensions are the same.
        - print_: boolean to enable or disable printing of the error (default = True)
        - noise: None or a tuple of two tensors corresponding to the noise to be used. If None then
        new noise is sampled.
        - steady_state: boolean, if true the steady state LQR gain is used (default = False).

        Returns a tuple of tuples with the resulting costs:
        - LQR: (LQR_cost, LQR_std)
        - KNet: (MSE_KNet, MSE_KNet_std)
        - KF: (MSE_KF, MSE_KF_std)
        '''
        self.lqr_loss_fn = LQRLoss(self.ssModel.QT, self.ssModel.Qx, self.ssModel.Qu, T)

        N = X0.shape[0]

        if noise:
            Q_noise, R_noise = noise

        LQR_cost_arr = torch.empty(N)
        state_cost_arr = torch.empty(N)
        control_cost_arr = torch.empty(N)
        MSE_KNet_arr = torch.empty(N)
        MSE_KF_arr = torch.empty(N)
        MSE_KNet_x1_arr = torch.empty(N)
        MSE_KF_x1_arr = torch.empty(N)
        MSE_y_arr = torch.empty(N)

        self.KNet.eval()

        X_hat_kf = torch.empty(N,self.ssModel.m, T+1)

        # Use a Kalman filter to compare state estimates
        KF = KalmanFilter(self.ssModel)

        with torch.no_grad():
        
            start = time.time()

            for k in range(N):
                x0 = X0[k]
                xT = XT[k]
                # Initialize simulation, KalmanNet, and Kalman filter
                self.KNet.InitSequence(x0, T)
                self.ssModel.system.InitSimulation(x0)
                KF.InitSequence(x0, self.ssModel.m2x_0)

                # distrib = MultivariateNormal(loc=torch.zeros(self.ssModel.n), covariance_matrix=self.ssModel.R)

                # Tensors for state estimates and inputs
                x_hat_knet = torch.empty(self.ssModel.m, T + 1)
                x_hat_knet[:,0] = x0
                x_hat_kf = torch.empty_like(x_hat_knet)
                x_hat_kf[:,0] = x0
                x_true = torch.empty_like(x_hat_knet)
                x_true[:,0] = x0
                y = torch.empty(self.ssModel.n, T + 1)
                y[:,0] = torch.matmul(self.ssModel.H, x0)
                u = torch.empty(self.ssModel.p, T)

                for t in range(1, T + 1):
                    # Calculate LQR input
                    if estimator == 'KNet':
                        dx = x_hat_knet[:,t-1] - xT
                    elif estimator == 'KF': 
                        dx = x_hat_kf[:,t-1] - xT
                    elif estimator == 'obs':
                        dx = y[:,t-1] - xT
                    else: # LQR with true state
                        dx = x_true[:,t-1] - xT

                    if not steady_state:
                        u[:,t-1] = - torch.matmul(self.ssModel.L[t-1], dx)
                    else:
                        u[:,t-1] = - torch.matmul(self.ssModel.L_infinite, dx) 
                    
                    # Simulate one step with LQR input
                    if noise:
                        y_sim, x_sim = self.ssModel.system.simulate_with_my_noise(u[:,t-1], Q_noise[k,:,t-1], R_noise[k,:,t-1])
                    else:
                        y_sim, x_sim = self.ssModel.system.simulate(u[:,t-1])

                    # Obtain state estimates from KalmanNet and Kalman filter
                    x_hat_knet[:,t] = self.KNet(y_sim, u[:,t-1])
                    x_hat_kf[:,t], _ = KF.Update(y_sim, u[:,t-1])

                    # Get true state from simulator
                    x_true[:,t] = x_sim
                    y[:,t] = y_sim

                X_hat_kf[k] = x_hat_kf

                # Compute cost for the trajectory
                terminal_costs, state_costs, control_costs = self.ssModel.trajectory_costs(x_true, u, xT)
                state_cost_arr[k] = terminal_costs[0] + state_costs[0]
                control_cost_arr[k] = control_costs[0]
                LQR_cost_arr[k] = terminal_costs[1] + state_costs[1] + control_costs[1]
                # LQR_cost_arr[k] = self.lqr_loss_fn((x_true, u), xT)

                MSE_KNet_arr[k] = self.mse_loss_fn(x_hat_knet[:,1:], x_true[:,1:])
                MSE_KF_arr[k] = self.mse_loss_fn(x_hat_kf[:,1:], x_true[:,1:])    
                MSE_KNet_x1_arr[k] = self.mse_loss_fn(x_hat_knet[0,1:], x_true[0,1:])
                MSE_KF_x1_arr[k] = self.mse_loss_fn(x_hat_kf[0,1:], x_true[0,1:]) 
                if self.ssModel.m == self.ssModel.n:
                    MSE_y_arr[k] = self.mse_loss_fn(y[:,1:], x_true[:,1:])        

            end = time.time()
            t = end - start
       
        # Compute average cost and standard deviation

        # Control performance
        LQR_cost, LQR_cost_dB, LQR_std, LQR_std_dB = mean_and_std_linear_and_dB(LQR_cost_arr)
        state_cost, state_cost_dB, state_std, state_std_dB = mean_and_std_linear_and_dB(state_cost_arr)
        control_cost, control_cost_dB, control_std, control_std_dB = mean_and_std_linear_and_dB(control_cost_arr)
        # Estimator performance
        MSE_KNet, MSE_KNet_dB, MSE_KNet_std, MSE_KNet_std_dB  = mean_and_std_linear_and_dB(MSE_KNet_arr)
        MSE_KF, MSE_KF_dB, MSE_KF_std, MSE_KF_std_dB  = mean_and_std_linear_and_dB(MSE_KF_arr)
        MSE_KNet_x1, MSE_KNet_x1_dB, MSE_KNet_x1_std, MSE_KNet_x1_std_dB  = mean_and_std_linear_and_dB(MSE_KNet_x1_arr)
        MSE_KF_x1, MSE_KF_x1_dB, MSE_KF_x1_std, MSE_KF_x1_std_dB  = mean_and_std_linear_and_dB(MSE_KF_x1_arr)

        if self.ssModel.m == self.ssModel.n:
            MSE_y, MSE_y_dB, MSE_y_std, MSE_y_std_dB = mean_and_std_linear_and_dB(MSE_y_arr)

        if print_:
            info = f"Estimator: {estimator} \n" \
                    f"LQR loss: {LQR_cost_dB: .5f} [dB], STD: {LQR_std_dB: .5f} [dB], " \
                    f"State cost: {state_cost_dB: .5f} [dB], STD: {state_std_dB: .5f} [dB], " \
                    f"Control cost: {control_cost_dB: .5f} [dB], STD: {control_std_dB: .5f} [dB] \n" \
                    f"MSE KNet: {MSE_KNet_dB: .5f} [dB], STD: {MSE_KNet_std_dB: .5f} [dB] \n" \
                    f"MSE KF: {MSE_KF_dB: .5f} [dB], STD: {MSE_KF_std_dB: .5f} [dB] \n" \
                    f"x1 MSE KNet: {MSE_KNet_x1_dB: .5f} [dB], STD: {MSE_KNet_x1_std_dB: .5f} [dB] \n" \
                    f"x1 MSE KF: {MSE_KF_x1_dB: .5f} [dB], STD: {MSE_KF_x1_std_dB: .5f} [dB]"

            print(info)

            if self.ssModel.n == self.ssModel.m:
                print(f"Observation MSE: {MSE_y_dB: .5f} [dB], STD: {MSE_y_std_dB: .5f} [dB]")

            # Print Run Time
            print("Inference Time:", t)

        LQR = (LQR_cost, LQR_std)
        KNet = (MSE_KNet, MSE_KNet_std)
        KF = (MSE_KF, MSE_KF_std)

        return LQR, KNet, KF, X_hat_kf


    def run_simulation_no_knet(self, x0, xT, T=100, noise=None, estimator='KF', print_=True, steady_state=False, true_system=False, correct_model_kf=False, limit_u=None):
        '''
        Simulates a trajectory with an estimator (Kalman Filter, or none) in 
        the control loop. This method can be used when no KalmanNet is needed. It is much faster than the normal "run_simulation".

        Parameters:
        - x0: A tensor of shape (m,) containing the initial state 
        - xT: A tensor of shape (m,) containing the target state
        - estimator: A string ('KF', 'obs', or 'LQR') specifying which estimator should be used to 
        compute the control input (default = 'KF)
        - print_: boolean to enable or disable printing of the error (default = True)
        - noise: None or a tuple of two tensors corresponding to the noise to be used. If None then
        new noise is sampled.
        - true_system: boolean, if true the controller derived from the true system is used. Use this 
        if a reference is needed (default = False).
        - steady_state: boolean, if true the steady state LQR gain is used (default = False).
        - correct_model_kf: boolean, if true the Kalman filter uses the true system as its model. Use this 
        if a reference is needed (default=False).
        - limit_u: None or float, can be used to limit the magnitude of the control input (default = None)

        Returns a tuple of tuples:
        - estimates: x_hat_kf is the estimate of the state from the Kalman Filter
        - x_y_u: (x, y, u) are the true trajectories resulting from the simulation
        - cost: (LQR_cost, MSE_KF) are the different costs
        '''
        self.lqr_loss_fn = LQRLoss(self.ssModel.QT, self.ssModel.Qx, self.ssModel.Qu, T)

        start = time.time()

        if correct_model_kf:
            KF = KalmanFilter(self.ssModel.system)
        else:
            KF = KalmanFilter(self.ssModel)

        # Initialize simulation and Kalman filter
        self.ssModel.system.InitSimulation(x0)
        KF.InitSequence(x0, self.ssModel.m2x_0)

        # Tensors for state estimates and inputs
        x_hat_kf = torch.empty(self.ssModel.m, T + 1)
        x_hat_kf[:,0] = x0
        x_true = torch.empty_like(x_hat_kf)
        x_true[:,0] = x0
        y = torch.empty(self.ssModel.n, T + 1)
        y[:,0] = torch.matmul(self.ssModel.H, x0) 
        u = torch.empty(self.ssModel.p, T)

        for t in range(1, T + 1):
            # Calculate LQR input
            if estimator == 'KF': 
                dx = x_hat_kf[:,t-1] - xT
            elif estimator == 'obs':
                dx = y[:,t-1] - xT
            else: # LQR with true state
                dx = x_true[:,t-1] - xT

            if not steady_state:
                if not true_system:
                    u[:,t-1] = - torch.matmul(self.ssModel.L[t-1], dx)
                else:
                    u[:,t-1] = - torch.matmul(self.ssModel.L_true[t-1], dx)
            else:
                if not true_system:
                    u[:,t-1] = - torch.matmul(self.ssModel.L_infinite, dx)
                else:
                    u[:,t-1] = - torch.matmul(self.ssModel.L_infinite_true, dx)

            if limit_u is not None:
                u[:,t-1] = torch.clip(u[:,t-1], -limit_u, limit_u)
            
            # Simulate one step with LQR input
            if noise:
                y_sim, x_sim = self.ssModel.system.simulate_with_my_noise(u[:,t-1], noise[0][:,t-1], noise[1][:,t-1])
            else:
                y_sim, x_sim = self.ssModel.system.simulate(u[:,t-1])

            # Obtain state estimates from Kalman filter
            x_hat_kf[:,t], _ = KF.Update(y_sim, u[:,t-1])

            # Get true state from simulator
            x_true[:,t] = x_sim
            y[:,t] = y_sim


            # Compute cost for the trajectory
            LQR_cost = self.lqr_loss_fn((x_true, u), xT)
            LQR_cost_dB = 10 * torch.log10(LQR_cost)

            MSE_KF = self.mse_loss_fn(x_hat_kf[:,1:], x_true[:,1:])
            MSE_KF_dB = 10 * torch.log10(MSE_KF)

            MSE_KF_x1 = self.mse_loss_fn(x_hat_kf[0,1:], x_true[0,1:])
            MSE_KF_x1_dB = 10 * torch.log10(MSE_KF_x1)


            end = time.time()
            t = end - start
       
        if print_:
            info = f"Estimator: {estimator}, \n" \
                    f"LQR loss: {LQR_cost_dB: .5f} [dB], \n" \
                    f"MSE KF: {MSE_KF_dB: .5f} [dB], x1 MSE KF: {MSE_KF_x1_dB: .5f} [dB]" 

            print(info)
            # Print Run Time
            print("Inference Time:", t)

        estimates = x_hat_kf
        x_y_u = (x_true, y, u)
        cost = (LQR_cost, MSE_KF)

        return estimates, x_y_u, cost


    def run_simulation_batch_no_knet(self, X0, XT, T=100, noise=None, estimator='KF', print_=True, steady_state=False, correct_model_kf=False):
        '''
        Simulates a batch of trajectories with an estimator (Kalman Filter, or none) in 
        the control loop. This method can be used when no KalmanNet is needed. It is much faster than the normal "run_simulation".

        Parameters:
        - X0: A tensor of shape (N, m) containing the initial states 
        - XT: A tensor of shape (N, m) containing the target states
        - estimator: A string ('KF', 'LQR', 'obs') specifying which estimator should be used to 
        compute the control input (default = 'KF'). 'LQR' means that the true state is used. 'obs' means 
        that the observation is used as state estimate and works only when the two dimensions are the same.
        - print_: boolean to enable or disable printing of the error (default = False)
        - noise: None or a tuple of two tensors corresponding to the noise to be used. If None then
        new noise is sampled.
        - steady_state: boolean, if true the steady state LQR gain is used (default = False).
        - correct_model_kf: boolean, if true the Kalman filter uses the true system as its model. Use this 
        if a reference is needed (default=False).

        Returns a tuple of tuples with the resulting costs:
        - LQR_cost: (LQR_cost_dB, LQR_std_dB)
        - KF_cost: (MSE_KF_dB, MSE_KF_std_dB)
        '''
        self.lqr_loss_fn = LQRLoss(self.ssModel.QT, self.ssModel.Qx, self.ssModel.Qu, T)

        N = X0.shape[0]

        if noise:
            Q_noise, R_noise = noise

        LQR_cost_arr = torch.empty(N)
        state_cost_arr = torch.empty(N)
        control_cost_arr = torch.empty(N)
        MSE_KF_arr = torch.empty(N)
        MSE_KF_x1_arr = torch.empty(N)
        MSE_y_arr = torch.empty(N)

        if correct_model_kf:
            KF = KalmanFilter(self.ssModel.system)
        else:
            KF = KalmanFilter(self.ssModel)

        start = time.time()

        for k in range(N):
            x0 = X0[k]
            xT = XT[k]
            # Initialize simulation and Kalman filter
            self.ssModel.system.InitSimulation(x0)
            KF.InitSequence(x0, self.ssModel.m2x_0)

            # distrib = MultivariateNormal(loc=torch.zeros(self.ssModel.n), covariance_matrix=self.ssModel.R)

            # Tensors for state estimates and inputs
            x_hat_kf = torch.empty(self.ssModel.m, T + 1)
            x_hat_kf[:,0] = x0
            x_true = torch.empty_like(x_hat_kf)
            x_true[:,0] = x0
            y = torch.empty(self.ssModel.n, T + 1)
            y[:,0] = torch.matmul(self.ssModel.H, x0)
            u = torch.empty(self.ssModel.p, T)

            for t in range(1, T + 1):
                # Calculate LQR input
                if estimator == 'KF': 
                    dx = x_hat_kf[:,t-1] - xT
                elif estimator == 'obs':
                    dx = y[:,t-1] - xT
                else: # LQR with true state
                    dx = x_true[:,t-1] - xT

                if not steady_state:
                    u[:,t-1] = - torch.matmul(self.ssModel.L[t-1], dx)
                else:
                    u[:,t-1] = - torch.matmul(self.ssModel.L_infinite, dx) 
                
                # Simulate one step with LQR input
                if noise:
                    y_sim, x_sim = self.ssModel.system.simulate_with_my_noise(u[:,t-1], Q_noise[k,:,t-1], R_noise[k,:,t-1])
                else:
                    y_sim, x_sim = self.ssModel.system.simulate(u[:,t-1])

                # Obtain state estimates from Kalman filter
                x_hat_kf[:,t], _ = KF.Update(y_sim, u[:,t-1])

                # Get true state from simulator
                x_true[:,t] = x_sim
                y[:,t] = y_sim


            # Compute cost for the trajectory
            terminal_costs, state_costs, control_costs = self.ssModel.trajectory_costs(x_true, u, xT)
            state_cost_arr[k] = terminal_costs[0] + state_costs[0]
            control_cost_arr[k] = control_costs[0]
            LQR_cost_arr[k] = terminal_costs[1] + state_costs[1] + control_costs[1]
            # LQR_cost_arr[k] = self.lqr_loss_fn((x_true, u), xT)
            MSE_KF_arr[k] = self.mse_loss_fn(x_hat_kf[:,1:], x_true[:,1:])    
            MSE_KF_x1_arr[k] = self.mse_loss_fn(x_hat_kf[0,1:], x_true[0,1:]) 
            if self.ssModel.m == self.ssModel.n:
                MSE_y_arr[k] = self.mse_loss_fn(y[:,1:], x_true[:,1:])        


        end = time.time()
        t = end - start
               
        # Compute average cost and standard deviation

        # Control performance
        LQR_cost, LQR_cost_dB, LQR_std, LQR_std_dB = mean_and_std_linear_and_dB(LQR_cost_arr)
        state_cost, state_cost_dB, state_std, state_std_dB = mean_and_std_linear_and_dB(state_cost_arr)
        control_cost, control_cost_dB, control_std, control_std_dB = mean_and_std_linear_and_dB(control_cost_arr)
        # Estimator performance
        MSE_KF, MSE_KF_dB, MSE_KF_std, MSE_KF_std_dB  = mean_and_std_linear_and_dB(MSE_KF_arr)
        MSE_KF_x1, MSE_KF_x1_dB, MSE_KF_x1_std, MSE_KF_x1_std_dB  = mean_and_std_linear_and_dB(MSE_KF_x1_arr)

        if self.ssModel.m == self.ssModel.n:
            MSE_y, MSE_y_dB, MSE_y_std, MSE_y_std_dB = mean_and_std_linear_and_dB(MSE_y_arr)

        if print_:
            info = f"Estimator: {estimator} \n" \
                    f"LQR loss: {LQR_cost_dB: .5f} [dB], STD: {LQR_std_dB: .5f} [dB], " \
                    f"State cost: {state_cost_dB: .5f} [dB], STD: {state_std_dB: .5f} [dB], " \
                    f"Control cost: {control_cost_dB: .5f} [dB], STD: {control_std_dB: .5f} [dB] \n" \
                    f"MSE KF: {MSE_KF_dB: .5f} [dB], STD: {MSE_KF_std_dB: .5f} [dB] \n" \
                    f"x1 MSE KF: {MSE_KF_x1_dB: .5f} [dB], STD: {MSE_KF_x1_std_dB: .5f} [dB]"

            print(info)

            if self.ssModel.n == self.ssModel.m:
                print(f"Observation MSE: {MSE_y_dB: .5f} [dB], STD: {MSE_y_std_dB: .5f} [dB]")

            # Print Run Time
            print(f"Inference Time: {t} \n")

        LQR_cost = (LQR_cost_dB, LQR_std_dB)
        KF_cost = (MSE_KF_dB, MSE_KF_std_dB)

        return LQR_cost, KF_cost
    

    def estimate_with_zero_input(self, x0, N, T=100, noise=None, correct_model_kf=False, print_=True):
        '''
        Simulate a batch of trajectories without control input and compute MSE of state estimates from the Kalman filter.
        '''
        if noise:
            Q_noise, R_noise = noise

        MSE_KF_arr = torch.empty(N)
        MSE_KF_x1_arr = torch.empty(N)

        if correct_model_kf:
            KF = KalmanFilter(self.ssModel.system)
        else:
            KF = KalmanFilter(self.ssModel)

        start = time.time()

        for k in range(N):
            
            # Initialize simulation and Kalman filter
            self.ssModel.system.InitSimulation(x0)
            KF.InitSequence(x0, self.ssModel.m2x_0)

            # distrib = MultivariateNormal(loc=torch.zeros(self.ssModel.n), covariance_matrix=self.ssModel.R)

            # Tensors for state estimates and inputs
            x_hat_kf = torch.empty(self.ssModel.m, T + 1)
            x_hat_kf[:,0] = x0
            x_true = torch.empty_like(x_hat_kf)
            x_true[:,0] = x0
            y = torch.empty(self.ssModel.n, T + 1)
            y[:,0] = torch.matmul(self.ssModel.H, x0)
            u = torch.zeros(self.ssModel.p, T) * 0.0

            for t in range(1, T + 1):
                               
                if noise:
                    y_sim, x_sim = self.ssModel.system.simulate_with_my_noise(u[:,t-1], Q_noise[k,:,t-1], R_noise[k,:,t-1])
                else:
                    y_sim, x_sim = self.ssModel.system.simulate(u[:,t-1])

                # Obtain state estimates from Kalman filter
                x_hat_kf[:,t], _ = KF.Update(y_sim, u[:,t-1])

                # Get true state from simulator
                x_true[:,t] = x_sim
                y[:,t] = y_sim


            # Compute cost for the trajectory
            MSE_KF_arr[k] = self.mse_loss_fn(x_hat_kf[:,1:], x_true[:,1:])    
            MSE_KF_x1_arr[k] = self.mse_loss_fn(x_hat_kf[0,1:], x_true[0,1:])    


        end = time.time()
        t = end - start
       
        # Results
        MSE_KF, MSE_KF_dB, MSE_KF_std, MSE_KF_std_dB  = mean_and_std_linear_and_dB(MSE_KF_arr)
        MSE_KF_x1, MSE_KF_x1_dB, MSE_KF_x1_std, MSE_KF_x1_std_dB  = mean_and_std_linear_and_dB(MSE_KF_x1_arr)

        if print_:
            info =  f"MSE KF: {MSE_KF_dB: .5f} [dB], STD: {MSE_KF_std_dB: .5f} [dB] \n" \
                    f"x1 MSE KF: {MSE_KF_x1_dB: .5f} [dB], STD: {MSE_KF_x1_std_dB: .5f} [dB]"
            print(info)
            # Print Run Time
            print("Inference Time:", t)

        KF_results = (MSE_KF_dB, MSE_KF_std_dB, MSE_KF_x1_dB, MSE_KF_x1_std_dB)

        return KF_results
        