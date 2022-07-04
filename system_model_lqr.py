import torch
from Linear_KF import KalmanFilter
from lqr_support import lqr_finite, kalman_finite, lqr_infinite
from systems import LinearSystem

class SystemModelLQR:

    def __init__(self, F, G, q2, H, r2, T, T_test, system, prior_Q=None, prior_Sigma=None, prior_S=None):
        # True system -> simulator
        self.system = system

        ####################
        ### Motion Model ###
        ####################       
        self.F = F 
        self.FT = torch.transpose(self.F, 0, 1)
        self.m = self.F.size()[0]

        self.G = G 
        self.GT = torch.transpose(self.G, 0, 1)
        self.p = self.G.size()[1]

        self.q2 = q2
        self.Q = q2 * torch.eye(self.m)

        #########################
        ### Observation Model ###
        #########################
        self.H = H 
        self.HT = torch.transpose(self.H, 0 ,1)
        self.n = self.H.size()[0]

        self.r2 = r2
        self.R =  r2 * torch.eye(self.n)

        #Assign T and T_test
        self.T = T
        self.T_test = T_test

        #########################
        ### Cost Matrices LQR ###
        #########################
        self.QT = torch.eye(self.m)
        self.Qx = torch.eye(self.m)
        self.Qu = torch.eye(self.p)

        #########################
        ### Covariance Priors ###
        #########################
        if prior_Q is None:
            self.prior_Q = torch.eye(self.m)
        else:
            self.prior_Q = prior_Q

        if prior_Sigma is None:
            self.prior_Sigma = torch.eye(self.m)
        else:
            self.prior_Sigma = prior_Sigma

        if prior_S is None:
            self.prior_S = torch.eye(self.n)
        else:
            self.prior_S = prior_S    


    #########################
    ### Update Covariance ###
    #########################
    def UpdateCovariance_Gain(self, q2, r2):
        # remove old covariance gain
        self.Q = self.Q / self.q2 
        self.R = self.R / self.r2 

        # add new covariance gain
        self.q2 = q2
        self.Q = q2 * self.Q

        self.r2 = r2
        self.R = r2 * self.R


    def UpdateCovariance_Matrix(self, Q, R):
        self.Q = Q
        self.R = R


    ########################################
    ### Generate Sequence with new noise ###
    ########################################
    # i.e. noise is sampled "online"

    def InitSequence(self, m1x_0, m2x_0):
        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0


    def GenerateSequence(self, xT, T, q_noise=True, r_noise=True, x0=None, model=True, steady_state=False):
        # Pre allocate an array for current state
        self.x = torch.empty(size=[self.m, T])
        # Pre allocate an array for current observation
        self.y = torch.empty(size=[self.n, T])
        # Pre allocate an array for current input
        self.u = torch.empty(size=[self.p, T])

        # Specify computation method
        if model:
            # Use model for computation
            L = self.L
            if steady_state:
                L = self.L_infinite
        else:
            # Use true system for computation (i.e. model = true system)
            L = self.L_true
            if steady_state:
                L = self.L_infinite_true

        # Set x0 to be x previous
        if x0 is None:
            self.x_prev = self.m1x_0
        else:
            self.x_prev = x0
        xT = xT.reshape(self.x_prev.size()) # make sure they have the same shape

        # Generate Sequence Iteratively
        self.system.InitSimulation(self.x_prev)
        for t in range(0, T):
            #################
            ### LQR Input ###
            #################
            # Deviation from target state
            dx = self.x_prev - xT 
            # Linear state feedback: u = -L*dx
            if not steady_state:
                ut = - torch.matmul(L[t], dx) 
            else:
                ut = - torch.matmul(L, dx)

            #########################
            ### Simulate one step ###
            #########################
            yt, xt = self.system.simulate(ut, q_noise, r_noise)            
            
            ########################
            ### Squeeze to Array ###
            ########################

            # Save Current State to Trajectory Array
            self.x[:, t] = torch.squeeze(xt)

            # Save Current Observation to Trajectory Array
            self.y[:, t] = torch.squeeze(yt)

            # Save Current Input to Trajectory Array
            self.u[:, t] = torch.squeeze(ut)

            ################################
            ### Save Current to Previous ###
            ################################
            self.x_prev = xt


    def GenerateSequence_without_lqr(self, x0, T, u,  Q_noise, R_noise):
        # Pre allocate tensors
        self.x = torch.empty(size=[self.m, T])
        self.y = torch.empty(size=[self.n, T])
        self.u = u # (p, T)

        self.system.InitSimulation(x0)
        for t in range(T):
            self.y[:,t], self.x[:,t] = self.system.simulate_with_my_noise(self.u[:,t], Q_noise[:,t], R_noise[:,t])


    def GenerateSequence_with_LQG(self, x0, T, q_noise, r_noise, steady_state=False, xT=None):

        KF = KalmanFilter(self)

        # Allocate tensors
        x_hat = torch.empty(self.m, T+1)
        x_hat[:,0] = x0
        x_true = torch.empty_like(x_hat)
        x_true[:,0] = x0
        self.x = torch.empty(self.m, T)
        self.u = torch.empty(self.p, T)
        self.y = torch.empty(self.n, T)

        self.system.InitSimulation(x0)
        KF.InitSequence(x0, self.m2x_0)

        # Simulate trajectory with Kalman filter as estimator
        for t in range(1, T+1):
            # LQR input
            dx = x_hat[:,t-1] - xT
            if not steady_state:
                self.u[:,t-1] = - torch.matmul(self.L[t-1], dx)
            else:
                self.u[:,t-1] = - torch.matmul(self.L_infinite, dx)

            # Simulate one step
            self.y[:,t-1], self.x[:,t-1] = self.system.simulate_with_my_noise(self.u[:,t-1], q_noise[:, t-1], r_noise[:, t-1])

            # Estimate state with Kalman filter
            x_hat[:,t], _ = KF.Update(self.y[:,t-1], self.u[:,t-1])
      

    #############################################
    ### Generate Sequence with existing noise ###
    #############################################
    # i.e. noise was sampled before

    def GenerateSequence_with_my_noise(self, xT, T, Q_noise, R_noise, x0=None, model=True, steady_state=False):
        '''
        Generate a complete LQR sequence by using the given noise sequences Q_noise and R_noise.
        The resulting state, input, and output trajectories are stored in x, u, and y respectively (i.e., access with self.x)

        Parameters
        ----------
        xT: FloatTensor of shape (m,)
            Target state after T steps
        T: int
            Time horizon
        Q_noise: FloatTensor of shape (m, T)
            Noise sequence
        R_noise: FloatTensor of shape (n, T)
        x0: FloatTensor of shape (m,)
            Optional starting state. By default the starting state of the model will be used.
        steady_state: boolean
            If true the steady state LQR gain is used (default = False).
        '''
        # Pre allocate an array for current state
        self.x = torch.empty(size=[self.m, T])
        # Pre allocate an array for current observation
        self.y = torch.empty(size=[self.n, T])
        # Pre allocate an array for current input
        self.u = torch.empty(size=[self.p, T])

        # Specify computation method
        if model:
            # Use model for computation
            L = self.L
            if steady_state:
                L = self.L_infinite
        else:
            # Use true system for computation (i.e. model = true system)
            L = self.L_true
            if steady_state:
                L = self.L_infinite_true

        # Set x0 to be x previous
        if x0 is None:
            self.x_prev = self.m1x_0
        else:
            self.x_prev = x0
        xT = xT.reshape(self.x_prev.size()) # make sure they have the same shape

        # Generate Sequence Iteratively
        self.system.InitSimulation(self.x_prev)
        for t in range(0, T):
            #################
            ### LQR Input ###
            #################
            # Deviation from target state
            dx = self.x_prev - xT 
            # Linear state feedback: u = -L*dx
            if not steady_state:
                ut = - torch.matmul(L[t], dx) 
            else:
                ut = - torch.matmul(L, dx)

            #########################
            ### Simulate one step ###
            #########################
            yt, xt = self.system.simulate_with_my_noise(ut, Q_noise[:,t], R_noise[:,t])
            
            ########################
            ### Squeeze to Array ###
            ########################

            self.x[:, t] = torch.squeeze(xt)
            self.y[:, t] = torch.squeeze(yt)
            self.u[:, t] = torch.squeeze(ut)

            ################################
            ### Save Current to Previous ###
            ################################
            self.x_prev = xt


    ######################
    ### Generate Batch ###
    ######################
    def GenerateBatch(self, size, T, xT, randomInit=False, seqInit=False, T_test=0):

        # Allocate Empty Array for Input
        self.Input_y = torch.empty(size, self.n, T)
        self.Input_u = torch.empty(size, self.p, T)

        # Allocate Empty Array for Target
        self.Target = torch.empty(size, self.m, T)

        ### Generate Examples
        initConditions = self.m1x_0

        for i in range(0, size):
            # Generate Sequence

            # Randomize initial conditions to get a rich dataset
            if(randomInit):
                variance = 100
                initConditions = torch.rand_like(self.m1x_0) * variance
            if(seqInit):
                initConditions = self.x_prev
                if((i*T % T_test)==0):
                    initConditions = torch.zeros_like(self.m1x_0)

            self.InitSequence(initConditions, self.m2x_0)
            self.GenerateSequence(xT, T)

            # Training sequence input y 
            self.Input_y[i, :, :] = self.y

            # Training sequence input u 
            self.Input_u[i, :, :] = self.u

            # Training sequence output
            self.Target[i, :, :] = self.x


    ###########
    ### LQR ###
    ###########
    def InitCostMatrices(self, QN, Qx, Qu):
        self.QT = QN
        self.Qx = Qx
        self.Qu = Qu
        self.ComputeLQRgains()


    def ComputeLQRgains(self):
        self.L, self.S = lqr_finite(self.T, self.F, self.G, self.QT, self.Qx, self.Qu)
        self.L_infinite, self.S_infinite = lqr_infinite(self.F, self.G, self.Qx, self.Qu)
        if isinstance(self.system, LinearSystem):
            self.L_true, self.S_true = lqr_finite(self.T, self.system.F, self.system.G, self.QT, self.Qx, self.Qu)
            self.L_infinite_true, self.S_infinite_true = lqr_infinite(self.system.F, self.system.G, self.Qx, self.Qu)
        else:
            self.L_true, self.S_true = self.L, self.S
            self.L_infinite_true, self.S_infinite_true = self.L_infinite, self.S_infinite
 
    
    def ComputeKalmanGains(self):
        self.K, self.P = kalman_finite(self.T, self.F, self.H, self.Q, self.R, self.m2x_0)


    def ComputeOptimalCost(self, x0, xT, model=True, steady_state=False):
        '''
        Computes the optimal LQR cost for the horizon defined by the model. This uses the
        noise free model dynamics.

        Parameters
        ----------
        x0 : tensor
            Initial state
        xT : tensor
            Target state
        model : bool
            Indicates whether to use the model (True) or the true system (False) dynamics to compute the  
            control gain. Note that the use of the true system dynamics is just for comparison.

        Returns
        -------
        cost : torch.float
        '''
        # First, compute optimal trajectory
        self.GenerateSequence(xT, self.T, q_noise=False, r_noise=False, x0=x0, model=model, steady_state=steady_state)

        # Add initial state to trajectory
        x = torch.concat((x0.unsqueeze(1), self.x), 1)
        cost = self.LQR_cost(x, self.u, xT)
        return cost


    def ExpectedCost_LQR(self, m1x_0=None, m2x_0=None):
        '''
        Computes the expected minimum cost as on slide 14 in 
        https://stanford.edu/class/ee363/lectures/lqg.pdf
        '''
        if m1x_0 is None:
            m1x_0 = self.m1x_0
        if m2x_0 is None:
            m2x_0 = self.m2x_0

        # Expected cost according to Astroem, chapter 8, eq. 4.16
        l1 = m1x_0.matmul(self.S[0]).matmul(m1x_0)
        l2 = torch.trace(torch.matmul(self.S[0], m2x_0))
        l3 = 0
        for t in range(self.T):
            l3 += torch.trace(torch.matmul(self.S[t+1], self.Q))

        expected_cost = l1 + l2 + l3
        return expected_cost 


    def ExpectedCost_LQG(self, m1x_0=None, m2x_0=None):
        '''
        Computes the expected minimum cost as on slide 14 in 
        https://stanford.edu/class/ee363/lectures/lqg.pdf
        '''
        if m1x_0 is None:
            m1x_0 = self.m1x_0
        if m2x_0 is None:
            m2x_0 = self.m2x_0

        J_lqr = m1x_0.matmul(self.S[0]).matmul(m1x_0)
        J_lqr += torch.trace(torch.matmul(self.S[0], m2x_0))
        
        J_est = torch.trace(torch.matmul(self.QT - self.S[0], self.P[0]))
        for t in range(1,self.T+1):
            J_lqr += torch.trace(torch.matmul(self.S[t], self.Q))
            J_est += torch.trace(torch.matmul(self.Qx - self.S[t], self.P[t]))
            p = self.S[t].matmul(self.F).matmul(self.P[t-1]).matmul(self.FT)
            J_est += torch.trace(p)

        expected_cost = J_lqr + J_est
        return expected_cost 


    def ExpectedCost_LQG2(self, m1x_0=None, m2x_0=None):
        '''
        Computes the expected minimum cost as in eq. 6.25 of chapter 8 in Astroem
        '''
        if m1x_0 is None:
            m1x_0 = self.m1x_0
        if m2x_0 is None:
            m2x_0 = self.m2x_0

        # Expected cost according to Astroem, chapter 8, eq. 6.25
        l1 = m1x_0.matmul(self.S[0]).matmul(m1x_0)
        l2 = torch.trace(torch.matmul(self.S[0], m2x_0))
        l3 = 0
        l4 = 0
        for t in range(self.T):
            l3 += torch.trace(torch.matmul(self.S[t+1], self.Q))
            p1 = torch.matmul(self.P[t], torch.transpose(self.L[t],0,1))
            p2 = self.GT.matmul(self.S[t+1]).matmul(self.G) + self.Qu
            l4 += torch.trace(p1.matmul(p2).matmul(self.L[t]))

        expected_cost = l1 + l2 + l3 + l4
        return expected_cost 


    def EstimateLQRCost(self, x0, xT, M, my_noise=None, T=None, model=True, steady_state=False):
        '''
        Approximates the LQR cost for the horizon defined by the model by averaging 
        over M trajectories. This uses noisy state dynamics.

        Parameters
        ----------
        x0 : tensor
            Initial state
        xT : tensor
            Target state
        M : int
            Number of trajectories to use for averaging
        my_noise : None or tuple of two tensors
            The tuple is (Q_noise, R_noise) where Q_noise has shape (M,m,T) and R_noise has shape (M,n,T)
        model : bool
            Indicates whether to use the model (True) or the true system (False) dynamics to compute the  
            control gain. Note that the use of the true system dynamics is just for comparison.

        Returns
        -------
        cost : torch.float 
        '''
        ######################
        ### Initial checks ###
        ######################

        # Initial and target states
        if x0.numel() == self.m:
            # Always the same inital state
            X0 = torch.ones(M,self.m) * x0
        elif x0.numel() == self.m * M:
            X0 = x0
        else:
            print("Check initial states")
            return None

        if xT.numel() == self.m:
            # Always the same inital state
            XT = torch.ones(M,self.m) * xT
        elif xT.numel() == self.m * M:
            XT = xT
        else:
            print("Check target states")
            return None

        # In case noise is given
        if my_noise is not None:
            Q_noise, R_noise = my_noise
            if Q_noise.shape[0] < M:
                print(f"Number of noise trajectories ({Q_noise.shape[0]}) is less than M ({M})")
                return None

        # Define trajectory length
        if T is None:
            T = self.T

        # Generate M trajectories and average to approximate expectation
        cost = 0
        for k in range(M):
            # Generate a trajectory
            if my_noise is not None:
                self.GenerateSequence_with_my_noise(XT[k], T, Q_noise=Q_noise[k], R_noise=R_noise[k], x0=X0[k], model=model, steady_state=steady_state)
            else:
                self.GenerateSequence(XT[k], T, x0=X0[k], model=model, steady_state=steady_state)

            # Add initial state to trajectory
            x = torch.concat((X0[k].unsqueeze(1), self.x), 1)
            trajectory_cost = self.LQR_cost(x, self.u, XT[k])
            cost += trajectory_cost / M
        
        return cost


    def EstimateLQGCost(self, x0, xT, M, my_noise=None, T=None, model=True, steady_state=False):
        '''
        Approximates the LQG cost for the horizon defined by the model by averaging 
        over M trajectories. Here the state is not known but has to be estimated with
        a Kalman filter. Hence, this cost will be higher than the LQR cost.

        Parameters
        ----------
        x0 : tensor of shape (m,) or (M,m)
            Initial state(s)
        xT : tensor of shape (m,) or (M,m)
            Target state(s)
        M : int
            Number of trajectories to use for averaging
        my_noise : None or tuple of two tensors
            The tuple is (Q_noise, R_noise) where Q_noise has shape (M,m,T) and R_noise has shape (M,n,T)
        model : bool
            Indicates whether to use the model (True) or the true system (False) dynamics to compute the  
            control gain. Note that the use of the true system dynamics is just for comparison.

        Returns
        -------
        cost : torch.float 
        '''
        ######################
        ### Initial checks ###
        ######################

        # Initial and target states
        if x0.numel() == self.m:
            # Always the same inital state
            X0 = torch.ones(M,self.m) * x0
        elif x0.numel() == self.m * M:
            X0 = x0
        else:
            print("Check initial states")
            return None

        if xT.numel() == self.m:
            # Always the same inital state
            XT = torch.ones(M,self.m) * xT
        elif xT.numel() == self.m * M:
            XT = xT
        else:
            print("Check target states")
            return None

        # In case noise is given
        if my_noise is not None:
            Q_noise, R_noise = my_noise
            if Q_noise.shape[0] < M:
                print(f"Number of noise trajectories ({Q_noise.shape[0]}) is less than M ({M})")
                return None

        # Specify computation method
        if model:
            # Use model for computation
            L = self.L
            KF = KalmanFilter(self) 
            if steady_state:
                L = self.L_infinite
        else:
            # Use true system for computation (i.e. model = true system)
            L = self.L_true
            KF = KalmanFilter(self.system)
            if steady_state:
                L = self.L_infinite_true

        # Define trajectory length
        if T is None:
            T = self.T

        ###################
        ### Computation ###
        ###################
        
        cost = 0
        cost2 = 0
        for k in range(M):
            # Initialize simulation and Kalman filter
            self.system.InitSimulation(X0[k])
            KF.InitSequence(X0[k], self.m2x_0)

            if my_noise is not None:
                q_noise = Q_noise[k]
                r_noise = R_noise[k]
            
            # Allocate tensors
            x_hat = torch.empty(self.m, T+1)
            x_hat[:,0] = X0[k]
            x_true = torch.empty_like(x_hat)
            x_true[:,0] = X0[k]
            u = torch.empty(self.p, T)

            # Simulate trajectory with Kalman filter as estimator
            for t in range(1, T+1):
                # LQR input
                dx = x_hat[:,t-1] - XT[k]
                if not steady_state:
                    u[:,t-1] = - torch.matmul(L[t-1], dx)
                else:
                    u[:,t-1] = - torch.matmul(L, dx)

                # Simulate one step
                if my_noise is not None:
                    y, x_true[:,t] = self.system.simulate_with_my_noise(u[:,t-1], q_noise[:, t-1], r_noise[:, t-1])
                else:
                    y, x_true[:,t] = self.system.simulate(u[:,t-1])

                # Estimate state with Kalman filter
                x_hat[:,t], _ = KF.Update(y, u[:,t-1])

            # Calculate cost for trajectory
            trajectory_cost = self.LQR_cost(x_true, u, XT[k])
            cost += trajectory_cost / M 

            # terminal_costs, state_costs, control_costs = self.trajectory_costs(x_true, u, XT[k])
            # cost2 += (terminal_costs[1] + state_costs[1] + control_costs[1]) / M 
        
        return cost


    def LQR_cost(self, x, u, xT):
        '''
        Computes the LQR cost for the given trajectories.

        Parameters
        ----------
        x : tensor
            State trajectory
        u : tensor
            Input trajectory
        xT : tensor
            Target state

        Returns
        -------
        cost : torch.float
        '''
        T = max(u.shape)

        # Scale the total cost by the time horizon
        scale = 1 / T

        x_tilde = x - xT.reshape(self.m,1)

        terminal_cost = scale * torch.matmul(x_tilde[:,-1], torch.matmul(self.QT, x_tilde[:,-1]))

        stage_costs = 0
        for t in range(self.T):
            # x'Qx
            state_cost = torch.matmul(x_tilde[:,t], torch.matmul(self.Qx, x_tilde[:,t])) 
            # u'Ru
            control_cost = torch.matmul(u[:,t], torch.matmul(self.Qu, u[:,t]))
            stage_costs += scale * (state_cost + control_cost)

        cost = terminal_cost + stage_costs
        return cost


    def developLQRCost(self, x0, xT, M, my_noise=None, model=True, steady_state=False, print_=False):
        '''
        Approximates the LQR cost for the horizon defined by the model by averaging 
        over M trajectories. This uses noisy state dynamics.

        Parameters
        ----------
        x0 : tensor
            Initial state
        xT : tensor
            Target state
        M : int
            Number of trajectories to use for averaging
        my_noise : None or tuple of two tensors
            The tuple is (Q_noise, R_noise) where Q_noise has shape (M,m,T) and R_noise has shape (M,n,T)
        model : bool
            Indicates whether to use the model (True) or the true system (False) dynamics to compute the  
            control gain. Note that the use of the true system dynamics is just for comparison.

        Returns
        -------
        cost : torch.float 
        '''
        ######################
        ### Initial checks ###
        ######################

        # Initial and target states
        if x0.numel() == self.m:
            # Always the same inital state
            X0 = torch.ones(M,self.m) * x0
        elif x0.numel() == self.m * M:
            X0 = x0
        else:
            print("Check initial states")
            return None

        if xT.numel() == self.m:
            # Always the same inital state
            XT = torch.ones(M,self.m) * xT
        elif xT.numel() == self.m * M:
            XT = xT
        else:
            print("Check target states")
            return None

        # In case noise is given
        if my_noise is not None:
            Q_noise, R_noise = my_noise
            if Q_noise.shape[0] < M:
                print(f"Number of noise trajectories ({Q_noise.shape[0]}) is less than M ({M})")
                return None

        # Generate M trajectories and average to approximate expectation
        cost_xT = 0
        cost_x = 0
        cost_u = 0
        cost_xT_lqr = 0
        cost_x_lqr = 0
        cost_u_lqr = 0
        for k in range(M):
            # Generate a trajectory
            if my_noise is not None:
                self.GenerateSequence_with_my_noise(XT[k], self.T, Q_noise=Q_noise[k], R_noise=R_noise[k], x0=X0[k], model=model, steady_state=steady_state)
            else:
                self.GenerateSequence(XT[k], self.T, x0=X0[k], model=model, steady_state=steady_state)

            # Add initial state to trajectory
            x = torch.concat((X0[k].unsqueeze(1), self.x), 1)

            terminal_costs, state_costs, control_costs = self.trajectory_costs(x, self.u, XT[k])
            cost_xT += terminal_costs[0] / M
            cost_xT_lqr += terminal_costs[1] / M
            cost_x += state_costs[0] / M
            cost_x_lqr += state_costs[1] / M
            cost_u += control_costs[0] / M
            cost_u_lqr += control_costs[1] / M

        if print_:
            print(f"Cost xT: {cost_xT: .5f}")
            print(f"Cost x: {cost_x: .5f}")
            print(f"Cost u: {cost_u: .5f}")
            print(f"Cost xT lqr: {cost_xT_lqr: .5f}")
            print(f"Cost x lqr: {cost_x_lqr: .5f}")
            print(f"Cost u lqr: {cost_u_lqr: .5f}")

        return (cost_xT, cost_xT_lqr), (cost_x, cost_x_lqr), (cost_u, cost_u_lqr)


    def developLQGCost(self, x0, xT, M, my_noise=None, model=True, steady_state=False, print_=False):
        '''
        Approximates the LQG cost for the horizon defined by the model by averaging 
        over M trajectories. Here the state is not known but has to be estimated with
        a Kalman filter. Hence, this cost will be higher than the LQR cost.

        Parameters
        ----------
        x0 : tensor of shape (m,) or (M,m)
            Initial state(s)
        xT : tensor of shape (m,) or (M,m)
            Target state(s)
        M : int
            Number of trajectories to use for averaging
        my_noise : None or tuple of two tensors
            The tuple is (Q_noise, R_noise) where Q_noise has shape (M,m,T) and R_noise has shape (M,n,T)
        model : bool
            Indicates whether to use the model (True) or the true system (False) dynamics to compute the  
            control gain. Note that the use of the true system dynamics is just for comparison.

        Returns
        -------
        cost : torch.float 
        '''
        ######################
        ### Initial checks ###
        ######################

        # Initial and target states
        if x0.numel() == self.m:
            # Always the same inital state
            X0 = torch.ones(M,self.m) * x0
        elif x0.numel() == self.m * M:
            X0 = x0
        else:
            print("Check initial states")
            return None

        if xT.numel() == self.m:
            # Always the same inital state
            XT = torch.ones(M,self.m) * xT
        elif xT.numel() == self.m * M:
            XT = xT
        else:
            print("Check target states")
            return None

        # In case noise is given
        if my_noise is not None:
            Q_noise, R_noise = my_noise
            if Q_noise.shape[0] < M:
                print(f"Number of noise trajectories ({Q_noise.shape[0]}) is less than M ({M})")
                return None

        # Specify computation method
        if model:
            # Use model for computation
            L = self.L
            KF = KalmanFilter(self) 
            if steady_state:
                L = self.L_infinite
        else:
            # Use true system for computation (i.e. model = true system)
            L = self.L_true
            KF = KalmanFilter(self.system)
            if steady_state:
                L = self.L_infinite_true

        ###################
        ### Computation ###
        ###################
        
        cost_xT = 0
        cost_x = 0
        cost_u = 0
        cost_xT_lqr = 0
        cost_x_lqr = 0
        cost_u_lqr = 0
        cost_xT_hat = 0
        for k in range(M):
            # Initialize simulation and Kalman filter
            self.system.InitSimulation(X0[k])
            KF.InitSequence(X0[k], self.m2x_0)

            if my_noise is not None:
                q_noise = Q_noise[k]
                r_noise = R_noise[k]
            
            # Allocate tensors
            x_hat = torch.empty(self.m, self.T+1)
            x_hat[:,0] = X0[k]
            x_true = torch.empty_like(x_hat)
            x_true[:,0] = X0[k]
            u = torch.empty(self.p, self.T)

            # Simulate trajectory with Kalman filter as estimator
            for t in range(1, self.T+1):
                # LQR input
                dx = x_hat[:,t-1] - XT[k]
                if not steady_state:
                    u[:,t-1] = - torch.matmul(L[t-1], dx)
                else:
                    u[:,t-1] = - torch.matmul(L, dx)

                # Simulate one step
                if my_noise is not None:
                    y, x_true[:,t] = self.system.simulate_with_my_noise(u[:,t-1], q_noise[:, t-1], r_noise[:, t-1])
                else:
                    y, x_true[:,t] = self.system.simulate(u[:,t-1])

                # Estimate state with Kalman filter
                x_hat[:,t], _ = KF.Update(y, u[:,t-1])

            # Calculate cost for trajectory
            terminal_costs, state_costs, control_costs = self.trajectory_costs(x_true, u, XT[k])
            cost_xT += terminal_costs[0] / M
            cost_xT_lqr += terminal_costs[1] / M
            cost_x += state_costs[0] / M
            cost_x_lqr += state_costs[1] / M
            cost_u += control_costs[0] / M
            cost_u_lqr += control_costs[1] / M
            
            # Squared estimation error
            cost_xT_hat += torch.sum( (x_hat[:,-1] - x_true[:,-1])**2 ) / M

        if print_:
            print(f"Cost xT: {cost_xT: .5f}")
            print(f"Cost x: {cost_x: .5f}")
            print(f"Cost u: {cost_u: .5f}")
            print(f"Cost xT lqr: {cost_xT_lqr: .5f}")
            print(f"Cost x lqr: {cost_x_lqr: .5f}")
            print(f"Cost u lqr: {cost_u_lqr: .5f}")
            print(f"Cost xT mse: {cost_xT_hat: .5f}")

        return (cost_xT, cost_xT_lqr), (cost_x, cost_x_lqr), (cost_u, cost_u_lqr)


    def trajectory_costs(self, x, u, xT):
        '''
        Computes the cost for the given trajectories for two cases:
        1. x'x and u'u
        2. x'Qx and u'Ru

        Parameters
        ----------
        x : tensor
            State trajectory
        u : tensor
            Input trajectory
        xT : tensor
            Target state

        Returns
        -------
        terminal_costs: (terminal_cost, terminal_cost_lqr)
        state_costs: (stage_costs_x, stage_costs_x_lqr)
        control_costs: (stage_costs_u, stage_costs_u_lqr)
        '''
        T = max(u.shape)
        scale = 1/T
        
        x_tilde = x - xT.reshape(self.m,1)

        terminal_cost =  scale * torch.matmul(x_tilde[:,-1], x_tilde[:,-1])
        terminal_cost_lqr = scale * torch.matmul(x_tilde[:,-1], torch.matmul(self.QT, x_tilde[:,-1]))

        stage_costs_u = 0
        stage_costs_x = 0
        stage_costs_u_lqr = 0
        stage_costs_x_lqr = 0
        for t in range(T):
            # x'Qx
            stage_costs_x += scale * torch.matmul(x_tilde[:,t], x_tilde[:,t])
            stage_costs_x_lqr += scale * torch.matmul(x_tilde[:,t], torch.matmul(self.Qx, x_tilde[:,t])) 
            # u'Ru
            stage_costs_u += scale * torch.matmul(u[:,t], u[:,t])
            stage_costs_u_lqr += scale * torch.matmul(u[:,t], torch.matmul(self.Qu, u[:,t]))

        terminal_costs = (terminal_cost, terminal_cost_lqr)
        state_costs = (stage_costs_x, stage_costs_x_lqr)
        control_costs = (stage_costs_u, stage_costs_u_lqr)
        return  terminal_costs, state_costs, control_costs


    def dev_RL_cost(self, x0, xT, M, my_noise=None, model=True, steady_state=False):
        '''
        Parameters
        ----------
        x0 : tensor of shape (m,) or (M,m)
            Initial state(s)
        xT : tensor of shape (m,) or (M,m)
            Target state(s)
        M : int
            Number of trajectories to use for averaging
        my_noise : None or tuple of two tensors
            The tuple is (Q_noise, R_noise) where Q_noise has shape (M,m,T) and R_noise has shape (M,n,T)
        model : bool
            Indicates whether to use the model (True) or the true system (False) dynamics to compute the  
            control gain. Note that the use of the true system dynamics is just for comparison.

        Returns
        -------
        - estimation cost
        - terminal state cost
        - control cost 
        '''
        ######################
        ### Initial checks ###
        ######################

        # Initial and target states
        if x0.numel() == self.m:
            # Always the same inital state
            X0 = torch.ones(M,self.m) * x0
        elif x0.numel() == self.m * M:
            X0 = x0
        else:
            print("Check initial states")
            return None

        if xT.numel() == self.m:
            # Always the same inital state
            XT = torch.ones(M,self.m) * xT
        elif xT.numel() == self.m * M:
            XT = xT
        else:
            print("Check target states")
            return None

        # In case noise is given
        if my_noise is not None:
            Q_noise, R_noise = my_noise
            if Q_noise.shape[0] < M:
                print(f"Number of noise trajectories ({Q_noise.shape[0]}) is less than M ({M})")
                return None

        # Specify computation method
        if model:
            # Use model for computation
            L = self.L
            if steady_state:
                L = self.L_infinite
        else:
            # Use true system for computation (i.e. model = true system)
            L = self.L_true
            if steady_state:
                L = self.L_infinite_true

        ###################
        ### Computation ###
        ###################
        cost_est = 0
        cost_term = 0
        cost_contr = 0

        KF = KalmanFilter(self)        
        for k in range(M):
            # Initialize simulation and Kalman filter
            self.system.InitSimulation(X0[k])
            KF.InitSequence(X0[k], self.m2x_0)

            if my_noise is not None:
                q_noise = Q_noise[k]
                r_noise = R_noise[k]
            
            # Allocate tensors
            x_hat = torch.empty(self.m, self.T+1)
            x_hat[:,0] = X0[k]
            x_true = torch.empty_like(x_hat)
            x_true[:,0] = X0[k]
            u = torch.empty(self.p, self.T)

            # Simulate trajectory with Kalman filter as estimator
            for t in range(1, self.T+1):
                # LQR input
                dx = x_hat[:,t-1] - XT[k]
                if not steady_state:
                    u[:,t-1] = - torch.matmul(L[t-1], dx)
                else:
                    u[:,t-1] = - torch.matmul(L, dx)

                # Simulate one step
                if my_noise is not None:
                    y, x_true[:,t] = self.system.simulate_with_my_noise(u[:,t-1], q_noise[:, t-1], r_noise[:, t-1])
                else:
                    y, x_true[:,t] = self.system.simulate(u[:,t-1])

                # Estimate state with Kalman filter
                x_hat[:,t], _ = KF.Update(y, u[:,t-1])

            # Calculate cost for trajectory
            est, term, contr = self.RL_cost((x_true[:,-1], x_hat[:,-1], u), XT[k])
            cost_est += est / M
            cost_term += term / M
            cost_contr += contr / M
        
        return cost_est, cost_term, cost_contr


    def RL_cost(self, inputs, targets):
        xT, xT_hat, u = inputs
        x_target = targets

        ### Estimation Cost ###
        # Difference between true final state and its estimate
        dx_est = xT.squeeze() - xT_hat.squeeze()
        estimation_cost = dx_est.matmul(dx_est)

        ### Terminal Cost ###
        # Difference between true final state and target state
        dx_terminal = xT.squeeze() - x_target.squeeze()
        terminal_cost = dx_terminal.matmul(dx_terminal)

        # U = u.flatten()
        # QQu = torch.diag(torch.kron(torch.eye(T), self.Qu))
        # control_cost = U.matmul(QQu * U)

        ### Control Cost ###
        T = max(u.shape)
        control_cost = 0
        for t in range(T):
            control_cost += u[:,t].matmul(u[:,t])

        # info = f"est: {estimation_cost: .5f}, "\
        #        f"terminal: {terminal_cost: .5f}, "\
        #        f"control: {control_cost: .5f}"

        # print(info) 

        return estimation_cost, terminal_cost, control_cost


    def EstimateLQRCost_with_my_noise(self, x0, xT, X, U):
        '''
        Computes the average LQR cost of the data set.

        Parameters
        ----------
        x0 : tensor of shape (m,)
            Initial state
        xT : tensor of shape (m,)
            Target state
        X : tensor of shape (N, m, T)
            State trajectories
        U : tensor of shape (N, p, T)
            Input trajectories

        Returns
        -------
        cost : torch.float 
        '''        
        cost = 0
        M = X.shape[0]
        x0 = torch.unsqueeze(x0,1)
        for k in range(M):
            x = torch.cat((x0, X[k]), dim=1)
            trajectory_cost = self.LQR_cost(x, U[k], xT)
            cost += trajectory_cost / M

        cost_estimate = cost 
        return cost_estimate

    
    def EstimateLQGCost_with_y_as_state_estimate(self, X0, XT, Q_noise, R_noise):
        '''
        Does the same as "EstimateLQGCost_with_my_noise()" but uses the observation y instead of 
        x_hat to compute the control input. This is to investigate how much state estimation is needed
        for a good cost. 
        '''
        if self.m != self.n:
            print("State and observation dimension differ. Cannot use this function.")
            return None

        M = Q_noise.shape[0]
        cost = 0
        for k in range(M):
            KF = KalmanFilter(self)

            # Initialize simulation and Kalman filter
            self.system.InitSimulation(X0[k])
            KF.InitSequence(X0[k], self.m2x_0)

            q_noise = Q_noise[k]
            r_noise = R_noise[k]
            
            # Allocate tensors
            x_hat = torch.empty(self.m, self.T+1)
            x_hat[:,0] = X0[k]
            x_true = torch.empty_like(x_hat)
            x_true[:,0] = X0[k]
            y = torch.empty_like(x_true)
            y[:,0] = torch.matmul(self.H, X0[k]) 
            u = torch.empty(self.p, self.T)

            # Simulate trajectory with Kalman filter as estimator
            for t in range(1, self.T+1):
                # LQR input
                dx = y[:,t-1] - XT[k]
                u[:,t-1] = - torch.matmul(self.L[t-1], dx)

                # Simulate one step
                y[:,t], x_true[:,t] = self.system.simulate_with_my_noise(u[:,t-1], q_noise[:, t-1], r_noise[:, t-1])

                # Estimate state with Kalman filter
                x_hat[:,t], _ = KF.Update(y[:,t], u[:,t-1])

            # Calculate cost for trajectory
            trajectory_cost = self.LQR_cost(x_true, u, XT[k])
            cost += trajectory_cost
        
        cost_estimate = cost / M
        return cost_estimate


    def toDict(self):
        '''
        Returns a dict of relevant model parameters:
        F, G, H, Q, q, R, r, QN, Qx, Qu, T.
        Tensors are represented by lists for readability.
        '''
        model_dict = {
            'F' : self.F.tolist(),
            'G' : self.G.tolist(),
            'H' : self.H.tolist(),
            'Q' : self.Q.tolist(),
            'q' : self.q,
            'R' : self.R.tolist(),
            'r' : self.r,
            'QN' : self.QT.tolist(),
            'Qx' : self.Qx.tolist(),
            'Qu' : self.Qu.tolist(),
            'T' : self.T
        }

        return model_dict