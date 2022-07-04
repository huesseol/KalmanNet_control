import torch
import time
from lqr_loss import LQRLoss, ControlLoss
import torch.nn as nn
import random
from support_functions import mean_and_std_linear_and_dB

class Pipeline_KF_loop:

    def __init__(self, Time, folderName, modelName):
        super().__init__()
        self.Time = Time
        
        if folderName.endswith('/'):
          self.folderName = folderName
        else:
          self.folderName = folderName + '/'

        self.modelName = modelName
        self.modelFileName = self.folderName + "model_" + self.modelName 
        self.PipelineName = self.folderName + "pipeline_" + self.modelName
        self.LQR_cost = 0
        self.LQG_cost = 0 
        self.LQR_cost_true_system = 0
        self.LQG_cost_true_system = 0 

    def save(self):
        torch.save(self, self.PipelineName)

    def setssModel(self, ssModel):
        self.ssModel = ssModel

    def setModel(self, model):
        self.model = model

    def setTrainingParams(self, n_Epochs, n_Batch, learningRate, weightDecay, alpha=1.0, beta=1.0, gamma=1.0):
        self.N_Epochs = n_Epochs  # Number of Training Epochs
        self.N_B = n_Batch # Number of Samples in Batch
        self.learningRate = learningRate # Learning Rate
        self.weightDecay = weightDecay # L2 Weight Regularization - Weight Decay

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # LQR Loss Function
        self.lqr_loss_fn = LQRLoss(self.ssModel.QT, self.ssModel.Qx, self.ssModel.Qu, self.ssModel.T)
        # MSE Loss Function
        self.mse_loss_fn = nn.MSELoss(reduction='mean')
        # Control Loss Function
        self.control_loss_fn = ControlLoss(self.alpha, self.beta, self.gamma, self.ssModel.m, self.ssModel.p)

        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)

        # In case you want to use manual learning rate reduction
        # self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=lr_decay, verbose=True)

    def setControlLossParameters(self, alpha=None, beta=None, gamma=None, R_est=None, R_terminal=None, R_u=None):
        if alpha is not None:
            self.control_loss_fn.alpha = alpha
        if beta is not None:
            self.control_loss_fn.beta = beta
        if gamma is not None:
            self.control_loss_fn.gamma = gamma
        if R_est is not None:
            self.control_loss_fn.R_est = R_est
        if R_terminal is not None:
            self.control_loss_fn.R_terminal = R_terminal
        if R_u is not None:
            self.control_loss_fn.R_u = R_u
                           

    def NNTrain_state_access(self, inputs, targets, train_noise, val_noise, n_val=None, controller=False, num_restarts=0, T=100):
        '''
        Train the KalmanNet in the feedback loop on the loss L = alpha*MSE + beta*LQR. In each epoch, 
        simulate N_B trajectories with the current KalmanNet weights. Then backpropagate the error.

        Parameters
        ----------
        inputs: Tuple of two tensors (X0_train, X0_val)
            X0_train has shape (N_train, m) and X0_val has shape (N_val, m). These are the initial states of the simulation.
        targets: FloatTensor of shape (m, Nt)
            XT_train has shape (N_train, m) and XT_val has shape (N_val, m). These are the target states of the simulation.
        train_noise: Tuple of two tensors (train_Q, train_R)
            train_Q has shape (N_train, m, T) and train_R (N_train, n, T), 
        val_noise: Tuple of two tensors (val_Q, val_R)
            val_Q has shape (N_val, m, T) and val_R (N_val, n, T). 
        n_val: None or int
            If an integer is given then this is the number of samples are used validation in each epoch. If None then the 
            number available from the validation noise is used.
        T: int
            Length of trajectories used for training. Useful in case of model mismatch because long trajectories might 
            diverge. Default is T = 100.
        num_restarts: int
            Number of times the optimizer is re-initialized during training. Doing this can help overcome slow progress.
            Default is 0.
        controller: bool
            Indicates whether to use the controller derived from the model (False) or the true system (True). Default is False. 
        '''
        self.training_type = 'LQR + MSE'

        # Unpack data 
        X0_train, X0_val = inputs
        XT_train, XT_val = targets
        train_Q, train_R = train_noise
        val_Q, val_R = val_noise
        
        # Make sure desired training trajectory length is feasible
        assert T <= train_Q.shape[-1]

        self.N_train = train_Q.shape[0]
        self.N_CV = val_Q.shape[0]
        if n_val:
            self.N_CV = n_val

        Total_loss_train_linear_batch = torch.empty([self.N_B])
        self.Total_loss_train_linear_epoch = torch.empty([self.N_Epochs])
        self.Total_loss_train_dB_epoch = torch.empty([self.N_Epochs])

        Total_loss_val_batch = torch.empty([self.N_CV])
        self.Total_loss_val_epoch = torch.empty([self.N_Epochs])
        self.Total_loss_val_dB_epoch = torch.empty([self.N_Epochs])

        LQR_val_linear_batch = torch.empty([self.N_CV])
        self.LQR_val_linear_epoch = torch.empty([self.N_Epochs])
        self.LQR_val_dB_epoch = torch.empty([self.N_Epochs])

        MSE_val_batch = torch.empty([self.N_CV])
        self.MSE_val_epoch = torch.empty([self.N_Epochs])
        self.MSE_val_dB_epoch = torch.empty([self.N_Epochs])

        MSE_val_position_batch = torch.empty([self.N_CV])
        self.MSE_val_position_epoch = torch.empty([self.N_Epochs])
        self.MSE_val_position_dB_epoch = torch.empty([self.N_Epochs])

        self.Loss_val_dB_opt = 1000
        self.Loss_val_idx_opt = 0
        self.LQR_val_dB_opt = 1000
        self.LQR_val_idx_opt = 0
        self.MSE_val_dB_opt = 1000
        self.MSE_val_idx_opt = 0

        # Decide which controller to use: the one derived from the true system or the one 
        # derived from the possibly wrong model
        if controller:
            L = self.ssModel.L_true
        else:
            L = self.ssModel.L

        
        if num_restarts > 0:
            restart_every = int(self.N_Epochs / (num_restarts+1))

        ##############
        ### Epochs ###
        ##############

        for ti in range(0, self.N_Epochs):

            if num_restarts > 0:
                if ti % restart_every == 0:
                    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)

            ###############################
            ### Training Sequence Batch ###
            ###############################

            # Training Mode
            self.model.train()

            # Init Hidden State
            self.model.init_hidden()

            Batch_Optimizing_Loss_sum = 0

            # Simulate N_B trajectories with the current weights
            for j in range(0, self.N_B):
                
                # Select random noise sequence from training set
                idx = random.randint(0, self.N_train - 1)
                q_noise = train_Q[idx]
                r_noise = train_R[idx]
                x0_train = X0_train[idx]
                xT_train = XT_train[idx]

                # Initialize simulation and KalmanNet
                self.model.InitSequence(x0_train, T)
                self.ssModel.system.InitSimulation(x0_train)

                # Tensors for state estimates and inputs
                x_hat = torch.empty(self.ssModel.m, T + 1)
                x_hat[:,0] = x0_train
                x_true = torch.empty_like(x_hat)
                x_true[:,0] = x0_train
                u = torch.empty(self.ssModel.p, T)

                # Simulate trajectory
                for t in range(1, T + 1):
                    # Calculate LQR input
                    dx = x_hat[:,t-1] - xT_train
                    u[:,t-1] = - torch.matmul(L[t-1], dx) 
                    
                    # Simulate one step with LQR input 
                    y, x = self.ssModel.system.simulate_with_my_noise(u[:,t-1], q_noise[:, t-1], r_noise[:, t-1])
                    x_true[:,t] = x 

                    # Obtain state estimate from KalmanNet
                    x_hat[:,t] = self.model(y, u[:,t-1])

                # Compute loss for the trajectory
                Loss_lqr = self.lqr_loss_fn((x_true, u), xT_train)
                Loss_mse = self.mse_loss_fn(x_hat[:,1:], x_true[:,1:])

                Loss = self.alpha*Loss_mse + self.beta*Loss_lqr # combine MSE and LQR
                Total_loss_train_linear_batch[j] = Loss.item()

                Batch_Optimizing_Loss_sum = Batch_Optimizing_Loss_sum + Loss

            # Average
            self.Total_loss_train_linear_epoch[ti] = torch.mean(Total_loss_train_linear_batch)
            self.Total_loss_train_dB_epoch[ti] = 10 * torch.log10(self.Total_loss_train_linear_epoch[ti])


            ##################
            ### Optimizing ###
            ##################

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.
            self.optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            Batch_Optimizing_Loss_mean = Batch_Optimizing_Loss_sum / self.N_B
            Batch_Optimizing_Loss_mean.backward()

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            self.optimizer.step()


            #################################
            ### Validation Sequence Batch ###
            #################################

            # Cross Validation Mode
            self.model.eval()

            with torch.no_grad():

                for j in range(0, self.N_CV):

                    # Noise sequence to be used
                    q_noise = val_Q[j]
                    r_noise = val_R[j]
                    x0_val = X0_val[j]
                    xT_val = XT_val[j]

                    # Initialize simulation and KalmanNet
                    self.model.InitSequence(x0_val, T)
                    self.ssModel.system.InitSimulation(x0_val)

                    # Tensors for state estimates and inputs
                    x_hat = torch.empty(self.ssModel.m, T + 1)
                    x_hat[:,0] = x0_val
                    x_true = torch.empty_like(x_hat)
                    x_true[:,0] = x0_val
                    u = torch.empty(self.ssModel.p, T)

                    # Simulate trajectory
                    for t in range(1, T + 1):
                        # Calculate LQR input
                        dx = x_hat[:,t-1] - xT_val
                        u[:,t-1] = - torch.matmul(L[t-1], dx) 
                        
                        # Simulate one step with LQR input 
                        y, x = self.ssModel.system.simulate_with_my_noise(u[:,t-1], q_noise[:, t-1], r_noise[:, t-1]) 
                        x_true[:,t] = x

                        # Obtain state estimate from KalmanNet
                        x_hat[:,t] = self.model(y, u[:,t-1])

                    # Compute LQR Loss
                    LQR_val_linear_batch[j] = self.lqr_loss_fn((x_true, u), xT_val).item()

                    # MSE of state estimation
                    MSE_val_batch[j] = self.mse_loss_fn(x_hat[:,1:], x_true[:,1:]).item()
                    MSE_val_position_batch[j] = self.mse_loss_fn(x_hat[0,1:], x_true[0,1:]).item()

                    # Total loss: MSE + LQR
                    Total_loss_val_batch[j] = self.alpha*MSE_val_batch[j] + self.beta*LQR_val_linear_batch[j]


                # Average losses
                self.LQR_val_linear_epoch[ti] = torch.mean(LQR_val_linear_batch)
                self.LQR_val_dB_epoch[ti] = 10 * torch.log10(self.LQR_val_linear_epoch[ti])

                self.MSE_val_epoch[ti] = torch.mean(MSE_val_batch)
                self.MSE_val_dB_epoch[ti] = 10 * torch.log10(self.MSE_val_epoch[ti])
                self.MSE_val_position_epoch[ti] = torch.mean(MSE_val_position_batch)
                self.MSE_val_position_dB_epoch[ti] = 10 * torch.log10(self.MSE_val_position_epoch[ti])

                self.Total_loss_val_epoch[ti] = torch.mean(Total_loss_val_batch)
                self.Total_loss_val_dB_epoch[ti] = 10 * torch.log10(self.Total_loss_val_epoch[ti])

                # Save model in case of improvement
                if (self.Total_loss_val_dB_epoch[ti] < self.Loss_val_dB_opt):
                    self.Loss_val_dB_opt = self.Total_loss_val_dB_epoch[ti]
                    self.Loss_val_idx_opt = ti
                    torch.save(self.model, self.modelFileName)

                # Save best LQR model
                if (self.LQR_val_dB_epoch[ti] < self.LQR_val_dB_opt):
                    self.LQR_val_dB_opt = self.LQR_val_dB_epoch[ti]
                    self.LQR_val_idx_opt = ti
                    torch.save(self.model, self.modelFileName[:-3] + '_best_LQR.pt')

                # Save best MSE model
                if (self.MSE_val_dB_epoch[ti] < self.MSE_val_dB_opt):
                    self.MSE_val_dB_opt = self.MSE_val_dB_epoch[ti]
                    self.MSE_val_idx_opt = ti
                    torch.save(self.model, self.modelFileName[:-3] + '_best_MSE.pt')


            ########################
            ### Training Summary ###
            ########################
            
            if (ti > 0):
                d_val = self.Total_loss_val_dB_epoch[ti] - self.Total_loss_val_dB_epoch[ti - 1]
                d_mse = self.MSE_val_dB_epoch[ti] - self.MSE_val_dB_epoch[ti - 1] 
                d_lqr = self.LQR_val_dB_epoch[ti] - self.LQR_val_dB_epoch[ti - 1]
                info = f"{ti} LQG train: {self.Total_loss_train_dB_epoch[ti]: .5f} [dB], " \
                        f"LQG val: {self.Total_loss_val_dB_epoch[ti]: .5f} [dB], " \
                        f"LQR val: {self.LQR_val_dB_epoch[ti]: .5f} [dB], " \
                        f"MSE val: {self.MSE_val_dB_epoch[ti]: .5f} [dB]" \
                        f"diff LQG val: {d_val: .5f} [dB], diff LQR val: {d_lqr: .5f} [dB] , diff MSE val: {d_mse: .5f} [dB] " \
                        f"best idx: {self.Loss_val_idx_opt}, Best cost: {self.Loss_val_dB_opt: .5f} [dB] " \
                        f"best idx MSE: {self.MSE_val_idx_opt}, best MSE: {self.MSE_val_dB_opt: .5f} [dB]" \
                        f"best idx LQR: {self.LQR_val_idx_opt}, best LQR: {self.LQR_val_dB_opt: .5f} [dB]"
                print(info)
            else:
                print(f"{ti} LQG train : {self.Total_loss_train_dB_epoch[ti]: .5f} [dB], LQG val : {self.Total_loss_val_dB_epoch[ti]: .5f} [dB]")

            # If loss is nan stop
            if self.Total_loss_train_dB_epoch[ti].isnan():
                break


    def NNTrain_no_state_access(self, inputs, targets, train_noise, val_noise, n_val=None, num_restarts=0, T=100):
        # Unpack data 
        X0_train, X0_val = inputs
        XT_train, XT_val = targets
        train_Q, train_R = train_noise
        val_Q, val_R = val_noise

        self.N_train = train_Q.shape[0]
        self.N_CV = val_Q.shape[0]
        if n_val:
            self.N_CV = n_val

        Control_train_linear_batch = torch.empty([self.N_B])
        self.Control_train_linear_epoch = torch.empty([self.N_Epochs])
        self.Control_train_dB_epoch = torch.empty([self.N_Epochs])

        Control_val_linear_batch = torch.empty([self.N_CV])
        self.Control_val_linear_epoch = torch.empty([self.N_Epochs])
        self.Control_val_dB_epoch = torch.empty([self.N_Epochs])

        LQR_val_linear_batch = torch.empty([self.N_CV])
        self.LQR_val_linear_epoch = torch.empty([self.N_Epochs])
        self.LQR_val_dB_epoch = torch.empty([self.N_Epochs])

        MSE_val_batch = torch.empty([self.N_CV])
        self.MSE_val_epoch = torch.empty([self.N_Epochs])
        self.MSE_val_dB_epoch = torch.empty([self.N_Epochs])

        MSE_val_position_batch = torch.empty([self.N_CV])
        self.MSE_val_position_epoch = torch.empty([self.N_Epochs])
        self.MSE_val_position_dB_epoch = torch.empty([self.N_Epochs])

        self.Loss_val_dB_opt = 1000
        self.Loss_val_idx_opt = 0
        self.LQR_val_dB_opt = 1000
        self.LQR_val_idx_opt = 0
        self.MSE_val_dB_opt = 1000
        self.MSE_val_idx_opt = 0

        L = self.ssModel.L

        if num_restarts > 0:
            restart_every = int(self.N_Epochs / (num_restarts+1))

        ##############
        ### Epochs ###
        ##############

        for ti in range(0, self.N_Epochs):

            if num_restarts > 0:
                if ti % restart_every == 0:
                    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)

            ###############################
            ### Training Sequence Batch ###
            ###############################

            # Training Mode
            self.model.train()

            # Init Hidden State
            self.model.init_hidden()

            Batch_Optimizing_Loss_sum = 0

            # Simulate N_B trajectories with the current weights
            for j in range(0, self.N_B):
                
                # Select random noise sequence from training set
                idx = random.randint(0, self.N_train - 1)
                q_noise = train_Q[idx]
                r_noise = train_R[idx]
                x0_train = X0_train[idx]
                xT_train = XT_train[idx]

                # Initialize simulation and KalmanNet
                self.model.InitSequence(x0_train, T)
                self.ssModel.system.InitSimulation(x0_train)

                # Tensors for state estimates and inputs
                x_hat = torch.empty(self.ssModel.m, T + 1)
                x_hat[:,0] = x0_train
                x_true = torch.empty_like(x_hat)
                x_true[:,0] = x0_train
                u = torch.empty(self.ssModel.p, T)

                # Simulate trajectory
                for t in range(1, T + 1):
                    # Calculate LQR input
                    dx = x_hat[:,t-1] - xT_train
                    u[:,t-1] = - torch.matmul(L[t-1], dx) 
                    
                    # Simulate one step with LQR input 
                    y, x = self.ssModel.system.simulate_with_my_noise(u[:,t-1], q_noise[:, t-1], r_noise[:, t-1])
                    x_true[:,t] = x 

                    # Obtain state estimate from KalmanNet
                    x_hat[:,t] = self.model(y, u[:,t-1])

                # Compute loss for the trajectory
                loss = self.control_loss_fn((x_true[:,-1], x_hat[:,-1], u), xT_train)
                Control_train_linear_batch[j] = loss.item()               

                Batch_Optimizing_Loss_sum = Batch_Optimizing_Loss_sum + loss

            # Average
            self.Control_train_linear_epoch[ti] = torch.mean(Control_train_linear_batch)
            self.Control_train_dB_epoch[ti] = 10 * torch.log10(self.Control_train_linear_epoch[ti])


            ##################
            ### Optimizing ###
            ##################

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.
            self.optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            Batch_Optimizing_Loss_mean = Batch_Optimizing_Loss_sum / self.N_B
            Batch_Optimizing_Loss_mean.backward()

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            self.optimizer.step()


            #################################
            ### Validation Sequence Batch ###
            #################################

            # Cross Validation Mode
            self.model.eval()

            with torch.no_grad():

                for j in range(0, self.N_CV):

                    # Noise sequence to be used
                    q_noise = val_Q[j]
                    r_noise = val_R[j]
                    x0_val = X0_val[j]
                    xT_val = XT_val[j]

                    # Initialize simulation and KalmanNet
                    self.model.InitSequence(x0_val, T)
                    self.ssModel.system.InitSimulation(x0_val)

                    # Tensors for state estimates and inputs
                    x_hat = torch.empty(self.ssModel.m, T + 1)
                    x_hat[:,0] = x0_val
                    x_true = torch.empty_like(x_hat)
                    x_true[:,0] = x0_val
                    u = torch.empty(self.ssModel.p, T)

                    # Simulate trajectory
                    for t in range(1, T + 1):
                        # Calculate LQR input
                        dx = x_hat[:,t-1] - xT_val
                        u[:,t-1] = - torch.matmul(L[t-1], dx) 
                        
                        # Simulate one step with LQR input 
                        y, x = self.ssModel.system.simulate_with_my_noise(u[:,t-1], q_noise[:, t-1], r_noise[:, t-1]) 
                        x_true[:,t] = x

                        # Obtain state estimate from KalmanNet
                        x_hat[:,t] = self.model(y, u[:,t-1])

                    # Control Loss
                    Control_val_linear_batch[j] = self.control_loss_fn((x_true[:,-1], x_hat[:,-1], u), xT_val).item()

                    # LQR Loss
                    LQR_val_linear_batch[j] = self.lqr_loss_fn((x_true, u), xT_val).item()

                    # MSE of state estimation
                    MSE_val_batch[j] = self.mse_loss_fn(x_hat[:,1:], x_true[:,1:]).item()
                    MSE_val_position_batch[j] = self.mse_loss_fn(x_hat[0,1:], x_true[0,1:]).item()


                # Average losses
                self.Control_val_linear_epoch[ti] = torch.mean(Control_val_linear_batch)
                self.Control_val_dB_epoch[ti] = 10 * torch.log10(self.Control_val_linear_epoch[ti])

                self.LQR_val_linear_epoch[ti] = torch.mean(LQR_val_linear_batch)
                self.LQR_val_dB_epoch[ti] = 10 * torch.log10(self.LQR_val_linear_epoch[ti])

                self.MSE_val_epoch[ti] = torch.mean(MSE_val_batch)
                self.MSE_val_dB_epoch[ti] = 10 * torch.log10(self.MSE_val_epoch[ti])

                self.MSE_val_position_epoch[ti] = torch.mean(MSE_val_position_batch)
                self.MSE_val_position_dB_epoch[ti] = 10 * torch.log10(self.MSE_val_position_epoch[ti])

                # Save model in case of improvement
                if (self.Control_val_dB_epoch[ti] < self.Loss_val_dB_opt):
                    self.Loss_val_dB_opt = self.Control_val_dB_epoch[ti]
                    self.Loss_val_idx_opt = ti
                    torch.save(self.model, self.modelFileName)

                if (self.LQR_val_dB_epoch[ti] < self.LQR_val_dB_opt):
                    self.LQR_val_dB_opt = self.LQR_val_dB_epoch[ti]
                    self.LQR_val_idx_opt = ti

                if (self.MSE_val_dB_epoch[ti] < self.MSE_val_dB_opt):
                    self.MSE_val_dB_opt = self.MSE_val_dB_epoch[ti]
                    self.MSE_val_idx_opt = ti

            ########################
            ### Training Summary ###
            ########################
            
            if (ti > 0):
                d_val = self.Control_val_dB_epoch[ti] - self.Control_val_dB_epoch[ti - 1]
                d_mse = self.MSE_val_dB_epoch[ti] - self.MSE_val_dB_epoch[ti - 1] 
                d_lqr = self.LQR_val_dB_epoch[ti] - self.LQR_val_dB_epoch[ti - 1]
                info = f"{ti} Control train: {self.Control_train_dB_epoch[ti]: .5f} [dB], " \
                        f"Control val: {self.Control_val_dB_epoch[ti]: .5f} [dB], " \
                        f"LQR val: {self.LQR_val_dB_epoch[ti]: .5f} [dB], " \
                        f"MSE val: {self.MSE_val_dB_epoch[ti]: .5f} [dB]" \
                        f"diff Control val: {d_val: .5f} [dB], diff LQR val: {d_lqr: .5f} [dB] , diff MSE val: {d_mse: .5f} [dB] " \
                        f"best idx: {self.Loss_val_idx_opt}, Best cost: {self.Loss_val_dB_opt: .5f} [dB] " \
                        f"best idx MSE: {self.MSE_val_idx_opt}, best MSE: {self.MSE_val_dB_opt: .5f} [dB]" \
                        f"best idx LQR: {self.LQR_val_idx_opt}, best LQR: {self.LQR_val_dB_opt: .5f} [dB]"
                print(info)
            else:
                print(f"{ti} Control train: {self.Control_train_dB_epoch[ti]: .5f} [dB], Control val: {self.Control_val_dB_epoch[ti]: .5f} [dB]")

            # If loss is nan stop
            if self.Control_train_dB_epoch[ti].isnan():
                break


    def NNTest(self, X0, XT, noise, model=None):
        '''
        Returns a tuple of tuples:
        - LQR_loss_summary 
        - MSE_loss_total_summary 
        - MSE_loss_position_summary
        '''
        # Unpack noise 
        test_Q, test_R = noise
        self.N_test = test_Q.shape[0]

        self.LQR_test_linear_arr = torch.empty([self.N_test])
        self.MSE_test_arr = torch.empty([self.N_test])
        self.MSE_test_position_arr = torch.empty([self.N_test])

        if model:
            self.model = torch.load(model)
        else:
            self.model = torch.load(self.modelFileName)

        self.model.eval()

        with torch.no_grad():
        
            start = time.time()

            for j in range(0, self.N_test):
                q_noise = test_Q[j]
                r_noise = test_R[j]
                x0 = X0[j]
                xT = XT[j]
                
                # Initialize simulation and KalmanNet
                self.ssModel.system.InitSimulation(x0)
                self.model.InitSequence(x0, self.ssModel.T_test)
                
                # Tensors for state estimates and inputs
                x_hat = torch.empty(self.ssModel.m, self.ssModel.T_test + 1)
                x_hat[:,0] = x0
                x_true = torch.empty_like(x_hat)
                x_true[:,0] = x0
                u = torch.empty(self.ssModel.p, self.ssModel.T_test)

                for t in range(1, self.ssModel.T + 1):
                    # Calculate LQR input
                    dx = x_hat[:,t-1] - xT
                    u[:,t-1] = - torch.matmul(self.ssModel.L[t-1], dx) 
                    
                    # Simulate one step with LQR input
                    y, x = self.ssModel.system.simulate_with_my_noise(u[:,t-1], q_noise[:, t-1], r_noise[:, t-1])
                    x_true[:,t] = x

                    # Obtain state estimate from KalmanNet
                    x_hat[:,t] = self.model(y, u[:,t-1])

                # Compute cost for the trajectory
                self.LQR_test_linear_arr[j] = self.lqr_loss_fn((x_true, u), xT).item()

                # MSE of state estimate
                self.MSE_test_arr[j] = self.mse_loss_fn(x_hat[:,1:], x_true[:,1:])
                self.MSE_test_position_arr[j] = self.mse_loss_fn(x_hat[0,1:], x_true[0,1:])

            end = time.time()
            t = end - start

            # Average and standard deviation
            self.LQR_test_linear_avg, self.LQR_test_dB_avg, self.LQR_test_std, self.LQR_test_dB_std = mean_and_std_linear_and_dB(self.LQR_test_linear_arr)
            self.MSE_test_avg, self.MSE_test_dB_avg, self.MSE_test_std, self.MSE_test_dB_std = mean_and_std_linear_and_dB(self.MSE_test_arr)
            self.MSE_test_position_avg, self.MSE_test_position_dB_avg, self.MSE_test_position_std, self.MSE_test_position_dB_std = mean_and_std_linear_and_dB(self.MSE_test_position_arr)

            
        print(f"{self.modelName} - LQR Test: {self.LQR_test_dB_avg} [dB], STD: {self.LQR_test_dB_std} [dB]")
        print(f"{self.modelName} - MSE Test: {self.MSE_test_dB_avg} [dB], STD: {self.MSE_test_dB_std} [dB]")
        print(f"{self.modelName} - Position MSE Test: {self.MSE_test_position_dB_avg} [dB], STD: {self.MSE_test_position_std} [dB]")
        print("Inference Time:", t)

        LQR_loss_summary = (self.LQR_test_linear_arr, self.LQR_test_linear_avg, self.LQR_test_dB_avg)
        MSE_loss_total_summary = (self.MSE_test_arr, self.MSE_test_avg, self.MSE_test_dB_avg)
        MSE_loss_position_summary = (self.MSE_test_position_arr, self.MSE_test_position_avg, self.MSE_test_position_dB_avg)

        return LQR_loss_summary, MSE_loss_total_summary, MSE_loss_position_summary


    def NNTest2(self, X0, XT, noise, controller=False, model=None):
        '''
        Returns a tuple of tuples:
        - Total_loss_summary 
        - LQR_loss_summary 
        - MSE_loss_total_summary 
        - MSE_loss_position_summary
        '''
        # Unpack noise 
        test_Q, test_R = noise
        self.N_test = test_Q.shape[0]

        self.LQR_test_linear_arr = torch.empty([self.N_test])
        self.MSE_test_arr = torch.empty([self.N_test])
        self.MSE_test_position_arr = torch.empty([self.N_test])
        self.Total_loss_test_arr = torch.empty([self.N_test])

        if model:
            self.model = torch.load(model)
        else:
            self.model = torch.load(self.modelFileName)

        # Decide which controller to use: the one derived from the true system or the one 
        # derived from the possibly wrong model
        if controller:
            L = self.ssModel.L_true
        else:
            L = self.ssModel.L

        self.model.eval()

        with torch.no_grad():
        
            start = time.time()

            for j in range(0, self.N_test):
                q_noise = test_Q[j]
                r_noise = test_R[j]
                x0 = X0[j]
                xT = XT[j]
                
                # Initialize simulation and KalmanNet
                self.ssModel.system.InitSimulation(x0)
                self.model.InitSequence(x0, self.ssModel.T_test)
                
                # Tensors for state estimates and inputs
                x_hat = torch.empty(self.ssModel.m, self.ssModel.T_test + 1)
                x_hat[:,0] = x0
                x_true = torch.empty_like(x_hat)
                x_true[:,0] = x0
                u = torch.empty(self.ssModel.p, self.ssModel.T_test)

                for t in range(1, self.ssModel.T + 1):
                    # Calculate LQR input
                    dx = x_hat[:,t-1] - xT
                    u[:,t-1] = - torch.matmul(L[t-1], dx) 
                    
                    # Simulate one step with LQR input
                    y, x = self.ssModel.system.simulate_with_my_noise(u[:,t-1], q_noise[:, t-1], r_noise[:, t-1])
                    x_true[:,t] = x

                    # Obtain state estimate from KalmanNet
                    x_hat[:,t] = self.model(y, u[:,t-1])

                # Compute cost for the trajectory
                self.LQR_test_linear_arr[j] = self.lqr_loss_fn((x_true, u), xT).item()

                # MSE of state estimate
                self.MSE_test_arr[j] = self.mse_loss_fn(x_hat[:,1:], x_true[:,1:])
                self.MSE_test_position_arr[j] = self.mse_loss_fn(x_hat[0,1:], x_true[0,1:])

                # Total loss
                self.Total_loss_test_arr[j] = self.alpha*self.MSE_test_arr[j] \
                                                + self.beta*self.LQR_test_linear_arr[j]

            end = time.time()
            t = end - start

            # Average
            self.LQR_test_linear_avg = torch.mean(self.LQR_test_linear_arr)
            self.LQR_test_dB_avg = 10 * torch.log10(self.LQR_test_linear_avg)

            self.MSE_test_avg = torch.mean(self.MSE_test_arr)
            self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_avg)
            
            self.MSE_test_position_avg = torch.mean(self.MSE_test_position_arr)
            self.MSE_test_position_dB_avg = 10 * torch.log10(self.MSE_test_position_avg)

            self.Total_loss_test_avg = torch.mean(self.Total_loss_test_arr)
            self.Total_loss_test_dB_avg = 10 * torch.log10(self.Total_loss_test_avg)

            # Standard deviation
            self.LQR_test_std = torch.std(self.LQR_test_linear_arr, unbiased=True)
            self.LQR_test_dB_std = 10 * torch.log10(self.LQR_test_linear_avg + self.LQR_test_std) - self.LQR_test_dB_avg

            self.MSE_test_std = torch.std(self.MSE_test_arr, unbiased=True)
            self.MSE_test_dB_std = 10 * torch.log10(self.MSE_test_avg + self.MSE_test_std) - self.MSE_test_dB_avg

            self.MSE_test_position_std = torch.std(self.MSE_test_position_arr, unbiased=True)
            self.MSE_test_position_dB_std = 10 * torch.log10(self.MSE_test_position_avg + self.MSE_test_position_std) - self.MSE_test_position_dB_avg

            self.Total_loss_test_std = torch.std(self.Total_loss_test_arr, unbiased=True)
            self.Total_loss_test_dB_std = 10 * torch.log10(self.Total_loss_test_avg + self.Total_loss_test_std) - self.Total_loss_test_dB_avg


        print(f"{self.modelName} - LQG Test: {self.Total_loss_test_dB_avg} [dB], STD: {self.Total_loss_test_dB_std} [dB]")
        print(f"{self.modelName} - LQR Test: {self.LQR_test_dB_avg} [dB], STD: {self.LQR_test_dB_std} [dB]")
        print(f"{self.modelName} - MSE Test: {self.MSE_test_dB_avg} [dB], STD: {self.MSE_test_dB_std} [dB]")
        print(f"{self.modelName} - Position MSE Test: {self.MSE_test_position_dB_avg} [dB], STD: {self.MSE_test_position_dB_std} [dB]")
        print("Inference Time:", t)

        Total_loss_summary = (self.Total_loss_test_arr, self.Total_loss_test_avg, self.Total_loss_test_dB_avg)
        LQR_loss_summary = (self.LQR_test_linear_arr, self.LQR_test_linear_avg, self.LQR_test_dB_avg)
        MSE_loss_total_summary = (self.MSE_test_arr, self.MSE_test_avg, self.MSE_test_dB_avg)
        MSE_loss_position_summary = (self.MSE_test_position_arr, self.MSE_test_position_avg, self.MSE_test_position_dB_avg)

        return Total_loss_summary, LQR_loss_summary, MSE_loss_total_summary, MSE_loss_position_summary


    