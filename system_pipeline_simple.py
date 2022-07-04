import torch
import torch.nn as nn
import random
import time
from Linear_KF import KalmanFilter
from support_functions import mean_and_std_linear_and_dB


class Pipeline_KF_simple:

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


    def save(self):
        torch.save(self, self.PipelineName)

    def setssModel(self, ssModel):
        self.ssModel = ssModel

    def setModel(self, model):
        self.model = model

    def setTrainingParams(self, n_Epochs, n_Batch, learningRate, weightDecay):
        self.N_Epochs = n_Epochs  # Number of Training Epochs
        self.N_B = n_Batch # Number of Samples in Batch
        self.learningRate = learningRate # Learning Rate
        self.weightDecay = weightDecay # L2 Weight Regularization - Weight Decay

        # MSE LOSS Function
        self.loss_fn = nn.MSELoss(reduction='mean')

        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)


    def NNTrain(self, train_input, train_target, val_input, val_target, n_val=None, num_restarts=0):
        '''
        Trains a KalmanNet offline with collected trajectories.

        Parameters
        ----------
        train_input: tuple of tensors
            Observation trajectories, control input trajectories and initial states (Y,U,X0) used for training. 
            Y has shape (N_train, n, T), U has shape (N_train, p, T), X0 has shape (N_train, m)
        train_target: tensor
            Trajectories of true states X. X has shape (N_train, m, T) 
        val_input: tuple of tensors
            Observation trajectories, control input trajectories and initial states (Y,U,X0) used for validation. 
            Y has shape (N_val, n, T), U has shape (N_val, p, T), X0 has shape (N_val, m)
        val_target: tensor
            Trajectories of true states X. X has shape (N_val, m, T)
        n_val: None or int
            If an integer is given then this is the number of samples are used validation in each epoch. If None then the 
            number available from the validation set (N_val) is used.
        num_restarts: int
            Number of times the optimizer is re-initialized during training. Doing this can help overcome slow progress.
        '''
        # Unpack training inputs
        train_input_y, train_input_u, X0_train = train_input
        val_input_y, val_input_u, X0_val = val_input
        
        self.N_train = train_input_y.shape[0]
        self.N_CV = val_input_y.shape[0]
        if n_val:
            self.N_CV = n_val

        MSE_val_linear_batch = torch.empty([self.N_CV])
        self.MSE_val_linear_epoch = torch.empty([self.N_Epochs])
        self.MSE_val_dB_epoch = torch.empty([self.N_Epochs])

        MSE_val_position_batch = torch.empty([self.N_CV])
        self.MSE_val_position_epoch = torch.empty([self.N_Epochs])
        self.MSE_val_position_dB_epoch = torch.empty([self.N_Epochs])

        MSE_train_linear_batch = torch.empty([self.N_B])
        self.MSE_train_linear_epoch = torch.empty([self.N_Epochs])
        self.MSE_train_dB_epoch = torch.empty([self.N_Epochs])

        if num_restarts > 0:
            restart_every = int(self.N_Epochs / (num_restarts+1))

        ##############
        ### Epochs ###
        ##############

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

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

            Batch_Optimizing_LOSS_sum = 0

            # Select N_B examples (trajectories) uniformly at random from all samples to form batch
            for j in range(0, self.N_B):
                idx = random.randint(0, self.N_train - 1)

                y_training = train_input_y[idx, :, :]
                u_training = train_input_u[idx, :, :]
                self.model.InitSequence(X0_train[idx], self.ssModel.T)

                x_hat_training = torch.empty(self.ssModel.m, self.ssModel.T)
                for t in range(0, self.ssModel.T):
                    x_hat_training[:, t] = self.model(y_training[:, t], u_training[:, t])

                # Compute Training Loss
                LOSS = self.loss_fn(x_hat_training, train_target[idx, :, :])
                MSE_train_linear_batch[j] = LOSS.item()

                Batch_Optimizing_LOSS_sum = Batch_Optimizing_LOSS_sum + LOSS

            # Average
            self.MSE_train_linear_epoch[ti] = torch.mean(MSE_train_linear_batch)
            self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])

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
            Batch_Optimizing_LOSS_mean = Batch_Optimizing_LOSS_sum / self.N_B
            Batch_Optimizing_LOSS_mean.backward()

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
                    y_val = val_input_y[j, :, :]
                    u_val = val_input_u[j, :, :]
                    self.model.InitSequence(X0_val[j], self.ssModel.T)

                    x_hat_val = torch.empty(self.ssModel.m, self.ssModel.T)
                    for t in range(0, self.ssModel.T):
                        x_hat_val[:, t] = self.model(y_val[:, t], u_val[:, t])

                    # Compute Validation Loss
                    MSE_val_linear_batch[j] = self.loss_fn(x_hat_val, val_target[j, :, :]).item()
                    MSE_val_position_batch[j] = self.loss_fn(x_hat_val[0, :], val_target[j, 0, :]).item()

                # Average
                self.MSE_val_linear_epoch[ti] = torch.mean(MSE_val_linear_batch)
                self.MSE_val_dB_epoch[ti] = 10 * torch.log10(self.MSE_val_linear_epoch[ti])
                self.MSE_val_position_epoch[ti] = torch.mean(MSE_val_position_batch)
                self.MSE_val_position_dB_epoch[ti] = 10 * torch.log10(self.MSE_val_position_epoch[ti])

                if (self.MSE_val_dB_epoch[ti] < self.MSE_cv_dB_opt):
                    self.MSE_cv_dB_opt = self.MSE_val_dB_epoch[ti]
                    self.MSE_cv_idx_opt = ti
                    torch.save(self.model, self.modelFileName)


            ########################
            ### Training Summary ###
            ########################
            if (ti > 1):
                # d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
                d_cv = self.MSE_val_dB_epoch[ti] - self.MSE_val_dB_epoch[ti - 1]
                info = f"{ti} MSE train: {self.MSE_train_dB_epoch[ti]: .6f} [dB] " \
                        f"MSE val: {self.MSE_val_dB_epoch[ti]: .6f} [dB], diff MSE val: {d_cv: .6f} [dB], " \
                        f"Optimal idx: {self.MSE_cv_idx_opt}, Optimal MSE val: {self.MSE_cv_dB_opt: .6f} [dB]"
                print(info)        
            else:
                print(f"{ti} MSE train: {self.MSE_train_dB_epoch[ti]: .6f} [dB], MSE val: {self.MSE_val_dB_epoch[ti]: .6f} [dB], Optimal idx: {self.MSE_cv_idx_opt}, Optimal MSE val: {self.MSE_cv_dB_opt: .6f} [dB]")
           
            # if there is no training progress
            if ti > 1 and self.MSE_train_dB_epoch[ti].isnan():
                break
            if ti == self.N_Epochs - 1:
                print(f'Kalman Gain: {self.model.KGain}')


    def NNTest(self, test_input, test_target, model=None):

        # Unpack test inputs
        test_input_y, test_input_u, X0_test = test_input

        self.N_test = test_input_y.shape[0]

        self.MSE_test_arr_knet = torch.empty([self.N_test])
        self.MSE_test_position_arr_knet = torch.empty([self.N_test])
        self.MSE_test_arr_kf = torch.empty([self.N_test])
        self.MSE_test_position_arr_kf = torch.empty([self.N_test])

        # MSE LOSS Function
        loss_fn = nn.MSELoss(reduction='mean')

        if model:
            self.model = torch.load(model)
        else:
            self.model = torch.load(self.modelFileName)

        self.model.eval()

        # Evaluate Kalman filter for reference
        KF = KalmanFilter(self.ssModel)

        with torch.no_grad(): 
        
            start = time.time()

            for j in range(0, self.N_test):
                # Select trajectories
                y_test = test_input_y[j, :, :]
                u_test = test_input_u[j, :, :]

                # Initialize KalmanNet and Kalman filter
                self.model.InitSequence(X0_test[j], self.ssModel.T_test)
                KF.InitSequence(X0_test[j], self.ssModel.m2x_0)

                x_hat_knet = torch.empty(self.ssModel.m, self.ssModel.T)
                x_hat_kf = torch.empty(self.ssModel.m, self.ssModel.T)

                for t in range(0, self.ssModel.T):
                    x_hat_knet[:, t] = self.model(y_test[:, t], u_test[:, t])
                    x_hat_kf[:,t], _ = KF.Update(y_test[:, t], u_test[:, t])

                self.MSE_test_arr_knet[j] = loss_fn(x_hat_knet, test_target[j, :, :]).item()
                self.MSE_test_position_arr_knet[j] = loss_fn(x_hat_knet[0, :], test_target[j, 0, :]).item()
                self.MSE_test_arr_kf[j] = loss_fn(x_hat_kf, test_target[j, :, :]).item()
                self.MSE_test_position_arr_kf[j] = loss_fn(x_hat_kf[0, :], test_target[j, 0, :]).item()

            end = time.time()
            t = end - start

            # Mean and standard deviation
            self.MSE_test_avg_knet, self.MSE_test_dB_avg_knet, self.MSE_test_std_knet, self.MSE_test_dB_std_knet = mean_and_std_linear_and_dB(self.MSE_test_arr_knet)
            self.MSE_test_position_avg_knet, self.MSE_test_position_dB_avg_knet, self.MSE_test_position_std_knet, self.MSE_test_position_dB_std_knet = mean_and_std_linear_and_dB(self.MSE_test_position_arr_knet)
            self.MSE_test_avg_kf, self.MSE_test_dB_avg_kf, self.MSE_test_std_kf, self.MSE_test_dB_std_kf = mean_and_std_linear_and_dB(self.MSE_test_arr_kf)
            self.MSE_test_position_avg_kf, self.MSE_test_position_dB_avg_kf, self.MSE_test_position_std_kf, self.MSE_test_position_dB_std_kf = mean_and_std_linear_and_dB(self.MSE_test_position_arr_kf)
            
        # Print MSE test results
        print(f"{self.modelName} - Total MSE Test: {self.MSE_test_dB_avg_knet: .5f} [dB], STD: {self.MSE_test_dB_std_knet: .5f} [dB]")
        print(f"{self.modelName} - Position MSE Test: {self.MSE_test_position_dB_avg_knet: .5f} [dB], STD: {self.MSE_test_position_dB_std_knet: .5f} [dB]")
        print(f"Kalman filter - Total MSE Test: {self.MSE_test_dB_avg_kf: .5f} [dB], STD: {self.MSE_test_dB_std_kf: .5f} [dB]")
        print(f"Kalman filter - Position MSE Test: {self.MSE_test_position_dB_avg_kf: .5f} [dB], STD: {self.MSE_test_position_dB_std_kf: .5f} [dB]")

        # Print Run Time
        print("Inference Time:", t)

        return [self.MSE_test_arr_knet, self.MSE_test_dB_avg_knet, self.MSE_test_position_arr_knet, self.MSE_test_position_dB_avg_knet]

