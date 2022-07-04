import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from math import log10

# Legend
Klegend = ["KNet - Train", "KNet - Validation", "KNet - Test", "Kalman Filter"]
loop_legend = ["KNet - Train", "KNet - Validation", "KNet - Test", "LQG", "LQR", "LQG - true system", "LQR - true system"]

# Color
KColor = ['-ro', 'k-', 'b-', 'g-', 'y']

class Plot:
    
    def __init__(self, pipeline):
        self.pipeline = pipeline


    def plot_epochs_simple(self, MSE_KF_true_system=None, fontSize=32, lineWidth=2, title=None, saveName=None, ylim=None, color=['-ro', 'k-', 'b-', 'g-', 'y-']):
        if title is None:
            title = self.pipeline.modelName

        # Figure
        plt.figure(figsize = (25, 10))

        x_plt = torch.tensor(range(0, self.pipeline.N_Epochs))

        y_plt1 = self.pipeline.MSE_train_dB_epoch[x_plt]
        plt.plot(x_plt, y_plt1, color[0], label=Klegend[0], linewidth=lineWidth)

        y_plt2 = self.pipeline.MSE_val_dB_epoch[x_plt]
        plt.plot(x_plt, y_plt2, color[1], label=Klegend[1], linewidth=lineWidth)

        y_plt3 = self.pipeline.MSE_test_dB_avg_knet * torch.ones_like(x_plt)
        plt.plot(x_plt, y_plt3, color[2], label=Klegend[2], linewidth=lineWidth)

        y_plt4 = self.pipeline.MSE_test_dB_avg_kf * torch.ones_like(x_plt)
        plt.plot(x_plt, y_plt4, color[3], label=Klegend[3], linewidth=lineWidth)

        if MSE_KF_true_system is not None:
            y_plt5 = MSE_KF_true_system * torch.ones_like(x_plt)
            plt.plot(x_plt, y_plt5, color[4], label='Kalman Filter - true system', linewidth=lineWidth)
        
        if ylim:
            plt.ylim(ylim)

        plt.legend(fontsize=fontSize)
        plt.xlabel('Epoch', fontsize=fontSize)
        plt.ylabel('MSE Loss Value [dB]', fontsize=fontSize)
        plt.title(title + "MSE Loss [dB] - per Epoch", fontsize=fontSize)
        
        if saveName:
            plt.savefig(self.pipeline.folderName + saveName)


    def plot_epochs_simple_new_data(self, N_epochs, MSE_KF, MSE_train_epoch, MSE_val_epoch, MSE_test, options):
        ylim = options['ylim']
        color = options['color']
        lineWidth = options['linewidth']
        legend = options['legend']
        fontSize = options['fontsize']
        title = options['title']
        saveName = options['saveName']

        # Figure
        plt.figure(figsize = (25, 10))

        x_plt = torch.tensor(range(0, N_epochs))

        y_plt1 = MSE_train_epoch[x_plt]
        plt.plot(x_plt, y_plt1, color[0], label=legend[0], linewidth=lineWidth)

        y_plt2 = MSE_val_epoch[x_plt]
        plt.plot(x_plt, y_plt2, color[1], label=legend[1], linewidth=lineWidth)

        y_plt3 = MSE_test * torch.ones_like(x_plt)
        plt.plot(x_plt, y_plt3, color[2], label=legend[2], linewidth=lineWidth)

        y_plt4 = MSE_KF * torch.ones_like(x_plt)
        plt.plot(x_plt, y_plt4, color[3], label=legend[3], linewidth=lineWidth)
        
        if ylim:
            plt.ylim(ylim)

        plt.legend(fontsize=fontSize)
        plt.xlabel('Epoch', fontsize=fontSize)
        plt.ylabel('MSE Loss Value [dB]', fontsize=fontSize)
        plt.title(title + ": " + "MSE Loss [dB] - per Epoch", fontsize=fontSize)
        
        if saveName:
            plt.savefig(self.pipeline.folderName + saveName)


    def plot_hist(self, MSE_KF_linear_arr, MSE_KNet_linear_arr, fontSize=32, title=None, saveName=None):
        plt.figure(figsize=(25, 10))
        
        sns.kdeplot(10*torch.log10(MSE_KF_linear_arr), color='r', linewidth=3, label='Kalman Filter')
        sns.kdeplot(10*torch.log10(MSE_KNet_linear_arr), color='g', linewidth=3, label='KalmanNet')
        
        plt.legend(fontsize=fontSize)
        plt.xlabel('MSE [dB]', fontsize=fontSize)
        
        if title is None:
            title = "Histogram [dB]"
        plt.title(title, fontsize=fontSize)
        
        if saveName:
            plt.savefig(self.pipeline.folderName + saveName)
    

    def plot_epochs_position_mse(self, MSE_KF, figSize=(25,8), fontSize=32, lineWidth=2, title=None, saveName=None, ylim=None, color=['k-', 'b-', 'g-', 'y-']):

        if title is None:
            title = self.pipeline.modelName

        # Figure
        plt.figure(figsize = figSize)

        x_plt = torch.tensor(range(0, self.pipeline.N_Epochs))

        # MSE validation
        y_plt1 = self.pipeline.MSE_val_position_dB_epoch[x_plt]
        plt.plot(x_plt, y_plt1, color[0], label="KNet - Validation", linewidth=lineWidth)

        # MSE test
        y_plt2 = self.pipeline.MSE_test_position_dB_avg_knet * torch.ones_like(x_plt)
        plt.plot(x_plt, y_plt2, color[1], label="KNet - Test", linewidth=lineWidth)

        # MSE Kalman filter
        y_plt3 = MSE_KF * torch.ones_like(x_plt)
        plt.plot(x_plt, y_plt3, color[2], label="KF - Test", linewidth=lineWidth)

        # Noise level
        y_plt4 = 10 * log10(self.pipeline.ssModel.r2) * torch.ones_like(x_plt)
        plt.plot(x_plt, y_plt4, color[3], label="Observation noise variance", linewidth=lineWidth)
        
        if ylim:
            plt.ylim(ylim)

        plt.legend(fontsize=fontSize)
        plt.xlabel('Epoch', fontsize=fontSize)
        plt.ylabel('Loss Value [dB]', fontsize=fontSize)
        plt.title(title + "Position MSE Loss [dB] - per Epoch", fontsize=fontSize)
        
        if saveName:
            plt.savefig(self.pipeline.folderName + saveName)


    def plot_loop_combined(self, LQR_cost_dB, LQG_cost_KF_dB, MSE_KF, figSize=(25, 25), fontSize=32, 
        lineWidth=2, title=None, saveName=None, ylim1=None, ylim2=None, color=['r-o','k-', 'b-', 'g-', 'y-']):
        
        if title is None:
            title = f"lr = {self.pipeline.learningRate}, weight decay = {self.pipeline.weightDecay}, batch size = {self.pipeline.N_B}"

        # Figure
        f = plt.figure(figsize = figSize)
        f.suptitle(title, fontsize=fontSize+4)
        ax1 = f.add_subplot(2,1,1)
        ax2 = f.add_subplot(2,1,2)

        x_plt = torch.tensor(range(0, self.pipeline.N_Epochs))

        # LQR training loss each epoch
        y_plt1 = self.pipeline.LQR_train_dB_epoch[x_plt]
        ax1.plot(x_plt, y_plt1, color[0], label=loop_legend[0], linewidth=lineWidth)

        # LQR validation loss each epoch
        y_plt2 = self.pipeline.LQR_val_dB_epoch[x_plt]
        ax1.plot(x_plt, y_plt2, color[1], label=loop_legend[1], linewidth=lineWidth)

        # LQR test loss
        y_plt3 = self.pipeline.LQR_test_dB_avg * torch.ones_like(x_plt)
        ax1.plot(x_plt, y_plt3, color[2], label=loop_legend[2], linewidth=lineWidth)

        # LQR loss when using a Kalman filter (i.e. LQG)
        y_plt4 = LQG_cost_KF_dB * torch.ones_like(x_plt)
        ax1.plot(x_plt, y_plt4, color[3], label=loop_legend[3], linewidth=lineWidth)

        # LQR loss when knowing the state
        y_plt5 = LQR_cost_dB * torch.ones_like(x_plt)
        ax1.plot(x_plt, y_plt5, color[4], label=loop_legend[4], linewidth=lineWidth)

        if ylim1:
            ax1.set_ylim(ylim1)

        ax1.legend(fontsize=fontSize)
        # ax1.set_xlabel('Epoch', fontsize=fontSize)
        ax1.set_ylabel('LQR Loss Value [dB]', fontsize=fontSize)
        ax1.set_title("LQR Loss [dB] - per Epoch", fontsize=fontSize)


        # MSE validation
        y_plt1 = self.pipeline.MSE_val_dB_epoch[x_plt]
        ax2.plot(x_plt, y_plt1, color[1], label="KNet - Validation", linewidth=lineWidth)

        # MSE test
        y_plt2 = self.pipeline.MSE_test_dB_avg * torch.ones_like(x_plt)
        ax2.plot(x_plt, y_plt2, color[2], label="KNet - Test", linewidth=lineWidth)

        # MSE Kalman filter
        y_plt3 = MSE_KF * torch.ones_like(x_plt)
        ax2.plot(x_plt, y_plt3, color[3], label="KF - Test", linewidth=lineWidth)

        # Noise level
        y_plt4 = 10 * log10(self.pipeline.ssModel.r2) * torch.ones_like(x_plt)
        ax2.plot(x_plt, y_plt4, color[4], label="Noise variance", linewidth=lineWidth)
        
        if ylim2:
            ax2.set_ylim(ylim2)

        ax2.legend(fontsize=fontSize)
        ax2.set_xlabel('Epoch', fontsize=fontSize)
        ax2.set_ylabel('MSE Loss Value [dB]', fontsize=fontSize)
        ax2.set_title("MSE Loss [dB] - per Epoch", fontsize=fontSize)
        
        if saveName:
            f.savefig(self.pipeline.folderName + saveName)
    

    def plot_loop_lqr_and_mse(self, MSE_KF, figSize=(25, 25), fontSize=32, 
        lineWidth=2, title=None, saveName=None, ylim1=None, ylim2=None, ylim3=None, color=['r-o','k-', 'b-', 'g-', 'y-']):

        # Figure
        f = plt.figure(figsize = figSize)
        f.suptitle(title, fontsize=fontSize+4)
        ax1 = f.add_subplot(3,1,1) # Total loss = LQR + MSE
        ax2 = f.add_subplot(3,1,2) # LQR
        ax3 = f.add_subplot(3,1,3) # MSE

        x_plt = torch.tensor(range(0, self.pipeline.N_Epochs))

        # Total training loss each epoch
        y_plt1 = self.pipeline.Total_loss_train_dB_epoch[x_plt]
        ax1.plot(x_plt, y_plt1, color[0], label=loop_legend[0], linewidth=lineWidth)

        # Total validation loss each epoch
        y_plt2 = self.pipeline.Total_loss_val_dB_epoch[x_plt]
        ax1.plot(x_plt, y_plt2, color[1], label=loop_legend[1], linewidth=lineWidth)

        # Total test loss
        y_plt3 = self.pipeline.Total_loss_test_dB_avg * torch.ones_like(x_plt)
        ax1.plot(x_plt, y_plt3, color[2], label=loop_legend[2], linewidth=lineWidth)

        if ylim1:
            ax1.set_ylim(ylim1)

        ax1.legend(fontsize=fontSize)
        # ax1.set_xlabel('Epoch', fontsize=fontSize)
        ax1.set_ylabel('Loss Value [dB]', fontsize=fontSize)
        ax1.set_title(f"Total Loss: {self.pipeline.alpha}*MSE + {self.pipeline.beta}*LQR [dB] - per Epoch", fontsize=fontSize)


        # LQR validation loss each epoch
        y_plt2 = self.pipeline.LQR_val_dB_epoch[x_plt]
        ax2.plot(x_plt, y_plt2, color[1], label=loop_legend[1], linewidth=lineWidth)

        # LQR test loss
        y_plt3 = self.pipeline.LQR_test_dB_avg * torch.ones_like(x_plt)
        ax2.plot(x_plt, y_plt3, color[2], label=loop_legend[2], linewidth=lineWidth)

        # LQR loss when using a Kalman filter (i.e. LQG)
        y_plt4 = self.pipeline.LQG_cost * torch.ones_like(x_plt)
        ax2.plot(x_plt, y_plt4, color[3], label=loop_legend[3], linewidth=lineWidth)

        # LQR loss when knowing the state
        y_plt5 = self.pipeline.LQR_cost * torch.ones_like(x_plt)
        ax2.plot(x_plt, y_plt5, color[4], label=loop_legend[4], linewidth=lineWidth)

        if ylim2:
            ax2.set_ylim(ylim2)

        ax2.legend(fontsize=fontSize)
        # ax2.set_xlabel('Epoch', fontsize=fontSize)
        ax2.set_ylabel('Loss Value [dB]', fontsize=fontSize)
        ax2.set_title("LQR Loss [dB] - per Epoch", fontsize=fontSize)


        # MSE validation
        y_plt1 = self.pipeline.MSE_val_dB_epoch[x_plt]
        ax3.plot(x_plt, y_plt1, color[1], label="KNet - Validation", linewidth=lineWidth)

        # MSE test
        y_plt2 = self.pipeline.MSE_test_dB_avg * torch.ones_like(x_plt)
        ax3.plot(x_plt, y_plt2, color[2], label="KNet - Test", linewidth=lineWidth)

        # MSE Kalman filter
        y_plt3 = MSE_KF * torch.ones_like(x_plt)
        ax3.plot(x_plt, y_plt3, color[3], label="KF - Test", linewidth=lineWidth)

        # Noise level
        y_plt4 = 10 * log10(self.pipeline.ssModel.r2) * torch.ones_like(x_plt)
        ax3.plot(x_plt, y_plt4, color[4], label="Noise variance", linewidth=lineWidth)
        
        if ylim3:
            ax3.set_ylim(ylim3)

        ax3.legend(fontsize=fontSize)
        ax3.set_xlabel('Epoch', fontsize=fontSize)
        ax3.set_ylabel('Loss Value [dB]', fontsize=fontSize)
        ax3.set_title("MSE Loss [dB] - per Epoch", fontsize=fontSize)

        if saveName:
            f.savefig(self.pipeline.folderName + saveName)


    def plot_lqr_and_mse(self, MSE_KF, figSize=(25, 25), fontSize=32, 
        lineWidth=2, title=None, saveName=None, ylim2=None, ylim3=None, color=['r-o','k-', 'b-', 'g-', 'y-']):

        # Figure
        f = plt.figure(figsize = figSize)
        f.suptitle(title, fontsize=fontSize+4)
        ax1 = f.add_subplot(2,1,1) # LQR
        ax2 = f.add_subplot(2,1,2) # MSE

        x_plt = torch.tensor(range(0, self.pipeline.N_Epochs))

        # Total training loss each epoch
        y_plt1 = self.pipeline.Total_loss_train_dB_epoch[x_plt]
        ax1.plot(x_plt, y_plt1, color[0], label=loop_legend[0], linewidth=lineWidth)

        # LQR validation loss each epoch
        y_plt2 = self.pipeline.LQR_val_dB_epoch[x_plt]
        ax1.plot(x_plt, y_plt2, color[1], label=loop_legend[1], linewidth=lineWidth)

        # LQR test loss
        y_plt3 = self.pipeline.LQR_test_dB_avg * torch.ones_like(x_plt)
        ax1.plot(x_plt, y_plt3, color[2], label=loop_legend[2], linewidth=lineWidth)

        # LQR loss when using a Kalman filter (i.e. LQG)
        y_plt4 = self.pipeline.LQG_cost * torch.ones_like(x_plt)
        ax1.plot(x_plt, y_plt4, color[3], label=loop_legend[3], linewidth=lineWidth)

        # LQR loss when knowing the state
        y_plt5 = self.pipeline.LQR_cost * torch.ones_like(x_plt)
        ax1.plot(x_plt, y_plt5, color[4], label=loop_legend[4], linewidth=lineWidth)

        if ylim2:
            ax1.set_ylim(ylim2)

        ax1.legend(fontsize=fontSize)
        # ax1.set_xlabel('Epoch', fontsize=fontSize)
        ax1.set_ylabel('Loss Value [dB]', fontsize=fontSize)
        ax1.set_title("LQG Loss [dB] - per Epoch", fontsize=fontSize)


        # MSE validation
        y_plt1 = self.pipeline.MSE_val_dB_epoch[x_plt]
        ax2.plot(x_plt, y_plt1, color[1], label="KNet - Validation", linewidth=lineWidth)

        # MSE test
        y_plt2 = self.pipeline.MSE_test_dB_avg * torch.ones_like(x_plt)
        ax2.plot(x_plt, y_plt2, color[2], label="KNet - Test", linewidth=lineWidth)

        # MSE Kalman filter
        y_plt3 = MSE_KF * torch.ones_like(x_plt)
        ax2.plot(x_plt, y_plt3, color[3], label="KF - Test", linewidth=lineWidth)

        # Noise level
        y_plt4 = 10 * log10(self.pipeline.ssModel.r2) * torch.ones_like(x_plt)
        ax2.plot(x_plt, y_plt4, color[4], label="Noise variance", linewidth=lineWidth)
        
        if ylim3:
            ax2.set_ylim(ylim3)

        ax2.legend(fontsize=fontSize)
        ax2.set_xlabel('Epoch', fontsize=fontSize)
        ax2.set_ylabel('Loss Value [dB]', fontsize=fontSize)
        ax2.set_title("MSE Loss [dB] - per Epoch", fontsize=fontSize)

        if saveName:
            f.savefig(self.pipeline.folderName + saveName)


    def plot_control(self, MSE_KF, figSize=(25, 25), fontSize=32, 
        lineWidth=2, title=None, saveName=None, ylim1=None, ylim2=None, ylim3=None, color=['r-o','k-', 'b-', 'g-', 'y-']):

        # Figure
        f = plt.figure(figsize = figSize)
        f.suptitle(title, fontsize=fontSize+4)
        ax1 = f.add_subplot(3,1,1) # Control Loss
        ax2 = f.add_subplot(3,1,2) # LQR
        ax3 = f.add_subplot(3,1,3) # MSE

        x_plt = torch.tensor(range(0, self.pipeline.N_Epochs))

        # training loss each epoch
        y_plt1 = self.pipeline.Control_train_dB_epoch[x_plt]
        ax1.plot(x_plt, y_plt1, color[0], label=loop_legend[0], linewidth=lineWidth)

        # validation loss each epoch
        y_plt2 = self.pipeline.Control_val_dB_epoch[x_plt]
        ax1.plot(x_plt, y_plt2, color[1], label=loop_legend[1], linewidth=lineWidth)

        # test loss TODO
        # y_plt3 = self.pipeline.Total_loss_test_dB_avg * torch.ones_like(x_plt)
        # ax1.plot(x_plt, y_plt3, color[2], label=loop_legend[2], linewidth=lineWidth)

        if ylim1:
            ax1.set_ylim(ylim1)

        ax1.legend(fontsize=fontSize)
        # ax1.set_xlabel('Epoch', fontsize=fontSize)
        ax1.set_ylabel('Loss Value [dB]', fontsize=fontSize)
        ax1.set_title(f"Control Loss [dB] - per Epoch", fontsize=fontSize)


        # LQR validation loss each epoch
        y_plt2 = self.pipeline.LQR_val_dB_epoch[x_plt]
        ax2.plot(x_plt, y_plt2, color[1], label=loop_legend[1], linewidth=lineWidth)

        # LQR test loss
        y_plt3 = self.pipeline.LQR_test_dB_avg * torch.ones_like(x_plt)
        ax2.plot(x_plt, y_plt3, color[2], label=loop_legend[2], linewidth=lineWidth)

        # LQR loss when using a Kalman filter (i.e. LQG)
        y_plt4 = self.pipeline.LQG_cost * torch.ones_like(x_plt)
        ax2.plot(x_plt, y_plt4, color[3], label=loop_legend[3], linewidth=lineWidth)

        # LQR loss when knowing the state
        y_plt5 = self.pipeline.LQR_cost * torch.ones_like(x_plt)
        ax2.plot(x_plt, y_plt5, color[4], label=loop_legend[4], linewidth=lineWidth)

        if ylim2:
            ax2.set_ylim(ylim2)

        ax2.legend(fontsize=fontSize)
        # ax2.set_xlabel('Epoch', fontsize=fontSize)
        ax2.set_ylabel('Loss Value [dB]', fontsize=fontSize)
        ax2.set_title("LQG Loss [dB] - per Epoch", fontsize=fontSize)


        # MSE validation
        y_plt1 = self.pipeline.MSE_val_dB_epoch[x_plt]
        ax3.plot(x_plt, y_plt1, color[1], label="KNet - Validation", linewidth=lineWidth)

        # MSE test
        y_plt2 = self.pipeline.MSE_test_dB_avg * torch.ones_like(x_plt)
        ax3.plot(x_plt, y_plt2, color[2], label="KNet - Test", linewidth=lineWidth)

        # MSE Kalman filter
        y_plt3 = MSE_KF * torch.ones_like(x_plt)
        ax3.plot(x_plt, y_plt3, color[3], label="KF - Test", linewidth=lineWidth)

        # Noise level
        y_plt4 = 10 * log10(self.pipeline.ssModel.r2) * torch.ones_like(x_plt)
        ax3.plot(x_plt, y_plt4, color[4], label="Noise variance", linewidth=lineWidth)
        
        if ylim3:
            ax3.set_ylim(ylim3)

        ax3.legend(fontsize=fontSize)
        ax3.set_xlabel('Epoch', fontsize=fontSize)
        ax3.set_ylabel('Loss Value [dB]', fontsize=fontSize)
        ax3.set_title("MSE Loss [dB] - per Epoch", fontsize=fontSize)

        if saveName:
            f.savefig(self.pipeline.folderName + saveName)
    

    def plot_loop_lqr_and_mse_model_mismatch(self, MSE_KF, figSize=(25, 25), fontSize=32, 
        lineWidth=2, title=None, saveName=None, ylim1=None, ylim2=None, ylim3=None, color=['r-o','k-', 'b-', 'g-', 'y-', 'g--', 'y--']):

        # Figure
        f = plt.figure(figsize = figSize)
        f.suptitle(title, fontsize=fontSize+4)
        ax1 = f.add_subplot(3,1,1) # LQR + MSE
        ax2 = f.add_subplot(3,1,2) # LQR
        ax3 = f.add_subplot(3,1,3) # MSE

        x_plt = torch.tensor(range(0, self.pipeline.N_Epochs))

        # Total training loss each epoch
        y_plt1 = self.pipeline.Total_loss_train_dB_epoch[x_plt]
        ax1.plot(x_plt, y_plt1, color[0], label=loop_legend[0], linewidth=lineWidth)

        # Total validation loss each epoch
        y_plt2 = self.pipeline.Total_loss_val_dB_epoch[x_plt]
        ax1.plot(x_plt, y_plt2, color[1], label=loop_legend[1], linewidth=lineWidth)

        # Total test loss
        y_plt3 = self.pipeline.Total_loss_test_dB_avg * torch.ones_like(x_plt)
        ax1.plot(x_plt, y_plt3, color[2], label=loop_legend[2], linewidth=lineWidth)

        if ylim1:
            ax1.set_ylim(ylim1)

        ax1.legend(fontsize=fontSize)
        # ax1.set_xlabel('Epoch', fontsize=fontSize)
        ax1.set_ylabel('Loss Value [dB]', fontsize=fontSize)
        ax1.set_title(f"Total Loss: {self.pipeline.alpha}*MSE + {self.pipeline.beta}*LQR [dB] - per Epoch", fontsize=fontSize)


        # LQR validation loss each epoch
        y_plt2 = self.pipeline.LQR_val_dB_epoch[x_plt]
        ax2.plot(x_plt, y_plt2, color[1], label=loop_legend[1], linewidth=lineWidth)

        # LQR test loss
        y_plt3 = self.pipeline.LQR_test_dB_avg * torch.ones_like(x_plt)
        ax2.plot(x_plt, y_plt3, color[2], label=loop_legend[2], linewidth=lineWidth)

        # LQR loss when using a Kalman filter (i.e. LQG)
        y_plt4 = self.pipeline.LQG_cost * torch.ones_like(x_plt)
        ax2.plot(x_plt, y_plt4, color[3], label=loop_legend[3], linewidth=lineWidth)

        # LQR loss when knowing the state
        y_plt5 = self.pipeline.LQR_cost * torch.ones_like(x_plt)
        ax2.plot(x_plt, y_plt5, color[4], label=loop_legend[4], linewidth=lineWidth)

        # LQG with true system
        y_plt6 = self.pipeline.LQG_cost_true_system * torch.ones_like(x_plt)
        ax2.plot(x_plt, y_plt6, color[5], label=loop_legend[5], linewidth=lineWidth)

        # LQR with true system
        y_plt7 = self.pipeline.LQR_cost_true_system * torch.ones_like(x_plt)
        ax2.plot(x_plt, y_plt7, color[6], label=loop_legend[6], linewidth=lineWidth)

        if ylim2:
            ax2.set_ylim(ylim2)

        ax2.legend(fontsize=fontSize)
        # ax2.set_xlabel('Epoch', fontsize=fontSize)
        ax2.set_ylabel('Loss Value [dB]', fontsize=fontSize)
        ax2.set_title("LQR Loss [dB] - per Epoch", fontsize=fontSize)


        # MSE validation
        y_plt1 = self.pipeline.MSE_val_dB_epoch[x_plt]
        ax3.plot(x_plt, y_plt1, color[1], label="KNet - Validation", linewidth=lineWidth)

        # MSE test
        y_plt2 = self.pipeline.MSE_test_dB_avg * torch.ones_like(x_plt)
        ax3.plot(x_plt, y_plt2, color[2], label="KNet - Test", linewidth=lineWidth)

        # MSE Kalman filter
        y_plt3 = MSE_KF * torch.ones_like(x_plt)
        ax3.plot(x_plt, y_plt3, color[3], label="KF - True system", linewidth=lineWidth)

        # Noise level
        y_plt4 = 10 * log10(self.pipeline.ssModel.r2) * torch.ones_like(x_plt)
        ax3.plot(x_plt, y_plt4, color[4], label="Noise variance", linewidth=lineWidth)
        
        if ylim3:
            ax3.set_ylim(ylim3)

        ax3.legend(fontsize=fontSize)
        ax3.set_xlabel('Epoch', fontsize=fontSize)
        ax3.set_ylabel('Loss Value [dB]', fontsize=fontSize)
        ax3.set_title("MSE Loss [dB] - per Epoch", fontsize=fontSize)

        if saveName:
            f.savefig(self.pipeline.folderName + saveName)


    def plot_lqr_and_mse_model_mismatch(self, MSE_KF, LQG_correct_kf, figSize=(25, 25), fontSize=32, 
        lineWidth=2, title=None, saveName=None, ylim2=None, ylim3=None, color=['r-o','k-', 'b-', 'g-', 'y-']):

        # Figure
        f = plt.figure(figsize = figSize)
        f.suptitle(title, fontsize=fontSize+4)
        ax1 = f.add_subplot(2,1,1) # LQR
        ax2 = f.add_subplot(2,1,2) # MSE

        x_plt = torch.tensor(range(0, self.pipeline.N_Epochs))

        # Total training loss each epoch
        y_plt1 = self.pipeline.Total_loss_train_dB_epoch[x_plt]
        ax1.plot(x_plt, y_plt1, color[0], label=loop_legend[0], linewidth=lineWidth)

        # LQR validation loss each epoch
        y_plt2 = self.pipeline.LQR_val_dB_epoch[x_plt]
        ax1.plot(x_plt, y_plt2, color[1], label=loop_legend[1], linewidth=lineWidth)

        # LQR test loss
        y_plt3 = self.pipeline.LQR_test_dB_avg * torch.ones_like(x_plt)
        ax1.plot(x_plt, y_plt3, color[2], label=loop_legend[2], linewidth=lineWidth)

        # LQR loss when using a Kalman filter (i.e. LQG)
        y_plt4 = self.pipeline.LQG_cost * torch.ones_like(x_plt)
        ax1.plot(x_plt, y_plt4, color[3], label=loop_legend[3], linewidth=lineWidth)

        # LQR loss when knowing the state
        y_plt5 = self.pipeline.LQR_cost * torch.ones_like(x_plt)
        ax1.plot(x_plt, y_plt5, color[4], label=loop_legend[4], linewidth=lineWidth)

        # LQG loss when using the correct model in the KF
        y_plt6 = LQG_correct_kf * torch.ones_like(x_plt)
        ax1.plot(x_plt, y_plt6, label='KF with correct model', linewidth=lineWidth)

        if ylim2:
            ax1.set_ylim(ylim2)

        ax1.legend(fontsize=fontSize)
        # ax1.set_xlabel('Epoch', fontsize=fontSize)
        ax1.set_ylabel('Loss Value [dB]', fontsize=fontSize)
        ax1.set_title("LQG Loss [dB] - per Epoch", fontsize=fontSize)


        # MSE validation
        y_plt1 = self.pipeline.MSE_val_dB_epoch[x_plt]
        ax2.plot(x_plt, y_plt1, color[1], label="KNet - Validation", linewidth=lineWidth)

        # MSE test
        y_plt2 = self.pipeline.MSE_test_dB_avg * torch.ones_like(x_plt)
        ax2.plot(x_plt, y_plt2, color[2], label="KNet - Test", linewidth=lineWidth)

        # MSE Kalman filter
        y_plt3 = MSE_KF * torch.ones_like(x_plt)
        ax2.plot(x_plt, y_plt3, color[3], label="KF - Test", linewidth=lineWidth)

        # Noise level
        y_plt4 = 10 * log10(self.pipeline.ssModel.r2) * torch.ones_like(x_plt)
        ax2.plot(x_plt, y_plt4, color[4], label="Noise variance", linewidth=lineWidth)
        
        if ylim3:
            ax2.set_ylim(ylim3)

        ax2.legend(fontsize=fontSize)
        ax2.set_xlabel('Epoch', fontsize=fontSize)
        ax2.set_ylabel('Loss Value [dB]', fontsize=fontSize)
        ax2.set_title("MSE Loss [dB] - per Epoch", fontsize=fontSize)

        if saveName:
            f.savefig(self.pipeline.folderName + saveName)


    def plot_epochs_simple_model_mismatch(self, MSE_KF_true_system=None, fontSize=32, lineWidth=2, title=None, saveName=None, ylim=None, color=['-ro', 'k-', 'b-', 'r-', 'g-']):
        if title is None:
            title = self.pipeline.modelName

        # Figure
        plt.figure(figsize = (25, 10))

        x_plt = torch.tensor(range(0, self.pipeline.N_Epochs))

        y_plt1 = self.pipeline.MSE_train_dB_epoch[x_plt]
        plt.plot(x_plt, y_plt1, color[0], label=Klegend[0], linewidth=lineWidth)

        y_plt2 = self.pipeline.MSE_val_dB_epoch[x_plt]
        plt.plot(x_plt, y_plt2, color[1], label=Klegend[1], linewidth=lineWidth)

        y_plt3 = self.pipeline.MSE_test_dB_avg_knet * torch.ones_like(x_plt)
        plt.plot(x_plt, y_plt3, color[2], label=Klegend[2], linewidth=lineWidth)

        y_plt4 = self.pipeline.MSE_test_dB_avg_kf * torch.ones_like(x_plt)
        plt.plot(x_plt, y_plt4, color[3], label='KF - correct model', linewidth=lineWidth)

        if MSE_KF_true_system is not None:
            y_plt5 = MSE_KF_true_system * torch.ones_like(x_plt)
            plt.plot(x_plt, y_plt5, color[4], label='KF - correct model', linewidth=lineWidth)
        
        if ylim:
            plt.ylim(ylim)

        plt.legend(fontsize=fontSize)
        plt.xlabel('Epoch', fontsize=fontSize)
        plt.ylabel('Loss Value [dB]', fontsize=fontSize)
        plt.title(title + "MSE Loss [dB] - per Epoch", fontsize=fontSize)
        
        if saveName:
            plt.savefig(self.pipeline.folderName + saveName)


    def plot_control_model_mismatch(self, MSE_KF, LQG_correct, figSize=(25, 25), fontSize=32, 
        lineWidth=2, title=None, saveName=None, ylim1=None, ylim2=None, ylim3=None, color=['r-o','k-', 'b-', 'g-', 'y-']):

        # Figure
        f = plt.figure(figsize = figSize)
        f.suptitle(title, fontsize=fontSize+4)
        ax1 = f.add_subplot(3,1,1) # Control Loss
        ax2 = f.add_subplot(3,1,2) # LQR
        ax3 = f.add_subplot(3,1,3) # MSE

        x_plt = torch.tensor(range(0, self.pipeline.N_Epochs))

        # training loss each epoch
        y_plt1 = self.pipeline.Control_train_dB_epoch[x_plt]
        ax1.plot(x_plt, y_plt1, color[0], label=loop_legend[0], linewidth=lineWidth)

        # validation loss each epoch
        y_plt2 = self.pipeline.Control_val_dB_epoch[x_plt]
        ax1.plot(x_plt, y_plt2, color[1], label=loop_legend[1], linewidth=lineWidth)

        # test loss TODO
        # y_plt3 = self.pipeline.Total_loss_test_dB_avg * torch.ones_like(x_plt)
        # ax1.plot(x_plt, y_plt3, color[2], label=loop_legend[2], linewidth=lineWidth)

        if ylim1:
            ax1.set_ylim(ylim1)

        ax1.legend(fontsize=fontSize)
        # ax1.set_xlabel('Epoch', fontsize=fontSize)
        ax1.set_ylabel('Loss Value [dB]', fontsize=fontSize)
        ax1.set_title(f"Control Loss [dB] - per Epoch", fontsize=fontSize)


        # LQR validation loss each epoch
        y_plt2 = self.pipeline.LQR_val_dB_epoch[x_plt]
        ax2.plot(x_plt, y_plt2, color[1], label=loop_legend[1], linewidth=lineWidth)

        # LQR test loss
        y_plt3 = self.pipeline.LQR_test_dB_avg * torch.ones_like(x_plt)
        ax2.plot(x_plt, y_plt3, color[2], label=loop_legend[2], linewidth=lineWidth)

        # LQR loss when using a Kalman filter (i.e. LQG)
        y_plt4 = self.pipeline.LQG_cost * torch.ones_like(x_plt)
        ax2.plot(x_plt, y_plt4, 'r-', label='LQG - wrong model', linewidth=lineWidth)

        # LQR loss when knowing the state
        y_plt5 = LQG_correct * torch.ones_like(x_plt)
        ax2.plot(x_plt, y_plt5, 'g-', label='LQG - correct model', linewidth=lineWidth)

        if ylim2:
            ax2.set_ylim(ylim2)

        ax2.legend(fontsize=fontSize)
        # ax2.set_xlabel('Epoch', fontsize=fontSize)
        ax2.set_ylabel('Loss Value [dB]', fontsize=fontSize)
        ax2.set_title("LQG Loss [dB] - per Epoch", fontsize=fontSize)


        # MSE validation
        y_plt1 = self.pipeline.MSE_val_dB_epoch[x_plt]
        ax3.plot(x_plt, y_plt1, color[1], label="KNet - Validation", linewidth=lineWidth)

        # MSE test
        y_plt2 = self.pipeline.MSE_test_dB_avg * torch.ones_like(x_plt)
        ax3.plot(x_plt, y_plt2, color[2], label="KNet - Test", linewidth=lineWidth)

        # MSE Kalman filter
        y_plt3 = MSE_KF * torch.ones_like(x_plt)
        ax3.plot(x_plt, y_plt3, color[3], label="KF - correct model", linewidth=lineWidth)

        # Noise level
        y_plt4 = 10 * log10(self.pipeline.ssModel.r2) * torch.ones_like(x_plt)
        ax3.plot(x_plt, y_plt4, color[4], label="Noise variance", linewidth=lineWidth)
        
        if ylim3:
            ax3.set_ylim(ylim3)

        ax3.legend(fontsize=fontSize)
        ax3.set_xlabel('Epoch', fontsize=fontSize)
        ax3.set_ylabel('Loss Value [dB]', fontsize=fontSize)
        ax3.set_title("MSE Loss [dB] - per Epoch", fontsize=fontSize)

        if saveName:
            f.savefig(self.pipeline.folderName + saveName)