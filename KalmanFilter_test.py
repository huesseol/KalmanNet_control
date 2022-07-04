import torch
import torch.nn as nn
import time
from Linear_KF import KalmanFilter
# from Extended_data import N_T

def KFTest(SysModel, test_input, test_target):
    # Unpack test inputs 
    test_input_y, test_input_u, X0_test = test_input
    

    N_T = test_input_y.size()[0]

    # LOSS
    loss_fn = nn.MSELoss(reduction='mean')

    # MSE [Linear]
    MSE_KF_total_arr = torch.empty(N_T) # position and velocity
    MSE_KF_position_arr = torch.empty(N_T) # only position

    start = time.time()
    KF = KalmanFilter(SysModel)
        
    for j in range(0, N_T):

        KF.InitSequence(X0_test[j], SysModel.m2x_0)
        KF.GenerateSequence(test_input_y[j, :, :], test_input_u[j, :, :])

        MSE_KF_total_arr[j] = loss_fn(KF.x, test_target[j, :, :]).item() 
        MSE_KF_position_arr[j] = loss_fn(KF.x[0, :], test_target[j, 0, :]).item() 
        
    end = time.time()
    t = end - start

    MSE_KF_total_avg = torch.mean(MSE_KF_total_arr)
    MSE_KF_total_dB_avg = 10 * torch.log10(MSE_KF_total_avg)

    MSE_KF_position_avg = torch.mean(MSE_KF_position_arr)
    MSE_KF_position_dB_avg = 10 * torch.log10(MSE_KF_position_avg)

    # Standard deviation
    MSE_KF_total_std = torch.std(MSE_KF_total_arr, unbiased=True)
    MSE_KF_total_dB_std = 10 * torch.log10(MSE_KF_total_avg + MSE_KF_total_std) - MSE_KF_total_dB_avg

    MSE_KF_position_std = torch.std(MSE_KF_position_arr, unbiased=True)
    MSE_KF_position_dB_std = 10 * torch.log10(MSE_KF_position_avg + MSE_KF_position_std) - MSE_KF_position_dB_avg

    print(f"Kalman Filter - Total MSE LOSS: {MSE_KF_total_dB_avg: .5f} [dB], STD: {MSE_KF_total_dB_std: .5f} [dB]")
    # print(f" MSE STD: {MSE_KF_total_dB_std} [dB]")
    print(f"Kalman Filter - Position MSE LOSS: {MSE_KF_position_dB_avg: .5f} [dB], STD: {MSE_KF_position_dB_std: .5f} [dB]")
    # print(f"EKF - MSE STD: {MSE_KF_position_dB_std} [dB]")
    # Print Run Time
    print("Inference Time:", t)

    return [MSE_KF_total_arr, MSE_KF_total_dB_avg, MSE_KF_total_dB_std, MSE_KF_position_arr, MSE_KF_position_dB_avg, MSE_KF_position_dB_std]



