import torch
import control
import numpy as np

def lqr_finite(horizon, F, G, QN, Qx, Qu):
    '''
    Computes the LQR gains for the finite horizon problem with the
    given matrices.

    Parameters
    ----------
    horizon : int
    F : tensor of shape (m,m)
    G : tensor of shape (m,p)
    QN : tensor of shape (m,m)
    Qx : tensor of shape (m,m)
    Qu : tensor of shape (p,p)

    Returns
    -------
    L_list : list of tensors of shape (p,m) where L_list[0] is the lqr gain to 
    be used for input u[0]
    S_list : list of tensors of shape (m,m) where S_list[0] is the last element 
    of the backward recursion for S 
    '''
    FT = torch.transpose(F,1,0)
    GT = torch.transpose(G,1,0)
    S = QN
    L_list = []
    S_list = [S]
    for t in range(horizon):
        S1 = FT.matmul(S).matmul(F) + Qx
        S2 = FT.matmul(S).matmul(G) 
        S3 = GT.matmul(S).matmul(G) + Qu
        S3 = torch.pinverse(S3)

        S = S1 - S2.matmul(S3).matmul(torch.transpose(S2,1,0))
        S = (S + S.transpose(0,1)) / 2.0 # force symmetry
        S_list.append(S)
    S_list.reverse()

    for t in range(horizon):
        L1 = GT.matmul(S_list[t]).matmul(G) + Qu
        L1 = torch.pinverse(L1)
        L2 = GT.matmul(S_list[t]).matmul(F)
        L_list.append(L1.matmul(L2))
    
    return L_list , S_list


def lqr_infinite(F,G,Qx,Qu):
    L, S, _ = control.dlqr(F,G,Qx,Qu)
    return torch.FloatTensor(L), torch.FloatTensor(S)


def kalman_finite(T, F, H, Q, R, m2x_0):
    '''
    Computes the Kalman gains for the finite horizon problem with the
    given matrices.

    Parameters
    ----------
    horizon : int
    F : tensor of shape (m,m)
    G : tensor of shape (m,p)
    Q : tensor of shape (m,m)
        Process noise covariance
    R : tensor of shape (m,m)
        Observation noise covariance
    m2x_0 : tensor of shape (m,m)
        Covariance of initial state. If we know it, then set it to the zero matrix.

    Returns
    -------
    K_list : list of tensors of shape (n,m) where K_list[t] is the Kalman gain to 
    be used for y[t]
    P_list : list of tensors of shape (m,m) where P_list[t] is the error covariance 
    matrix of the estimate at time t
    '''
    # Computaten based on prediction and correction formulation
    FT = torch.transpose(F, 0, 1)
    HT = torch.transpose(H, 0, 1)
       
    # Initial Kalman gain
    K0 = torch.matmul(m2x_0, HT)
    K1 = H.matmul(m2x_0).matmul(HT) + R
    K2 = torch.inverse(K1)
    K = torch.matmul(K0, K2)

    # Initial error covariance matrix
    posterior = m2x_0 - K.matmul(K1).matmul(torch.transpose(K, 0, 1))

    P_list = [posterior]
    K_list = [K]
    for t in range(T):
        prior = F.matmul(posterior).matmul(FT) + Q
        S = H.matmul(prior).matmul(HT) + R
        Kt = prior.matmul(HT).matmul(torch.inverse(S))
        K_list.append(Kt)
        posterior = prior - Kt.matmul(S).matmul(torch.transpose(Kt,0,1))
        P_list.append(posterior)

    return K_list, P_list


# I don't know why this recursion doesn't work...
#
# def kalman_recursion_finite(T, F, H, Q, R, m2x_0):
#     FT = torch.transpose(F,1,0)
#     HT = torch.transpose(H,1,0)
#     P = m2x_0
#     P_list = [P]
#     for t in range(T):
#         P1 = F.matmul(P).matmul(FT) + Q
#         P2 = F.matmul(P).matmul(HT) 
#         P3 = H.matmul(P).matmul(HT) + R
#         P3 = torch.inverse(P3)

#         P = P1 - P2.matmul(P3).matmul(torch.transpose(P2,1,0))
#         P_list.append(P)

#     K = []
#     for t in range(T+1):
#         K1 = P_list[t].matmul(HT)
#         K2 = H.matmul(P_list[t]).matmul(HT) + R
#         K2 = torch.pinverse(K2)

#         K.append(K1.matmul(K2))

#     return K, P_list




