import torch
import torch.nn as nn

#
# class MahalanobisLoss(nn.Module):
#     def __init__(self):
#         super(MahalanobisLoss, self).__init__()
#
#     def forward(self, x, y, covariance_matrix):
#         # Вычисляем разность между векторами x и y
#         diff = x - y
#
#         # Вычисляем расстояние Махаланобиса
#         distance = torch.sqrt(torch.matmul(torch.matmul(diff, covariance_matrix), diff.t()))
#
#         return distance


class MSEWithScore(nn.Module):
    def __init__(self, l: float = 1.0):
        super(MSEWithScore, self).__init__()
        self._lambda = l

    def forward(self, x, y, score_x, score_y):
        # Вычисляем разность между векторами x и y
        mse = torch.nn.MSELoss().forward(x, y)
        mse = mse + torch.nn.MSELoss().forward(score_x, score_y)
        return mse
