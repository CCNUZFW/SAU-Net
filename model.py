import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init



# classic residual block
class RB(nn.Module):
    def __init__(self, nf, bias, kz=3):
        super(RB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf, kz, padding=kz // 2, bias=bias), nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kz, padding=kz // 2, bias=bias),
        )

    def forward(self, x):
        return x + self.body(x)


class Inception(nn.Module):
    def __init__(self, in_channle):
        super(Inception, self).__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(in_channle, in_channle, kernel_size=1))
        self.branch2 = nn.Sequential(nn.Conv2d(in_channle, in_channle, kernel_size=1),
                                     nn.Conv2d(in_channle, in_channle, kernel_size=5, padding=2))
        self.branch3 = nn.Sequential(nn.Conv2d(in_channle, in_channle, kernel_size=1),
                                     nn.Conv2d(in_channle, in_channle, kernel_size=3, padding=1))

        self.branch_pool = nn.Conv2d(in_channle, in_channle, kernel_size=1)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        # branch4 = self.branch4(x)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        return torch.cat((branch1, branch2, branch3, branch_pool), dim=1)



class Shrinkage(nn.Module):
    def __init__(self, channel):
        super(Shrinkage, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),  # 可能应该是nn.Linear(channel, channel)
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_raw = x
        x = torch.abs(x)
        x_abs = x
        x = self.gap(x)
        x = torch.flatten(x, 1)
        # average = torch.mean(x, dim=1, keepdim=True)
        average = x
        x = self.fc(x)
        x = torch.mul(average, x)
        x = x.unsqueeze(2).unsqueeze(2)
        # 软阈值化
        sub = x_abs - x
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x_raw), n_sub)
        return x

class Phase(nn.Module):
    def __init__(self, img_nf, B):
        super(Phase, self).__init__()
        bias, nf, nb, onf = True, 8, 3, 3  # config of E
        self.soft = Shrinkage(32)

        self.rho = nn.Parameter(torch.Tensor([0.5]))

        self.S = nn.Sequential(nn.Conv2d(1, 8, kernel_size=1, padding=0, stride=1, bias=True),
                               nn.ReLU(inplace=True),
                               Inception(in_channle=8),
                               nn.ReLU(inplace=True),
                               Inception(in_channle=32),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=True),
                               nn.Conv2d(32, 8, kernel_size=1, padding=0, stride=1, bias=True),
                               nn.Conv2d(8, 3, kernel_size=1, padding=0, stride=1, bias=True),
                               )

        self.B = B  # default: 32
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))
        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 4, 3, 3)))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))

        self.conv1_G = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_G = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv3_G = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))

    def forward(self, x, saliency_map, PhiT_Phi, PhiT_y, mode, shape_info):
        b, l, h, w = shape_info

        S = self.S(saliency_map)
        # block gradient descent
        x = x - self.rho * (PhiT_Phi.matmul(x) - PhiT_y)  # 64 1024 1
        x = x.reshape(b, l, -1).permute(0, 2, 1)  # 4,1024,16

        x = F.fold(x, output_size=(h, w), kernel_size=self.B, stride=self.B)  # (4,1,128,128)
        x_input = x

        # proximal
        x_S = torch.cat([x_input, S], dim=1)
        x_D = F.conv2d(x_S, self.conv_D, padding=1)

        x = F.conv2d(x_D, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)
        # soft
        x=self.soft(x_forward)

        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)

        x = F.conv2d(F.relu(x_backward), self.conv1_G, padding=1)
        x = F.conv2d(F.relu(x), self.conv2_G, padding=1)
        x_G = F.conv2d(x, self.conv3_G, padding=1)

        x_pred = x_input + x_G
        # 约束
        x = F.conv2d(x_forward, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_D_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_D_est - x_D

        return [x_pred, symloss]



class D(nn.Module):
    def __init__(self, img_nf):
        super(D, self).__init__()
        bias, block, nb, mid_nf = False, RB, 3, 32
        conv = lambda in_nf, out_nf: nn.Conv2d(in_nf, out_nf, 3, padding=1, bias=bias)
        self.body = nn.Sequential(conv(img_nf, mid_nf), *[block(mid_nf, bias) for _ in range(nb)], conv(mid_nf, 1))

    def forward(self, x):
        return self.body(x).reshape(*x.shape[:2], -1).softmax(dim=2).reshape_as(x)


# error correction of BRA
def batch_correct(Q, target_sum, N):
    b, l = Q.shape
    i, max_desc_step = 0, 10
    while True:
        i += 1
        Q = torch.clamp(Q, 0, N).round()
        d = Q.sum(dim=1) - target_sum  # batch delta
        if float(d.abs().sum()) == 0.0:
            break
        elif i < max_desc_step:  # 1: uniform descent
            Q = Q - (d / l).reshape(-1, 1).expand_as(Q)
        else:  # 2: random allocation
            for j in range(b):
                D = np.random.multinomial(int(d[j].abs().ceil()), [1.0 / l] * l, size=1)
                Q[j] -= int(d[j].sign()) * torch.Tensor(D).squeeze(0).to(Q.device)
    return Q


class SAUNet(nn.Module):
    def __init__(self, phase_num, B, img_nf, Phi_init):
        super(SAUNet, self).__init__()
        self.phase_num = phase_num
        self.phase_num_minus_1 = phase_num - 1
        self.B = B
        self.N = B * B
        self.Phi = nn.Parameter(Phi_init.reshape(self.N, self.N))
        self.RS = nn.ModuleList([Phase(img_nf, B) for _ in range(phase_num)])
        self.D = D(img_nf)
        self.index_mask = torch.arange(1, self.N + 1)
        self.epsilon = 1e-6
        self.relu = nn.ReLU()

    def forward(self, x, salieny_map, q, modes):
        b, c, h, w = x.shape

        # saliency detection

        S1 = torch.ones((b, c, h, w)).cuda()
        S1 = S1.reshape(*S1.shape[:2], -1).softmax(dim=2).reshape_as(S1)
        S2 = salieny_map  # saliency map
        S2 = S2.reshape(*S2.shape[:2], -1).softmax(dim=2).reshape_as(S2)
        S = ((0.5 * S1) + (0.5 * S2))

        # CS ratio allocation (with BRA method)
        x_unfold = F.unfold(x, kernel_size=self.B, stride=self.B).permute(0, 2, 1)  # shape: (b, l, img_nf * B * B)
        l = x_unfold.shape[1]  # block number of an image patch

        Q = q * l * S  # measurement size map
        mask_1 = self.relu((Q - (q / self.N))).sign()
        Q_unfold = F.unfold(Q, kernel_size=self.B, stride=self.B).permute(0, 2, 1).sum(
            dim=2)  # sumpooling, shape: (b, l)
        Q_unfold = batch_correct(Q_unfold, q * l, self.N) + self.epsilon * (
                Q_unfold - Q_unfold.detach())  # error correction

        # divide image patch into blocks
        block_stack = x_unfold.reshape(-1, c * self.N, 1)  # shape: (b * l, img_nf * B * B, 1)
        block_volume = block_stack.shape[1]  # img_nf * B * B

        # generate sampling matrices
        L = block_stack.shape[0]  # total block number of batch
        Phi_stack = self.Phi.unsqueeze(0).repeat(L, 1, 1)

        index_mask = self.index_mask.unsqueeze(0).repeat(L, 1).to(Phi_stack.device)
        q_stack = Q_unfold.reshape(-1, 1).repeat(1, Phi_stack.shape[1])

        cur_mask = F.relu(q_stack - index_mask + 1.0).sign() + self.epsilon * (q_stack - q_stack.detach())
        Phi_stack = Phi_stack * cur_mask.unsqueeze(2)


        # sample and initialize simultaneously
        PhiT_Phi = Phi_stack.permute(0, 2, 1).matmul(Phi_stack)
        PhiT_y = PhiT_Phi.matmul(block_stack)

        x = PhiT_y

        # get expanded CS ratio map R'
        cs_ratio_map = (Q_unfold.detach() / self.N).unsqueeze(2).repeat(1, 1, block_volume).permute(0, 2, 1)
        cs_ratio_map = F.fold(cs_ratio_map, output_size=(h, w), kernel_size=self.B, stride=self.B)
        mask_2 = self.relu(cs_ratio_map - (q / self.N)).sign()

        # recover step-by-step
        shape_info = [b, l, h, w]
        layers_sym = []
        for i in range(self.phase_num):
            [x, layer_sym] = self.RS[i](x, S, PhiT_Phi, PhiT_y, modes[i], shape_info)
            layers_sym.append(layer_sym)
            if i < self.phase_num_minus_1:
                x = F.unfold(x, kernel_size=self.B, stride=self.B).permute(0, 2, 1)
                x = x.reshape(L, -1, 1)
        x_final = x

        return [x_final, layers_sym]
