import torch
import torch.nn.functional as F
import context
from models import sah_functions as SAHF
import unittest
import time

# reference solution full reduce to FC layer
#
def conv_binary_inner_SAH_ref1(x_in, q_in, weight, bias, stride=1, padding=0):
    """
    An inner stochastic binary layer
    :param x: - input state Tensor [B C_in H W], +/-1
    :param q: - input state linearized flip probabilities (assume very small) [B C_in H W]
    :param weight: - layer weight [C_out C_in K K]
    :param bias: - layer bias [C_out]
    :param x_out: - output state sample (for debugging)
    :return:
    x_out - output binary state [B C_out H1 W1] +/-1
    q_out - its linearized flip probabilities [B C_out H1 W1]
    #
    # method: using unfold / fold reduce to fully connected stichastic binary layer:
    # Convolution is equivalent with Unfold + Matrix Multiplication + Fold (or view to output shape)
    # https: // pytorch.org / docs / stable / nn.html?highlight = unfold  # torch.nn.Unfold
    """
    # reference output shape
    a0 = F.conv2d(x_in, weight, bias, stride=stride, padding=padding)
    #
    B = x_in.size(0)
    C_out = weight.size(0)
    CK = weight.size(1) * weight.size(2) * weight.size(3)  # CK = C_in*K1*K2
    # unfold
    x_in_unf = F.unfold(x_in, weight.size()[2:], padding=padding, stride=stride)  # [B CK L]
    q_in_unf = F.unfold(q_in, weight.size()[2:], padding=padding, stride=stride)  # [B CK L]
    L = x_in_unf.size(2)
    # reshape to vectors
    x_in_vec = x_in_unf.permute([0, 2, 1]).contiguous().view([-1, CK])  # [B*L CK]
    q_in_vec = q_in_unf.permute([0, 2, 1]).contiguous().view([-1, CK])  # [B*L CK]
    Weight = weight.view(weight.size(0), -1)  # [C_out, CK]
    Bias = bias.view([-1])  # [C_out]
    # use fully connected implementation
    x_out_vec, q_out_vec = SAHF.linear_binary_inner_SAH(x_in_vec, q_in_vec, Weight, Bias)  # [B*L C_out]
    # reshape back
    x_out_fold = x_out_vec.view([B, L, C_out]).permute([0, 2, 1]).contiguous()  # [B C_out L]
    q_out_fold = q_out_vec.view([B, L, C_out]).permute([0, 2, 1]).contiguous()  # [B C_out L]
    # output view
    x_out = x_out_fold.view(a0.size())
    q_out = q_out_fold.view(a0.size())
    return x_out, q_out


class SAHConvTestSuite(unittest.TestCase):
    def test_conv_binary_grad(self):
        seed = 2
        torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)
        
        B = 16  # batch size
        C_in = 32
        C_out = 32
        W = 32
        H = 32

        for C_out in [32, 64, 128]:
            for W in [32, 30, 16, 14, 8, 6]:
                H = W
                for K in [1,3]:
                    for stride in [1,2]:
                        for padding in [0,1,2]:
                            print("Shape: (B={},C_in={},C_out={},W={},K={}, stride={}, padding={})".format(B, C_in, C_out, W, K, stride, padding))
                            x_in = torch.rand([B, C_in, H, W]).cuda().bernoulli() * 2 - 1
                            weight = torch.rand([C_out, C_in, K, K]).cuda()
                            bias = torch.rand([C_out]).cuda()
                            q_in = torch.zeros_like(x_in, requires_grad=True)
                            
                            # reference forward-backward
                            # t = time.time()
                            x_out1, q_out1 = conv_binary_inner_SAH_ref1(x_in, q_in, weight, bias, stride=stride, padding=padding)
                            g = torch.empty_like(q_out1).normal_()
                            g_in1 = torch.autograd.grad(outputs=q_out1, inputs=q_in, grad_outputs=[g, ], only_inputs=True)[0]
                            torch.cuda.synchronize()
                            # t = time.time() - t
                            # print("Conv reference method (unfold) time: {}".format(t))
                            
                            # optimized forward-backward, use same output sample
                            x_out, q_out = SAHF.conv_binary_inner_SAH(x_in, q_in, weight, bias, x_out=x_out1, stride=stride, padding=padding)
                            g_in = torch.autograd.grad(outputs=q_out, inputs=q_in, grad_outputs=[g, ], only_inputs=True)[0]
                            #print(g_in.size())
                            #print((g_in - g_in1).abs().max())
                            #print((g_in - g_in1).abs().mean())
                            self.assertTrue(g_in.size() == g_in1.size())
                            mnorm = (g_in - g_in1).abs().max().item()
                            eps = g_in1.max().abs().item()*1e-4
                            print("max  diff:", (g_in - g_in1).abs().max().item())
                            print("mean diff:", (g_in - g_in1).abs().mean().item())
                            self.assertTrue(mnorm < eps, f"max norm = {mnorm} > {eps}")
                            self.assertTrue((g_in - g_in1).abs().mean() < eps)
    
            # # test performance too
            # t = time.time()
            # for trial in range(10):
            #     x_out, q_out = SAHF.conv_binary_inner_SAH(x_in, q_in, weight, bias, x_out=x_out1, stride=stride, padding=padding)
            #     g_in = torch.autograd.grad(outputs=q_out, inputs=q_in, grad_outputs=[g, ], only_inputs=True)[0]
            # torch.cuda.synchronize()
            # t = time.time() - t
            # print("Conv with cuda extension time: {}".format(t))
            # #


if __name__ == "__main__":
    # tests
    # test_conv_binary_grad()
    unittest.main()
