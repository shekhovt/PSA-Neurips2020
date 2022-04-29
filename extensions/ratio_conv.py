import context
import torch
import torch.nn.functional as F
import numpy as np
import context

# from extensions import ratio_conv2d
# import extensions.ratio_conv2d as ratio_conv2d
import extensions.ratio_conv2d as ratio_conv2d

# from torch.utils.cpp_extension import load

# print("loading extension ratio_conv2d")

# import os
#
# d = os.path.dirname(os.path.abspath(__file__))
# os.chdir(d)
# # path.append(dir(path[0]))
#
# ratio_conv2d = load(name="ratio_conv2d", sources=["./ratio_conv2d.cpp", "./ratio_conv2d_kernel.cu"], verbose=True)

def ratio_conv2d_backward(g, a, w_p, w_m, x_in, padding=0, stride=1, impl_v=1):
    """
    ratio_conv2d_backward(g, a, w_p, w_m, x_in, impl_v = 0)
    Computes the gradient of the modified (ratio) convolution in q_in
    The forward of the ratio convolution is
    
     q_out_{c,jj} = sum_{r,@ii} 1/(1 + a_{c,jj} * w_{c,r,ii-jj}) * q_in{r,ii}
     
     where @ii denotes the convolution in a 2d index ii
    
     g: [B C_out H1 W1]
     a: [B C_out H1 W1]
     wt: [C_in C_out K K] -- transposed weight
     x_in [B C_in H W] - binary input
     
    """
    # adjust for padding and stride
    # padded input size
    assert w_p.size(2) == w_p.size(3), 'need a square kernel'
    K = w_p.size(3)
    Sz = np.array(x_in.size())
    sz_p = Sz[2:] + padding * 2  # symmetric padding on all sides
    # expected output size
    sz1 = sz_p - (K // 2) * 2
    # expected output size after stride
    sz1_s = (sz1 + stride - 1) // stride  # divup
    g_sz = np.array(g.size()[2:])
    assert (g_sz == sz1_s).all(), 'output shape {} different from expected {}'.format(g_sz, sz1_s)
    # padd the input
    if padding > 0:
        Sz_p = x_in.size()[0:2] + tuple(sz_p)
        temp = x_in.new_zeros(Sz_p)
        temp[:, :, padding:-padding, padding:-padding] = x_in
        x_in = temp
        
    # output stride and alignment
    # align = 1  # messes up with expected oputput shapes -> can enable it and put a view ontop
    # sz1_sa = ((sz1 + align - 1) // align) * align
    sz1_sa = sz1
    Sz1_sa = list(g.size()[0:2]) + list(sz1_sa)
    if not (sz1_sa == g_sz).all(): # need either stride or alignment or both
        _g = g.new_zeros(Sz1_sa)
        _g[:,:,:sz1[0]:stride,:sz1[1]:stride] = g
        g = _g
        _a = g.new_zeros(Sz1_sa)
        _a[:, :, :sz1[1]:stride, :sz1[1]:stride] = a
        a = _a
    #
    r = ratio_conv2d.backward(g, a, w_p, w_m, x_in, impl_v)
    #
    # crop back the padded stuff
    if padding > 0:
        r[0] = r[0][:, :, padding:-padding, padding:-padding]
    return r


# tests
def ratio_conv2d_forward_ref(a, w, x_in, q_in, stride, padding):
    """
    :param a:  [B C_out H1 W1] -- preactivations
    :param w: [C_out C_in K K] -- transposed weight
    :param x_in: [B C_in H W]
    :param q_in: [B C_in H W]
    :return:
    :param q_out: [B C_out H1 W1]
    """
    # method: compute as many convolutions as there are output locations
    q_out = torch.zeros_like(a)
    for b in range(a.size(0)):
        for j1 in range(a.size(2)):
            for j2 in range(a.size(3)):
                # modify weights
                w1 = 1 / (1 + torch.einsum("c, crij -> crij", a[b, :, j1, j2], w))
                c = F.conv2d(q_in, w1, stride=stride, padding=padding)
                q_out[b, :, j1, j2] = c[b, :, j1, j2]
    
    return q_out


def ratio_conv2d_backward_ref(g, a, w, x_in, stride, padding):
    # method: automatic differentiation of forward_ref
    q_in = torch.zeros_like(x_in, requires_grad=True)
    q_out = ratio_conv2d_forward_ref(a, w, x_in, q_in, stride=stride, padding=padding)
    g_in = torch.autograd.grad(outputs=q_out, inputs=q_in, grad_outputs=[g, ], only_inputs=True)
    return g_in


def ratio_conv2d_backward_ref0(g, a, w, x_in):
    # applies to 1x1 convolution
    # method: automatic differentiation of a simplified forward
    q_in = torch.zeros_like(x_in, requires_grad=True)
    q_out = torch.zeros_like(a)
    for b in range(a.size(0)):
        M = 1 / (1 + torch.einsum("cij, cr -> crij", a[b, :, :, :], w[:, :, 0, 0]))
        q_out[b] = torch.einsum("rij, crij-> cij", q_in[b], M)
    
    g_in = torch.autograd.grad(outputs=q_out, inputs=q_in, grad_outputs=[g, ], only_inputs=True)
    return g_in


def ratio_conv2d_backward_ref00(g, a, w, x_in):
    # very special case
    g_in = g / 2
    return g_in



def test_correctness():
    print("TEST CORRECTNESS")
    B = 1
    for C_in in [1, 2, 3, 7]:
        for C_out in [1, 2, 3, 7, 39]:
            for W1 in [1, 2, 3, 4, 13, 39]:
                for K in [1, 3, 5]:
                    for pad in [True, False]:
                        for stride in [1, 2, 3]:
                            if pad:
                                padding = K // 2
                            else:
                                padding = 0
                            H1 = W1
                            W = W1 + 2 * (K // 2)  # so that there is at least one valid output, H1, W1 not used directly
                            H = H1 + 2 * (K // 2)
                            print("Shape: (B={},C_in={},C_out={},W={},K={}, stride={}, padding={})".format(B, C_in, C_out, W, K, stride, padding))
                            # a = torch.rand([B, C_out, H1, W1]).cuda()
                            # g = torch.rand([B, C_out, H1, W1]).cuda()
                            wp = torch.rand([C_out, C_in, K, K]).cuda()
                            wm = torch.rand([C_out, C_in, K, K]).cuda()
                            #
                            x_in = torch.ones([B, C_in, H, W]).cuda()  # this is unchecked
                            #
                            a = F.conv2d(x_in, wp, stride=stride, padding=padding)
                            a.random_()
                            g = torch.empty_like(a)
                            g.random_()
                            # w.fill_(1.0)
                            # a.fill_(1.0)
                            wp_t = wp.permute([1, 0, 2, 3]).flip(dims=[2, 3])  # "transposed" kernel [C_in C_out, K, K]
                            wm_t = wm.permute([1, 0, 2, 3]).flip(dims=[2, 3])  # "transposed" kernel [C_in C_out, K, K]
                            # x_in = torch.rand([B, C_in, H, W]).cuda()

                        
                            w = wp

                            print("Ref solution:")
                            y1 = ratio_conv2d_backward_ref(g, a, wp, x_in, stride=stride, padding=padding)[0]
                            # print(y1)
                            # if w.size(2) == 1 and w.size(3) == 1:
                            #     print("Ref0 solution:")
                            #     y0 = ratio_conv2d_backward_ref0(g, a, w, x_in)[0]
                            #     print((y1 - y0).abs().max().item())
                        
                            print("Kernel0  solution:")
                            y2 = ratio_conv2d_backward(g, a, wp_t, wm_t, x_in, stride=stride, padding=padding, impl_v = 0)[0]
                            dy2y1 = (y2 - y1).abs().max().item()
                            print(dy2y1)
                            # print(y2)
                        
                            print("Kernel1  solution:")
                            y3 = ratio_conv2d_backward(g, a, wp_t, wm_t, x_in, stride=stride, padding=padding, impl_v = 1)[0]
                            # print(y3.size())
                            dy3y1 = (y3 - y1).abs().max().item()
                            dy3y2 = (y3 - y2).abs().max().item()
                            eps = y1.abs().max().item()
                            print(dy3y1)
                            print(dy3y2)
                            test = dy3y1 < dy2y1 + eps * 1e-6
                            if not test:
                                print("Ref y=")
                                print(y1[0])
                                print("Kernel1 y=")
                                print(y3[0])
                            assert (test)
                            # assert (dy3y2 < dy2y1 * 2)
                            # print(y3)
                            # print(y1)
    print("TET PASSED")


def test_performance():
    print("TEST PERFORMANCE")
    # performance test
    B = 128
    C_in = 32
    C_out = 32
    W = 34
    H = 34
    K = 3
    W1 = W - 2 * (K // 2)
    H1 = H - 2 * (K // 2)
    #
    print("Shape: (B={},C_in={},C_out={},W={},K={})".format(B, C_in, C_out, W, H, K))
    #
    a = torch.rand([B, C_out, H1, W1]).cuda()
    g = torch.rand([B, C_out, H1, W1]).cuda()
    wp = torch.rand([C_out, C_in, K, K]).cuda()
    wm = torch.rand([C_out, C_in, K, K]).cuda()
    # w.fill_(1.0)
    # a.fill_(1.0)
    wp_t = wp.permute([1, 0, 2, 3]).flip(dims=[2, 3])  # "transposed" kernel [C_in C_out, K, K]
    wm_t = wm.permute([1, 0, 2, 3]).flip(dims=[2, 3])  # "transposed" kernel [C_in C_out, K, K]
    x_in = torch.rand([B, C_in, H, W]).cuda()
    
    import numpy as np
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    tt = []
    for i in range(10):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        start.record()
        # O = F.conv2d(Variable(Input), Variable(Weight))
        y2 = ratio_conv2d.backward(g, a, wp_t, wm_t, x_in, 1)[0]
        torch.cuda.synchronize()
        end.record()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if i > 0:  # skip first dry run
            tt.append(start.elapsed_time(end))
    
    print("ratio_conv2d Time: ", np.array(tt).mean(), "ms")  # CPU seconds elapsed (floating point)


if __name__ == "__main__":
    test_correctness()
    test_performance()