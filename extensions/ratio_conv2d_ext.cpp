#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include <vector>

// CUDA forward declarations
//std::vector<torch::Tensor> ratio_conv2d_backward_cuda(
//    torch::Tensor g,
//    torch::Tensor a,
//    torch::Tensor w_p,
//    torch::Tensor w_m,
//    torch::Tensor x_in
//    );

// CUDA forward declarations
std::vector<torch::Tensor> ratio_conv2d_backward_cuda2(
    torch::Tensor g,
    torch::Tensor a,
    torch::Tensor w_p,
    torch::Tensor w_m,
    torch::Tensor x_in,
    int impl_v
    );
//
//
// C++ interface
// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); //CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> ratio_conv2d_backward(
    torch::Tensor g, // [B C_out H1 W1]
    torch::Tensor a, // [B C_out H1 W1]
    torch::Tensor w_p, // [C_in C_out K K]
    torch::Tensor w_m, // [C_in C_out K K]
    torch::Tensor x_in, int impl_v = 0) { // [B C_in H W]
  CHECK_INPUT(g);
  CHECK_INPUT(a);
  CHECK_INPUT(w_p);
  CHECK_INPUT(w_m);
  CHECK_INPUT(x_in);
  AT_ASSERTM(g.size(0) == a.size(0), "batch size mismatch");
  AT_ASSERTM(g.size(0) == x_in.size(0), "batch size mismatch");
  AT_ASSERTM(g.size(1) == a.size(1), "C_out size mismatch");
  AT_ASSERTM(g.size(1) == w_p.size(1), "C_out size mismatch");
  AT_ASSERTM(x_in.size(1) == w_p.size(0), "C_in size mismatch");
  AT_ASSERTM(g.size(2) == a.size(2), "H1 size mismatch");
  AT_ASSERTM(g.size(3) == a.size(3), "W1 size mismatch");
  //AT_ASSERTM(g.size(3) % 4 == 0 , "W1 size is aligned");

  if(impl_v ==0){
//    return ratio_conv2d_backward_cuda(g, a, w_p, w_m, x_in);
    return ratio_conv2d_backward_cuda2(g, a, w_p, w_m, x_in, 0);
  }else{
    return ratio_conv2d_backward_cuda2(g, a, w_p, w_m, x_in, 1);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("backward", &ratio_conv2d_backward, "ratio_conv2d backward (CUDA)");
}