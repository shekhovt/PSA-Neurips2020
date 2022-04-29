#include <torch/extension.h>

#include <ratio_conv2d.h>
#include <ndarray/ndarray_ref.host.h>


template<int dims>
ndarray_ref<float, dims> to_ndarray(const torch::Tensor & x){
    intn<dims> size;
    intn<dims> stride_bytes;
    auto xa = x.packed_accessor<float,4,torch::DefaultPtrTraits,int>();
    //
    for(int i=0; i<dims; ++i){
        size[i] = xa.size(i);
        stride_bytes[i] = xa.stride(i)*sizeof(float);
    };
    return ndarray_ref<float, dims>(xa.data(), size, stride_bytes, ndarray_flags::device_only);
};

std::vector<torch::Tensor> ratio_conv2d_backward_cuda2(
    torch::Tensor g_out,
    torch::Tensor a,
    torch::Tensor w_p,
    torch::Tensor w_m,
    torch::Tensor x_in,
    int impl_v) {

  auto g_in = at::empty_like(x_in);

  ratio_conv2d_backward_kernel_data data(to_ndarray<4>(w_p), to_ndarray<4>(w_m), to_ndarray<4>(g_out), to_ndarray<4>(a), to_ndarray<4>(x_in), to_ndarray<4>(g_in));
  ration_conv2d_backward(data,impl_v);

  return {g_in};
}
