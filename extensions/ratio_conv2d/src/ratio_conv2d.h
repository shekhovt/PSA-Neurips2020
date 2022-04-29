#include "ndarray/ndarray_ref.kernel.h"

struct ratio_conv2d_backward_kernel_data {
public:
	kernel::ndarray_ref<float, 4> _wp; // [C_in C_out K K]
	kernel::ndarray_ref<float, 4> _wm; // [C_in C_out K K]
	kernel::ndarray_ref<float, 4> _g_out; // [B C_out H1 W1]
	kernel::ndarray_ref<float, 4> _a0; // [B C_out H1 W1]
	kernel::ndarray_ref<float, 4> _x_in;  // [B C_in H W]
	kernel::ndarray_ref<float, 4> _g_in; // [B C_in H W]
	kernel::ndarray_ref<float, 4> _g_w; // [C_out C_in K K]

	ratio_conv2d_backward_kernel_data(const kernel::ndarray_ref<float, 4> & wp, const kernel::ndarray_ref<float, 4> & wm, const kernel::ndarray_ref<float, 4> & g_out,
	        const kernel::ndarray_ref<float, 4> & a0, const kernel::ndarray_ref<float, 4> & x_in, const kernel::ndarray_ref<float, 4> & g_in) :
			_wp(wp), _wm(wm), _g_out(g_out), _a0(a0), _x_in(x_in), _g_in(g_in) {
	}
	//
};


void ration_conv2d_backward(const ratio_conv2d_backward_kernel_data & data, int impl_v);

int main();