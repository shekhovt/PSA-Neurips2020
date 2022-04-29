/*
 ============================================================================
 Name        : ratio_conv2d.cu
 Author      : ICML Blind
 Version     :
 Copyright   : GPL
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <cooperative_groups.h>

#define DEBUG_LVL 0

#include "slp_util.h"
#include "ndarray/error.kernel.h"
#include "ndarray/intn.h"
#include "ndarray/ndarray_ref.kernel.h"
#include "ndarray/error_cuda.h"
#include "ndarray/defines.h"
#include "ratio_conv2d.h"
//
__device__
void my_print(const intn<4> & a) {
	printf("thread:(%i, %i, %i), (%i, %i, %i, %i)\n", threadIdx.x, threadIdx.y, threadIdx.z, a[0], a[1], a[2], a[3]);
}

//
#define X_PACK 4 // elements processed by 1 thread, cannot be modified
//
//
/* Assumptions:
 * Kernel size K*K, K is odd
 * Last index is contiguous in all tensors (g_out a0_oit, wp, wm, g_in)
 * Weights are fully contiguous
 * g_out, a0_out is by K/2 smaller an all sides than g_in
 * g_out, a0_out are padded so W1 is divisible by 4
 * todo: chunk out_C so that it fits in SM
 */

template<typename F, int max_threads, int active_blocks>
__global__ void __launch_bounds__(max_threads, active_blocks) launch(F kernel) {
	kernel.kernel();
}

template<typename F>
void whatever_launch(dim3 dimGrid, dim3 dimBlock, F kernel) {
//	printf("Grid: %d %d %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
//	printf("Block: %d %d %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
	launch<F, F::MAX_THREADS, F::ACTIVE_BLOCKS> <<<dimGrid, dimBlock>>>(kernel);
	cudaDeviceSynchronize();
	cuda_check_error();
}

static texture<float4, cudaTextureType1D, cudaReadModeElementType> g_out_tex;
static texture<float4, cudaTextureType1D, cudaReadModeElementType> a_tex;

template<int K>
struct ratio_conv2d_backward_kernel_accessor: public ratio_conv2d_backward_kernel_data {
public:
	static
	__HOSTDEVICE__ kernel::ndarray_ref<float, 4> const &  as_const(kernel::ndarray_ref<float, 4> & x) {
		return const_cast<kernel::ndarray_ref<float, 4> const & >(x);
	}
	// size accessors
	__HOSTDEVICE__ int C_out() const {
		return this->_g_out.size(1);
	}
	__HOSTDEVICE__ int C_in() const {
		return this->_g_in.size(1);
	}
	__HOSTDEVICE__ int H() const {
		return this->_g_in.size(2);
	}
	__HOSTDEVICE__ int W() const {
		return this->_g_in.size(3);
	}
	__HOSTDEVICE__ int batch_sz() const {
		return this->_g_in.size(0);
	}
	__HOSTDEVICE__ int H1() const {
		return this->H() - K + 1;
	}
	__HOSTDEVICE__ int W1() const {
		return this->W() - K + 1;
	}
	intn<3> _out_stride_float4;
	intn<3> _out_stride_float;
	//
	// copy constructor from data
	ratio_conv2d_backward_kernel_accessor(const ratio_conv2d_backward_kernel_data & data) :
			ratio_conv2d_backward_kernel_data(data) {
		// this runs on HOST and prepares all the local field, accessible to DEVICE through const memory
		// verify assumptions
		// check assumptions on sizes
		// C_in is even
		//runtime_check(C_in() % BLOCKDIM_Z_TILE == 0);
		// Weight sizes assumption
		runtime_check(K == this->_wp.size(2));
		runtime_check(K == this->_wp.size(3));
		runtime_check(this->_wm.size() == this->_wp.size()); // sizes match
		runtime_check(this->_wm.stride_bytes() == this->_wp.stride_bytes()); // strides match
		runtime_check(this->_wm.is_contiguous_rev());
		runtime_check(this->_wm.size(0) == C_in());
		runtime_check(this->_wm.size(1) == C_out());
		// output tensors assumptions
		runtime_check(this->_g_out.size() == this->_a0.size()); // sizes match
		runtime_check(this->_g_out.stride_bytes() == this->_a0.stride_bytes()); // sizes match
		runtime_check(this->_g_out.is_contiguous_last());
		runtime_check(this->_a0.is_contiguous_last());
		runtime_check(this->_g_out.size(1) == C_out());
		runtime_check(this->_g_out.size(0) == batch_sz());
		// input tensors assumptions
		runtime_check(this->_g_in.is_contiguous_last());
		runtime_check(this->_g_out.size(2) == H1());
		runtime_check(this->_g_out.size(3) == W1());
		//
		runtime_check(this->_x_in.is_contiguous_last());
		runtime_check(this->_x_in.size() == this->_g_in.size());
		runtime_check(this->_x_in.stride_bytes() == this->_g_in.stride_bytes());
		//
//		runtime_check(batch_sz() >= 4);
		//
//		cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
//		cuda_check_error();
		// check padding
		runtime_check(_g_out.size_bytes() % 16 == 0);
//		size_t offset;
//		cudaBindTexture(&offset, g_out_tex, _g_out.ptr(), _g_out.size_bytes());
//		cuda_check_error();
//		std::cout << "ptr = " <<  ((long long)_g_out.ptr()) % 16 << " offset = "<<offset <<"\n";
//		runtime_check(offset == 0);
//		cudaBindTexture(&offset, a_tex, _a0.ptr(), _a0.size_bytes());
//		cuda_check_error();
//		std::cout << "ptr = " <<  ((long long)_a0.ptr()) % 16 << " offset = "<<offset <<"\n";
//		runtime_check(offset == 0);
		_out_stride_float4 = this->out_stride_bytes() / intsizeof(float4);
		_out_stride_float = this->out_stride_bytes() / intsizeof(float);
	}

	~ratio_conv2d_backward_kernel_accessor() {
//		cudaUnbindTexture(g_out_tex);
//		cudaUnbindTexture(a_tex);
	}
	;

	// accessors here HOSTDEVICE for checking outsize
	__HOSTDEVICE__
	const intn<4> out_size() const {
		return intn<4>(batch_sz(), C_out(), H1(), W1());
	}
	__HOSTDEVICE__
	const intn<3> out_stride_bytes() const {
		return this->_g_out.stride_bytes().template erase<3>();
	}

	__HOSTDEVICE__
	const intn<3> out_stride_float() const {
		return this->_out_stride_float;
	}

	__HOSTDEVICE__
	const intn<3> out_stride_float4() const {
		return this->_out_stride_float4;
	}

	__HOSTDEVICE__
	const intn<3> in_stride_bytes() const {
		return this->_g_in.stride_bytes().template erase<3>();
	}

	__HOSTDEVICE__
	const intn<3> w_stride_bytes() const {
		return intn<3>(C_out() * K * K * intsizeof(float), K * K * intsizeof(float), K * intsizeof(float));
	}
	// pointer accessors
	template<typename type, int ndims>
	__DEVICE__
	type & p_global(type * __restrict__ p, const intn<ndims - 1> & stride_bytes, const intn<ndims> & ii) const {
		int off = 0;
#pragma unroll
		for (int d = 0; d < ndims - 1; ++d) {
			off += stride_bytes[d] * ii[d];
		}
		int i = ii[ndims - 1];
		return ((type __RESTRICT__ *) ((char *) p + off))[ii[ndims - 1]];

	}

//	template<typename type, int ndims>
//	__DEVICE__
//	type & p_tex(type * __restrict__ p, const intn<ndims - 1> & stride_bytes, const intn<ndims> & ii) const {
//		int off = 0;
//#pragma unroll
//		for (int d = 0; d < ndims - 1; ++d) {
//			off += stride_bytes[d] * ii[d];
//		}
//		int i = ii[ndims - 1];
//		// need to clamp the address if preloading out-of-bouds
//		//if( off*intsizeof(type) + i < 0){
////			return p[0];
//		//};
////		return ((type __RESTRICT__ *) ((char *) p + off))[ii[ndims - 1]];
//		//texture<DataType, cudaTextureType1D, cudaReadModeElementType> texRef;
//		return tex1Dfetch(g_out_tex, off/intsizeof(float) + i);
//	}

	__DEVICE__
	int out_offset_bytes(int i0, int i1, int i2, int i3) const {
		const intn<4> ii(i0, i1, i2, i3);
		const int ndims = 3;
		const intn<3> stride = out_stride_bytes();
		int off = 0;
#pragma unroll
		for (int d = 0; d < ndims; ++d) {
			off += stride[d] * ii[d];
		}
		return off + i3 * intsizeof(float);
	}

	__DEVICE__
	int out_offset_float(int i0, int i1, int i2, int i3) const {
		const intn<4> ii(i0, i1, i2, i3);
		const int ndims = 3;
		const intn<3> stride_t = out_stride_float();
		int off = 0;
#pragma unroll
		for (int d = 0; d < ndims; ++d) {
			off += stride_t[d] * ii[d];
		}
		return off + i3;
	}

	__DEVICE__
	int out_offset_float4(int i0, int i1, int i2, int i3) const {
		const intn<4> ii(i0, i1, i2, i3);
		const int ndims = 3;
		const intn<3> stride = out_stride_float4();
		int off = 0;
#pragma unroll
		for (int d = 0; d < ndims; ++d) {
			off += stride[d] * ii[d];
		}
		return off + i3;
	}

	__DEVICE__
	afloat4 g_out_read(int i_bytes) const {
//		afloat4 r;
//		asm volatile ("tex.1d.v4.f32.s32 {%0, %1, %2, %3}, [g_out_tex, {%4}];" : "=f"(r[0]), "=f"(r[1]), "=f"(r[2]), "=f"(r[3]) : "r"(i/4) );
//		return r;
//		return tex1Dfetch(g_out_tex, i);
//		return __ldg((float4*) ((char*) _g_out.ptr() + i * intsizeof(float4)));
//		if(i_bytes < 0){
//			printf("foo");
//		};
//		if(i_bytes >= _g_out.size_bytes()){
//			printf("bar %i \n", i_bytes - _g_out.size_bytes());
//		};
		return __ldg((float4*) ((char*) _g_out.ptr() + i_bytes));
	}

	__DEVICE__
	afloat4 a_read(int i_bytes) const {
//		return tex1Dfetch(a_tex, i);
//		return __ldg((float4*) ((char*) _a0.ptr() + i * intsizeof(float4)));
		return __ldg((float4*) ((char*) _a0.ptr() + i_bytes));
	}

	__DEVICE__
	float & g_out(int i0, int i1, int i2, int i3) const {
		const intn<4> ii(i0, i1, i2, i3);
		assert(ii >= 0 && ii < _g_out.size());
		return p_global(this->_g_out.ptr(), out_stride_bytes(), ii);
	}

	__DEVICE__
	float & a0(int i0, int i1, int i2, int i3) const {
		const intn<4> ii(i0, i1, i2, i3);
		assert(ii >= 0 && ii < _a0.size());
		return p_global(this->_a0.ptr(), out_stride_bytes(), ii);
	}

	__DEVICE__
	float & g_in(int i0, int i1, int i2, int i3) const {
		const intn<4> ii(i0, i1, i2, i3);
		assert(ii >= 0 && ii < _g_in.size());
		return p_global(this->_g_in.ptr(), in_stride_bytes(), ii);
	}

	__DEVICE__
	float & flag_in(int i0, int i1, int i2, int i3) const {
		const intn<4> ii(i0, i1, i2, i3);
		assert(ii >= 0 && ii < _x_in.size());
		//return p_global(this->_x_in.ptr(), this->_x_in.stride_bytes().template erase<3>(), intn<4>(i0, i1, i2, i3));
		return p_global(this->_x_in.ptr(), in_stride_bytes(), ii);
	}

	__DEVICE__
	const float & wp(int i0, int i1, int i2, int i3) const {
		const intn<4> ii(i0, i1, i2, i3);
		assert(ii >= 0 && ii < _wp.size());
		return p_global(this->_wp.ptr(), w_stride_bytes(), ii);
	}

	__DEVICE__
	const float & wm(int i0, int i1, int i2, int i3) const {
		const intn<4> ii(i0, i1, i2, i3);
		assert(ii >= 0 && ii < _wm.size());
		return p_global(this->_wm.ptr(), w_stride_bytes(), ii);
	}
};

template<int K>
struct ratio_conv2d_backward_kernel1: public ratio_conv2d_backward_kernel_accessor<K> {
	typedef ratio_conv2d_backward_kernel_accessor<K> parent;
	using parent::batch_sz;
	using parent::C_out;
	using parent::C_in;
	using parent::H;
	using parent::W;
	using parent::H1;
	using parent::W1;
	using parent::g_in;
	using parent::flag_in;
	using parent::g_out;
	using parent::a0;
	using parent::wp;
	using parent::wm;
public:
	static const int ACTIVE_BLOCKS = 4; // stay with 50% occupancy, 64 registers / thread
	static const int MAX_THREADS = 256;
	dim3 dimBlock;
	dim3 dimGrid;
	ratio_conv2d_backward_kernel1(const ratio_conv2d_backward_kernel_data & data) :
			ratio_conv2d_backward_kernel_accessor<K>(data) {
		// block dimensions
		dimBlock = dim3(16, 16, 1);
		dimGrid = dim3(divup(H(), dimBlock.x), divup(W(), dimBlock.y), C_in() * batch_sz());
		// print
	}

	__DEVICE__
	void kernel() const {
		//
		//spatial coordinates
		const int i = blockIdx.x * blockDim.x + threadIdx.x; // H coordinate
		const int j = blockIdx.y * blockDim.y + threadIdx.y; // W coordinate
		//
		const int batch = blockIdx.z / C_in();
		const int c_in = blockIdx.z % C_in();
		// slice all the input accessors at the batch_idx and w at c_in
		//const bool x_in_pos = flag_in(batch, c_in, i, j) > 0;
		const bool x_in_pos = this->_g_in(batch, c_in, i, j) > 0;

		if (i >= H() || j >= W()) { // redundant threads out of output size
			return;
		};

		const int hK = K / 2;  // half kernel sizes

		float r = 0;
		for (int c_out = 0; c_out < C_out(); ++c_out) {
			//
			for (int ky = 0; ky < K; ++ky) {
				const int i1 = i + ky - 2 * hK; // one half for centering the kernel, another half for difference in input size
				// check in range
				if (i1 >= 0 && i1 < H1()) {
#pragma unroll
					for (int kx = 0; kx < K; ++kx) {
						const int j1 = j + kx - 2 * hK;
						if (j1 >= 0 && j1 < W1()) {
							float w_;
							if (x_in_pos > 0) {
								w_ = this->_wp(c_in, c_out, ky, kx);
							} else {
								w_ = this->_wm(c_in, c_out, ky, kx);
							}
							float a_ = this->_a0(batch, c_out, i1, j1);
							float g_ = this->_g_out(batch, c_out, i1, j1);
//							printf("t = (%i,%i,%i)  (kx=%i, ky=%i)   (i1=%i, j1=%i) (b=%i, c_in=%i, c_out=%i)\n", threadIdx.x, threadIdx.y, threadIdx.z, kx, ky, i1, j1, batch, c_in, c_out);
							r += g_ / (1.0f + w_ * a_);
						};
					};
				};
			};
		};
		//this->_
		g_in(batch, c_in, i, j) = r;
	}

	void launch() {
		whatever_launch(dimGrid, dimBlock, *this);
	}

};

template<int K>
struct ratio_conv2d_backward_kernel2: public ratio_conv2d_backward_kernel_accessor<K> {
	typedef ratio_conv2d_backward_kernel_accessor<K> parent;
	using parent::batch_sz;
	using parent::C_out;
	using parent::C_in;
	using parent::H;
	using parent::W;
	using parent::H1;
	using parent::W1;
	using parent::out_offset_bytes;
	using parent::g_out_read;
	using parent::a_read;
//	using parent::g_out;
//	using parent::a0;
	using parent::g_in;
	using parent::flag_in;
	using parent::wp;
	using parent::wm;
public:
	// launch configuration
	static const int ACTIVE_BLOCKS = 4; // stay with 50% occupancy, 64 registers / thread
	static const int MAX_THREADS = 256;
	static const int C_IN_TILE = 2;
	static const int C_OUT_TILE = 32;
	static const int MAX_B_TILE = 4;
	dim3 dimBlock;
	dim3 dimGrid;
	int C_OUT_TILES_B;
	int C_IN_TILES;
	//
	// copy constructor from data
	ratio_conv2d_backward_kernel2(const ratio_conv2d_backward_kernel_data & data) :
			ratio_conv2d_backward_kernel_accessor<K>(data) {
		// block dimensions
		int blockdim_x = min(32, max(divup(W(), X_PACK), K)); //need to cover the kernel
		int blockdim_y = min(8, max(MAX_THREADS / blockdim_x, K)); //need to cover the kernel or else load sucks
		int blockdim_z = min(4, MAX_THREADS / (blockdim_x * blockdim_y));
		dimBlock = dim3(blockdim_x, blockdim_y, blockdim_z);
		// Grid dimensions
		int griddim_x = divup(W(), blockdim_x * X_PACK);
		int griddim_y = divup(H(), blockdim_y);
		C_IN_TILES = divup(C_in(), C_IN_TILE);
		int griddim_z = C_IN_TILES * divup(batch_sz(), blockdim_z);
		dimGrid = dim3(griddim_x, griddim_y, griddim_z);
		//
		C_OUT_TILES_B = divup(C_OUT_TILE, blockdim_z);
		// print
	}

	__DEVICE__
	void kernel() const {

		__shared__ float _shared_wpm[C_OUT_TILE][K][C_IN_TILE][K][2];
#define shared_wp(c_in_t, c_out, ky, kx) _shared_wpm[c_out][ky][c_in_t][kx][0]
#define shared_wm(c_in_t, c_out, ky, kx) _shared_wpm[c_out][ky][c_in_t][kx][1]

		//#define shared_ss(t_b, c_in_t, t_y, t_x, t_xp) _shared_ss[t_x + blockDim.x*(t_y + blockDim.y*t_b) ][c_in_t][t_xp]

		const int t_x = threadIdx.x; // parallel over output x
		const int t_y = threadIdx.y; // parallel over output y
		const int t_b = threadIdx.z; // parallel over output batch dim
		const int x_in_pack = (t_x + blockIdx.x * blockDim.x) * X_PACK;
		const int y_in = t_y + blockIdx.y * blockDim.y;
		const int c_in_tile = (blockIdx.z % C_IN_TILES) * C_IN_TILE;
		const int batch_tile = (blockIdx.z / C_IN_TILES) * blockDim.z;
		const int batch = batch_tile + t_b;
		// writing address -> todo:can release several registers
		//float * p_g_in = &g_in(batch, c_in, y_in, x_in_pack);
		//
		//
		unsigned int flags[C_IN_TILE];
		if (x_in_pack < W() && y_in < H() && batch < batch_sz()) {
			// load flags
			for (int c_in_t = 0; c_in_t < C_IN_TILE; ++c_in_t) {
				flags[c_in_t] = 0;
				const int c_in = c_in_tile + c_in_t;
				if (c_in >= C_in())
					continue;
#pragma unroll
				for (int i = 0; i < X_PACK; ++i) {
					const int x_in = x_in_pack + i;
					if (x_in < W()) {
						bool flag = flag_in(batch, c_in, y_in, x_in_pack + i) > 0;
						if (flag) {
							flags[c_in_t] = flags[c_in_t] | (1 << i);
						};
					};
				};
			};
		};
		//
		// compile-time consts
		// data pointers
		constexpr int hK = K / 2;
		constexpr int pack_offset = divup(2 * hK, X_PACK); // kernel extends by K/2 out of curent in pack, out arrays are smaller and offset by K/2 wrt to in arrays
		constexpr int n_packs = divup((X_PACK + K - 1), X_PACK); // how many needed to cover X_PACK + K/2 + K/2
		//
		unsigned int _pack_flag = 0;
		for (int pack = 0; pack < n_packs; ++pack) {
			const int x_out_pack = x_in_pack - pack_offset * X_PACK + pack * X_PACK; // kx displacement and unpadding
			if (x_out_pack >= 0 && x_out_pack < W1()) {
				_pack_flag = _pack_flag | (1 << pack);
			};
		};
		const unsigned int pack_flag = _pack_flag;
		//
		//
		// sum accumulators
		small_array<float, X_PACK> ss[C_IN_TILE];
		for (int c_in_t = 0; c_in_t < C_IN_TILE; ++c_in_t) {
			for (int i = 0; i < X_PACK; ++i) {
				ss[c_in_t][i] = 0.0f;
			};
		};
		//
		//
		for (int c_out_tile = 0; c_out_tile < C_out(); c_out_tile += C_OUT_TILE) {
			//
			if (C_out() > C_OUT_TILE) {
				__syncthreads();
			};
			// load weights, all warps are used to load a portion
			// todo: optimize using v4 loads
			if (t_x < K && t_y < K) {  // parallel over t_x, t_y
				for (int c_in_t = 0; c_in_t < C_IN_TILE; ++c_in_t) {
					const int c_in = c_in_tile + c_in_t;
					for (int c_out_tb = 0; c_out_tb < C_OUT_TILES_B; ++c_out_tb) {
						const int c_out_t = t_b + c_out_tb * blockDim.z; // parallel over t_b
						if (c_out_t >= C_OUT_TILE)
							break;
						const int c_out = c_out_tile + c_out_t;
						if (c_out >= C_out())
							break;
						if (c_in < C_in()) {
							shared_wp(c_in_t, c_out_t, t_y, t_x)= wp(c_in, c_out, t_y, t_x);
							shared_wm(c_in_t, c_out_t, t_y, t_x)= wm(c_in, c_out, t_y, t_x);
						} else {
							shared_wp(c_in_t, c_out_t, t_y, t_x) = 0.0f;
							shared_wm(c_in_t, c_out_t, t_y, t_x) = 0.0f;
						};
					};
				};
			};
			//
			__syncthreads();
			//
			if (x_in_pack >= W() || y_in >= H() || batch >= batch_sz()) {
				continue;
			};
			const int x_out_pack = x_in_pack - pack_offset * X_PACK;
			int off_bytes = out_offset_bytes(batch, c_out_tile, y_in - hK * 2, x_out_pack);
			//
			const int stride_c_bytes = this->out_stride_bytes()[1];
			const int stride_y_bytes = this->out_stride_bytes()[2];
			// data buffers
			afloat4 g_packs[n_packs];
			afloat4 a_packs[n_packs];

			// data padding (block on the boundaries do not have data here)
#pragma unroll
			for (int pack = 0; pack < n_packs; ++pack) {
//				const int x_out_pack = x_in_pack - pack_offset * X_PACK + pack * X_PACK; // kx displacement and unpadding
//				if (!(x_out_pack >= 0 && x_out_pack < W1())) {
				if (! (pack_flag & (1 << pack)) ) {
					for (int i = 0; i < X_PACK; ++i) {
						g_packs[pack][i] = 0.0f;
						a_packs[pack][i] = 0.0f;
					};
				};
			};
			// loop over c_out channels
#pragma unroll 1
			for (int c_out_t = 0; c_out_t < C_OUT_TILE; ++c_out_t) {
				const int c_out = c_out_tile + c_out_t;
				if (c_out >= C_out())
					break;
				// loop over kernel ky
#pragma unroll
				for (int ky = 0; ky < K; ++ky) {
					const int y_out = y_in - hK * 2 + ky; // ky displacement and unpadding
					if (y_out >= 0 && y_out < H1()) { // will compute something for this line
//load in the data
#pragma unroll
						for (int pack = 0; pack < n_packs; ++pack) {
//							const int x_out_pack = x_in_pack - pack_offset * X_PACK + pack * X_PACK; // kx displacement and unpadding
//							if (x_out_pack >= 0 && x_out_pack < W1()) {
							if (pack_flag & (1 << pack)) {
								g_packs[pack] = g_out_read(off_bytes + ky * stride_y_bytes + pack * intsizeof(float4));
								a_packs[pack] = a_read(off_bytes + ky * stride_y_bytes + pack * intsizeof(float4));
							};
						};

						// load weights from SM to registers for current ky and c_out_t
						small_array<float, K> fwp[C_IN_TILE];
						small_array<float, K> fwm[C_IN_TILE];
#pragma unroll
						for (int c_in_t = 0; c_in_t < C_IN_TILE; ++c_in_t) {
#pragma unroll
							for (int kx = 0; kx < K; ++kx) { // kernel elems
								fwp[c_in_t][kx] = shared_wp(c_in_t, c_out_t, ky, kx); // this read is like broadcast, no need of coalescing
								fwm[c_in_t][kx] = shared_wm(c_in_t, c_out_t, ky, kx);
							};
						};
						// process the line
#pragma unroll
						for (int out_i = -hK; out_i < X_PACK + hK; ++out_i) { // output read locations
							const int pack = (out_i - hK + pack_offset * X_PACK) / X_PACK;
							const int pack_i = (out_i - hK + pack_offset * X_PACK) % X_PACK;
							// data for this location from the buffers
							const float g = g_packs[pack][pack_i];
							const float a = a_packs[pack][pack_i];
							// kernel kx location
#pragma unroll
							for (int kx = 0; kx < K; ++kx) {
								const int in_i = out_i + (K - kx - 1) - K / 2;
								if (in_i < 0 || in_i >= X_PACK){ // static
									continue;
								};
								// input channel tile
#pragma unroll
								for (int c_in_t = 0; c_in_t < C_IN_TILE; ++c_in_t) {
									const bool flag = (flags[c_in_t] & (1 << in_i)) > 0;
									const float w = flag ? fwp[c_in_t][kx] : fwm[c_in_t][kx];
									const float r = 1.0f / (1.0f + a * w);
									ss[c_in_t][in_i] += g * r;
								};
							};
						};
					};
				};
				off_bytes += stride_c_bytes;
			}
		};


		if (y_in >= H() || batch >= batch_sz()) {
			return;
		};

		// save result
		{
			const int x_in_pack = (t_x + blockIdx.x * blockDim.x) * X_PACK;
			for (int i = 0; i < X_PACK; ++i) {
				const int x_in = x_in_pack + i;
				if (x_in < W()) {
					for (int c_in_t = 0; c_in_t < C_IN_TILE; ++c_in_t) {
						const int c_in = c_in_tile + c_in_t;
						if (c_in >= C_in())
							continue;
//					float s = shared_ss(t_b, c_in_t, t_y, t_x, i);
						float s = ss[c_in_t][i];
						g_in(batch, c_in, y_in, x_in) = s;
						//p_g_in[i] = ss[ct][i];
					};
				};
			};
		};

#undef shared_wp
#undef shared_wm
	}

	void launch() {
		whatever_launch(dimGrid, dimBlock, *this);
	}
};

#include "ndarray/ndarray.h"

void ration_conv2d_backward(const ratio_conv2d_backward_kernel_data & _data, int impl_v = 0) {
	cudaDeviceSynchronize();
	cuda_check_error();
	ratio_conv2d_backward_kernel_data data = _data;
	intn<4> sz = data._g_out.size();
	ndarray<float, 4> g_out;
	ndarray<float, 4> a0;
	if (sz[3] % 4 != 0) { // not aligned width
		intn<4> sza = sz;
		sza[3] = roundup(sz[3], 4);
		g_out.create<memory::GPU>(sza, { 3, 2, 1, 0 }); //aligned pitch
		a0.create<memory::GPU>(sza, { 3, 2, 1, 0 }); //aligned pitch
//		g_out.permute_dims( { 3, 2, 1, 0 }).reshape(g_out.size().prod()) << 0.0f;
		        //ndarray_ref<int, 1>((int*)g_out.ptr(), g_out.size().prod(), sizeof(float)) << 0;
		cudaMemset(g_out.ptr(), 0.0f, g_out.size().prod() * sizeof(float));
		cudaMemset(a0.ptr(), 0.0f, a0.size().prod() * sizeof(float));
		cudaDeviceSynchronize();
		cuda_check_error();
		g_out.size(3) = sz[3]; //view
		g_out.permute_dims( { 3, 2, 1, 0 }) << ndarray_ref<float, 4>(data._g_out, ndarray_flags::device_only).permute_dims( { 3, 2, 1, 0 });
//		a0.permute_dims( { 3, 2, 1, 0 }).reshape(a0.size().prod()) << 0.0f;
//		ndarray_ref<int, 1>((int*)a0.ptr(), a0.size().prod(), sizeof(float)) << 0;
		a0.size(3) = sz[3]; //view
		a0.permute_dims( { 3, 2, 1, 0 }) << ndarray_ref<float, 4>(data._a0, ndarray_flags::device_only).permute_dims( { 3, 2, 1, 0 });
		cudaDeviceSynchronize();
		cuda_check_error();
		data._g_out = g_out.kernel();
		data._a0 = a0.kernel();
	};
	int K = data._wp.size(2);
	if (impl_v == 0) {
		switch (K) {
		case 1:
			ratio_conv2d_backward_kernel1<1>(data).launch();
			break;
		case 3:
			ratio_conv2d_backward_kernel1<3>(data).launch();
			break;
		case 5:
			ratio_conv2d_backward_kernel1<5>(data).launch();
			break;
		default:
			throw_error("No Template Instantiation for Kernel size = ") << K << "\n";
		};
	} else {
		switch (K) {
		case 1:
			ratio_conv2d_backward_kernel2<1>(data).launch();
			break;
		case 3:
			ratio_conv2d_backward_kernel2<3>(data).launch();
			break;
		case 5:
			ratio_conv2d_backward_kernel2<5>(data).launch();
			break;
		default:
			throw_error("No Template Instantiation for Kernel size = ") << K << "\n";
		};
	};
	//dlm_write(out_val,"out_val.vol");
}

