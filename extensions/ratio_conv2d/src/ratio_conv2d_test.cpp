#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <assert.h>
#include <string>
#include <iostream>
#include <sstream>
#include <cmath>
#include <vector>

#include "ratio_conv2d.h"
#include "ndarray/ndarray.h"
#include "slp_util.h"

int main() {
//	test_ops();

	int batch = 1;
	//
	int seed = 1;
	srand(seed);
	//

//	std::vector<int> cs = { 1, 3, 17, 211 };
//	std::vector<int> szs = { 1, 3, 6, 16, 17, 211 };
//	std::vector<int> Ks = { 1, 3, 5 };

//	// benchmark
	batch = 128;
	std::vector<int> c_ins = { 32 };
	std::vector<int> c_outs = { 32 };
	std::vector<int> szs = { 32, 30};
	std::vector<int> Ks = { 3};
	// test
//	std::vector<int> c_ins = {1};
//	std::vector<int> c_outs = {2};
//	std::vector<int> szs = {1, 2, 4};
//	std::vector<int> Ks = { 1, 3 };

	std::cout << "TEST LAUNCH\n";
	for (int ci = 0; ci < c_ins.size(); ++ci) {
		int C_in = c_ins[ci];
		int C_out = c_outs[ci];
		for (int si = 0; si < szs.size() - (ci > 2); ++si) {
			int H1 = szs[si];
//			int H1 = 1;
			int W1 = szs[si];

			for (int ki = 0; ki < Ks.size(); ++ki) {
				int K = Ks[ki];
				//
				int H = H1 + K - 1;
				int W = W1 + K - 1;
				std::cout << "K=" << K << " W="<< W << "\n";
				//
				//
				ndarray<float, 4> g_out;
				auto sz = intn<4>( { batch, C_in, H, W });
				auto sz1 = intn<4>( { batch, C_out, H1, W1 });
//		auto sz1_pad = intn<4>( { batch, C_out, H, W });
//				auto sz1_pad = intn<4>( { batch, C_out, roundup(H1, 4), roundup(W1, 4) }); // no padding, original data
				g_out.create<memory::GPU>(sz1, { 3, 2, 1, 0 });
				cudaDeviceSynchronize();
				cuda_check_error();
				std::cout << g_out << "\n";
				//g_out = g_out.subrange(intn<4>(0,0,0,0), sz1);
//				g_out.size(2) = H1;
//				g_out.size(3) = W1;
				g_out << 0.0f;
//				for(int i=0; i< g_out.size(3); ++i){
//					g_out.subdim<3>(i) += float(i);
//				};
//				for(int i=0; i< g_out.size(1); ++i){
//					g_out.subdim<1>(i) += float(i);
//				};
				g_out.subdim<1>(1) += 1.0f;
				ndarray<float, 4> y;
				y.create<memory::CPU>(g_out);
				y << g_out;
				auto view = y.permute_dims({0,1,3,2}).subdim<0>(0).subdim<0>(0);
//				print_array("g_out=",view,0);


				ndarray<float, 4> a0;
				a0.create<memory::GPU>(sz1, { 3, 2, 1, 0 });
//				a0.size(2) = H1;
//				a0.size(3) = W1;
				//a0 = a0.subrange(intn<4>(0,0,0,0), sz1);
				a0 << 1.0f;
				//
				ndarray<float, 4> g_in;
				g_in.create<memory::GPU>( { batch, C_in, H, W }, { 3, 2, 1, 0 });
				g_in << 1.0f;
				ndarray<float, 4> x_in;
				x_in.create<memory::GPU>( { batch, C_in, H, W }, { 3, 2, 1, 0 });
				x_in << 1.0f;
				//
				ndarray<float, 4> wp;
				wp.create<memory::GPU>( { C_in, C_out, K, K }, { 3, 2, 1, 0 });
				wp << 1.0f;
				ndarray<float, 4> wm;
				wm.create<memory::GPU>( { C_in, C_out, K, K }, { 3, 2, 1, 0 });
				wm << 1.0f;
				//
				//
				ratio_conv2d_backward_kernel_data data(wp, wm, g_out, a0, x_in, g_in);


				float t1 = 1e10;
				for (int tr = 0; tr < 5; ++tr) {
					//
					cudaEvent_t start, stop;
					cudaEventCreate(&start);
					cudaEventCreate(&stop);
					cudaEventRecord(start);
					//
					ration_conv2d_backward(data, 0);
					//
					cudaEventRecord(stop);
					cudaEventSynchronize(stop);
					float milliseconds = 0;
					cudaEventElapsedTime(&milliseconds, start, stop);
					t1 = std::min(t1, milliseconds);
					std::cout << "time:" << milliseconds << "ms\n";
					//
					cudaDeviceSynchronize();
					cuda_check_error();
					ndarray<float, 4> y1;
					y1.create<memory::CPU>(g_in);
					y1 << g_in;
					//
					cudaEventCreate(&start);
					cudaEventCreate(&stop);
					cudaEventRecord(start);
					//
					ration_conv2d_backward(data, 1);
					//
					cudaEventRecord(stop);
					cudaEventSynchronize(stop);
					milliseconds = 0;
					cudaEventElapsedTime(&milliseconds, start, stop);
					t1 = std::min(t1, milliseconds);
					std::cout << "time:" << milliseconds << "ms\n";
					//
					ndarray<float, 4> y2;
					y2.create<memory::CPU>(g_in);
					y2 << g_in;
					//
//			print_array("y1=",y1.subdim<0>(0).subdim<0>(0),0);
//			print_array("y2=",y2.subdim<0>(0).subdim<0>(0),0);
					//
					for (int b = 0; b < y1.size(0); ++b) {
						for (int c = 0; c < y1.size(1); ++c) {
							for (int i = 0; i < y1.size(2); ++i) {
								for (int j = 0; j < y1.size(3); ++j) {
									runtime_check(std::abs(y1(b, c, i, j) - y2(b, c, i, j)) < 1e-6);
								};
							};
						};
					};
					//
				};
			};
		};
		//std::cout << "min time:" << t1 << "ms\n";
		//
	};
	std::cout << "TEST PASSED \n";
	return 0;
}
