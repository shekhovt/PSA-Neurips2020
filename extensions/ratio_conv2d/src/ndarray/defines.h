#pragma once

//#ifndef __CUDACC__
//	#ifndef __host__
//		#define __host__
//	#endif
//	#ifndef __device__
//		#define __device__
//	#endif
////	#ifndef __restrict__
////		#define __restrict__ restrict
////	#endif
//
//#endif
//
//#ifndef __forceinline__
//#ifndef __CUDACC__
//	#if _MSC_VER > 1000
//		#define __forceinline__ __forceinline
//		#define __restrict__ __restrict
//	#else// not msvc
//		#define __forceinline__ inline
//	#endif
//#endif
//#endif

//#if defined _WIN32 || defined __CYGWIN__
//  #ifdef BUILDING_DLL
//    #ifdef __GNUC__
//      #define DLL_PUBLIC __attribute__ ((dllexport))
//    #else
//      #define DLL_PUBLIC __declspec(dllexport) // Note: actually gcc seems to also supports this syntax.
//    #endif
//  #else
//    #ifdef __GNUC__
//      #define DLL_PUBLIC __attribute__ ((dllimport))
//    #else
//      #define DLL_PUBLIC __declspec(dllimport) // Note: actually gcc seems to also supports this syntax.
//    #endif
//  #endif
//  #define DLL_LOCAL
//#else
//  #if __GNUC__ >= 4
//    #define DLL_PUBLIC __attribute__ ((visibility ("default")))
//    #define DLL_LOCAL  __attribute__ ((visibility ("hidden")))
//  #else
//    #define DLL_PUBLIC
//    #define DLL_LOCAL
//  #endif
//#endif

#if defined(_MSC_VER)
#define __HALT__ __debugbreak()
#define __FORCEINLINE__ __forceinline
#define __NOINLINE__ __declspec(noinline)
#else
#define __HALT__ __asm__ volatile("int3")
#define __FORCEINLINE__ __attribute__((always_inline)) inline
#define __NOINLINE__ __attribute__((noinline))
#endif

#ifdef __CUDACC__ //__CUDA_ARCH__
	#undef __HOSTDEVICE__
	#undef __HOST__
	#define __HOSTDEVICE__ __host__ __device__ __forceinline__
	#define __HOST__ __host__ __forceinline__
	#define __RESTRICT__ __restrict__
#else
	#undef __HOSTDEVICE__
	#undef __HOST__
	#define __HOSTDEVICE__ __FORCEINLINE__
	#define __HOST__ __FORCEINLINE__
	#define __RESTRICT__
#endif

#define __DEVICE__ __device__ __forceinline__

#if (_MSC_VER > 1000) && (_MSC_VER <= 1800) // c++11 support (lack of)
	#pragma push_macro("constexpr")
	#ifndef __cpp_constexpr
	#define constexpr const
	#endif
	#define __attribute__(A) /* do nothing */
	// this is what VS2013 can compile to implement constructor inheritance, up to 5 arguments. variadic inside the macro does not work
	template <typename A> const A & get_some();
	template <typename A> struct get_object_without_instantiation{
		static const A * a;
	};
	template <typename C, typename... AA> struct test_constructor{
		static const C c = C(*get_object_without_instantiation<AA>::a...);
	};
	#define inherit_constructors(derived, parent)\
	template <typename A0>\
	derived(const A0& a0, parent X = test_constructor<parent,A0>::c) : parent(a0){}\
	template <typename A0, typename A1>\
	derived(const A0& a0, const A1& a1, parent X = test_constructor<parent,A0,A1>::c) : parent(a0,a1){}\
	template <typename A0, typename A1, typename A2>\
	derived(const A0& a0, const A1& a1, const A2& a2, parent X = test_constructor<parent,A0,A1,A2>::c ) : parent(a0,a1,a2){}\
	template <typename A0, typename A1, typename A2, typename A3>\
	derived(const A0& a0, const A1& a1, const A2& a2, const A3& a3, parent X = test_constructor<parent,A0,A1,A2,A3>::c) : parent(a0, a1, a2, a3){}\
	template <typename A0, typename A1, typename A2, typename A3, typename A4>\
	derived(const A0& a0, const A1& a1, const A2& a2, const A3& a3, const A4& a4, parent X = test_constructor<parent, A0, A1, A2, A3,A4>::c) : parent(a0, a1, a2, a3,a4){}
#else
	#define inherit_constructors(derived, base) using base::base;
#endif

struct host_stream{};
struct device_stream{};

#define intsizeof(type) int(sizeof(type))
