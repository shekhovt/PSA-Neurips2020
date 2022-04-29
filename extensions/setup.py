"""
Running this should build and connect the extension
If this fails check that ratio_conv2d/CmakeLists.txt can be build with cmake and make
"""
from setuptools import setup, Extension
from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import sys
import os
import distutils

# distutils.dir_util.remove_tree

if os.environ.get("TORCH_CUDA_ARCH_LIST", None) is None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5"

sources=["./ratio_conv2d_ext.cpp",
        # "./ratio_conv2d_kernel.cu",
         "./ratio_conv2d_kernel2.cpp"]

if len(sys.argv) == 1:
      sys.argv.append("build_ext")
      # sys.argv.append("--inplace")

d = os.path.dirname(os.path.abspath(__file__))
build_dir = d + '/ratio_conv2d/build'
include_dirs = [d + "/ratio_conv2d/src"]
library_dirs = [d + "/ratio_conv2d/lib"]
libraries = ["ratio_conv2d"]
if not os.path.exists(build_dir):
    os.makedirs(build_dir)
os.chdir(build_dir)
os.system('cmake ..')
os.system('make')
os.chdir(d)
lib = d + '/ratio_conv2d/lib/libratio_conv2d.a'
# fix the Extesion build ignorance of the library modifications
if not os.path.exists(lib) or os.path.getmtime(lib) > os.path.getmtime(d + '/ratio_conv2d_ext.cpp'):
    os.system('touch ./ratio_conv2d_ext.cpp')

setup(name='ratio_conv2d_ext',
    ext_modules=[cpp_extension.CppExtension('ratio_conv2d_ext', sources, include_dirs=include_dirs, library_dirs=library_dirs, libraries=libraries)],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

# CUDAExtension
#-Wdeprecated-declarations
#-Wsign-compare