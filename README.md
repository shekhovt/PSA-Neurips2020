# Sample-Analytic Gradient Estimators for Stochastic Binary Networks

## Gradient Accuracy Evaluation

run python gradeva/test_grad1.py

This will read model parameters in the saved chekpoints, estimate gradient using different methods, including the exact enumeration method and produce the plots in
exp/gradeval/chk-*/test_grad/score/

The number of samples may be set in the "main". For this test, we use a basic non-optimized implementation of all methods found in gradeval/model.py


## PSA CUDA extension

To be able to run PSA method, it is needed to compile the CUDA extension. This is done by running

python extensions/setup.py

This should do the job, but there might be issues with the version of cmake and compute capability settings. Go to the directory
./extensions/ration_conv2d/. You need to build it using cmake. In CMakeLists.txt choose the GPU compute capability in  set(CMAKE_CUDA_ARCHITECTURES "60;61")
If you were able to work through these, copy the dynamic library generated in
extensions/build/lib-platform/
e.g. extensions/lib.linux-x86_64-3.7/ratio_conv2d.cpython-37m-x86_64-linux-gnu.so 
Copy that to extensions/
Test the extension by running
python /extensions/ratio_conv.py
The test should verify th


## Learning Experiments

For the learning experiments we used optimized implementations of all methods found in models/methods.py
The experiments from the paper can be reproduced by running
experiments/exp_cifar3.py

This will configure and run sequentially all methods. Executing it in a parallel process will learn using the next method on the list, i.e. in parallel. The learning may be continued from the last state by running

experiments/train.py runs/CIFAR/CIFAR_S1-NLL-FlipCrop/train.method=ST #epochs

The learning will save logs and plot results in runs/CIFAR/CIFAR_S1-NLL-FlipCrop/