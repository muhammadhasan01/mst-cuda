# Minimum Spanning Tree - CUDA
This repository is an implementation of __Minimum Spanning Tree__ (MST) with parallelization using CUDA.

## Running the Program

To run the program, you need to have CUDA Setup first, you can see the details [here](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html). After that, you basically need to compile and make an executable file of the program. So, you can move to the src folder first, then you can type in the command prompt:

```
nvcc -o <program_name> MST_CUDA.c && <program_name>.exe
```


## Approach on Getting Minimum Spanning Tree with Parallelization

To get the minimum spanning tree on a particular graph, we use _Kruskal Algorithm_, which is to use an edge list with weight included, and then sort the edge list ascendingly by its weight, then add every edge from left to right in that list, without making any cycle (handled by using Disjoint Set Union). So for the sort part we use the parallelization with CUDA, here we use *bitonic sort*.

<div align="center">

![BitonicSort](https://upload.wikimedia.org/wikipedia/commons/b/bd/BitonicSort1.svg)

</div>

**Bitonic sort** is a comparison-based sorting algorithm that can be run in parallel. It focuses on converting a random sequence of numbers into a bitonic sequence, one that monotonically increases, then decreases. Rotations of a bitonic sequence are also bitonic. You can read more about this sorting algorithm [here](https://en.wikipedia.org/wiki/Bitonic_sorter).

We will focus on how we use CUDA for bitonic sort parallelization. Our parallel solution is to first make the length to the nearest power of two, and fill in the gap with large numbers (INT_MAX), this is because our bitonic sort implementation doesn’t use recursive and therefore needs a length with a power of two. And then, we use `cudaMalloc` and `cudaMemcpy` to copy the data to the GPU. We then define the number of threads (`num_threads`) with value equal to `min(length, 512)` with `length` being the length of the data array (after the addition of the dummy element). We also define the number of blocks (`num_blocks`) with value equal to the `length / num_thread`. The device function `bitonic_sort_kernel` is used for comparing and swapping unordered elements according to bitonic rules (there are increasing and decreasing orders). Last, we copy the sorted data array back from the device to the host.

To see more detail, you can look at the implementation code [MST_CUDA.cu](./src/MST_CUDA.cu).

## Speedup Program Analyzation

In this section we will compare the speed between serial and parallel program, we use google collab as the environment to run the program. Here are the results:

| nodes 	| serial             	| parallel           	| speedup 	|
|-------	|--------------------	|--------------------	|---------	|
| 100   	| 0.009601000000 ms  	| 0.009177000000 ms  	| 1.046   	|
| 500   	| 0.270724000000 ms  	| 0.235964000000 ms  	| 1.147   	|
| 1000  	| 1.247940000000 ms  	| 1.060560000000 ms  	| 1.176   	|
| 3000  	| 22.257306000000 ms 	| 10.049784000000 ms 	| 2.215   	|

From the table above, we can see that the parallel program is generally faster than the serial program. The speedup is bigger when we use a larger dataset. That is because the parallel program has an overhead time to create, manage, and delete the thread. So, for bigger datasets the overhead time can be neglected.

## Serial Program Pastebin

Because the serial program is not required to be included in the repository, we attach the serial program via
pastebin [here](https://pastebin.com/HmiM47vB).

## Authors

- Muhammad Hasan (13518102) - [muhammadhasan01](https://github.com/muhammadhasan01)
- Naufal Dean Anugrah (13518123) - [naufal-dean](https://github.com/naufal-dean)
