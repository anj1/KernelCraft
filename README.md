# KernelCraft-GPU Benchmark v1 🧙‍♀️
This project is an LLM Benchmark for writing compute kernels on compute accelerators like GPUs, TPUs, ...

Writing performant GPU kernels by hand is [hard](https://siboehm.com/articles/22/CUDA-MMM). Making optimal use of hardware capabilities requires care and expertise. LLMs are increasingly being used for coding tasks, but so far, for GPU kernels, they do not seem to be as good at human experts. This project was conceived as a way to objectively benchmark the ability of LLMs to write GPU kernels for non-contrived, real-world tasks, and to help advance the state of the art in this area.

The benchmark is organized as a set of ground-truth examples in python, which are fed to the LM (or human participant). The expected output is code to run on a compute accelerator. This kernel is then compiled and run, and its output values are compared with the ground truth. If they match, the speed of the GPU kernel is evaluated. The ranking is based on how fast the kernels run on the target hardware.

## Benchmark Design

KernelCraft-GPU is designed to be used across a wide range of architectures and kernel programming languages. Thus we impose no requirement for a specific language. In the current implementation, we require CUDA C++ output, but we're soon going to add other architectures. During evaluation, the LLM must be told what language to use, and it must follow this requirement.

So, a single evaluation consists of a problem selected from the problem set, plus:

- The desired target architecture (e.g. "Nvidia Ampere", "Google TPU v6")
- The target language ("CUDA C", "Jax", etc.)
- An optional selection of specific hardware details
  - Number of cores
  - Cache sizes
  - Shared memory sizes
  - Global memory size
  - Bus speed
  - etc.

The hardware details can be as specific as required and can be specified in any format. The goal of the benchmark isn't to test how well the LLM has memorized specific details of each accelerator, but rather to test how well it can generate code _given_ available information.

## Tuning

Usually, kernels handcrafted by human programmers need additional tuning steps (for example, for block sizes)  to reach optimal performance on specific hardware. This is _not_ because the optimal values of these parameters cannot be determined _in principle_, but rather that such determination is very hard. 

However, running tuning requires searching a potentially large space and is time consuming. In addition, it's not unreasonable to expect that a perfect kernel crafting AI ought to be able to determine these parameters automatically, either from first principles or based on e.g. known block sizes from existing code, or by writing code that determines them from first principles. So, in our benchmark, we _don't_ performance tune the generated kernels. We run them as-is.

<!--## Problem Cases

The cases are designed to be simple and concise, but also nontrivial to optimize, and actually have real-world use. They are derived from various problems in AI, materials science, fluid dynamics, computer graphics, optimization, etc.

For some of the problems, like matrix multiplication (GEMM), FFT, and so on, highly performance tuned solutions are already available for some accelerator platforms, in libraries like cublas, cutensor, cufft, etc. A language model _could_ solve these just by reference to its training, so these serve as a baseline.

For other cases, like CCSD, solutions may have been documented in the literature or in libraries, but tuned code doesn't exist, either at all or for the architectures in question. The goal with these is to determine how well the LLM can reason about translating known parallel algorithms into optimal parallel algorithms for specific hardware.-->

## Evaluation

At the basic level, the evaluation consists of the raw achieved speed (in wall time) for every problem and accelerator combination. But to be able to compare different LLMs, we also need an aggregate score on the whole set.

Compute Accelerators are diverse in design and architecture, so an evaluation that is general most not assume any particular underlying design (like the existence of streaming multiprocessors or tensor cores). However, to ensure fairness, it must also correct for differences in accelerator capabilities. This creates the question of how to properly and fairly weigh performance on different accelerators. For example, is it better to achieve 10x speedup on an H100 and an RTX 2080, or 100x on an H100 but just 2x on an RTX 2080? 

One possibility is to just list out the raw achieved speed 


## The Problems

The tasks are designed to be simple and concise, but also nontrivial to optimize, and actually have real-world use. They are derived from various problems in AI, materials science, fluid dynamics, computer graphics, optimization, etc.

Checked boxes indicate if they have been implemented yet.

Linear Algebra:
  - [x] Point-wise addition
  - [x] Matrix multiplication
  - [x] [Cumsum](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)
  - [x] [GEMM](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)
  - [ ] Einsum

Computer Graphics:
  - [x] [Raytracing](https://cs.stanford.edu/people/eroberts/courses/soco/projects/1997-98/ray-tracing/implementation.html)
  - [x] [Mandelbrot set](https://mathr.co.uk/blog/2021-05-14_deep_zoom_theory_and_practice.html)
  - [ ] [Gaussian Splat Rendering](https://arxiv.org/abs/2308.04079)
  - [ ] [Monte Carlo radiation transport (pdf link)](https://indico.cern.ch/event/1123370/contributions/4715934/attachments/2444331/4188477/02_Monte_Carlo_Basics_2022_ULB.pdf)

Fluid dynamics:
  - [x] [Lattice Boltzmann](https://medium.com/swlh/create-your-own-lattice-boltzmann-simulation-with-python-8759e8b53b1c)
  - [ ] [Material Point Method (pdf link)](https://www.math.ucla.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf)

Numerical Methods:
  - [ ] [Fast Fourier Transform (pdf flink)](https://crasr.aut.ac.nz/__data/assets/pdf_file/0003/32295/Seth-Hall-FFT-Optimizations-for-GPU-and-Many-Core-Architectures.pdf)
  - [ ] [PDE methods (reaction-diffusion) (pdf link)](https://people.maths.ox.ac.uk/erban/Education/StochReacDiff.pdf)
  - [ ] [Optimal Transport](https://arxiv.org/abs/2106.01963)
  - [ ] [Fast Multipole Method](https://www.bu.edu/pasi/courses/12-steps-to-having-a-fast-multipole-method-on-gpus/)

Machine Learning:
  - [ ] [Attention](https://arxiv.org/abs/1706.03762)
  - [ ] [Flash Attention](https://arxiv.org/abs/2205.14135)
  - [ ] [Structured Sparse Attention](https://arxiv.org/abs/2102.04010)

Data Analysis:
  - [ ] [Histogram](https://developer.nvidia.com/blog/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/)
  - [x] [Radix sort (pdf link)](https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21572-a-faster-radix-sort-implementation.pdf)
  - [ ] [Clustering (K-means)](https://www.nvidia.com/en-us/glossary/k-means/)
  - [ ] [Graph Analytics](https://www.nvidia.com/en-us/glossary/graph-analytics/)

Optimization:
  - [ ] [Dancing Links (DLX)](https://arxiv.org/abs/cs/0011047)
  - [ ] [Traveling Salesman](https://www.sciencedirect.com/science/article/abs/pii/S0167926024001974)

Computational Chemistry:

  - [ ] [Hartree-Fock](https://web.ornl.gov/~kentpr/thesis/pkthnode13.html)
  - [ ] [Density Functional Theory](https://www.sciencedirect.com/topics/physics-and-astronomy/density-functional-theory)
  - [ ] [Coupled Cluster (CCSD) (pdf link)](https://www.chem.pku.edu.cn/jianghgroup/docs/20190416171616274502.pdf)
  - [ ] [Molecular docking](https://pmc.ncbi.nlm.nih.gov/articles/PMC3151162/)

