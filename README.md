# Tensor Contraction Code Generator #

The Tensor Contraction Code Generator (TCCG) generates high-performance (parallel and) vectorized C code for tensor contractions.

From a computational perspective, tensors
can be interpreted as higher dimensional matrices or simply as
multidimensional arrays; likewise, tensor contractions
are a generalization of the matrix-matrix multiplication to higher
dimensions. For instance, A[i,k], B[k,j] and C[i,j] denote two-dimensional
tensors (i.e., matrices) and C[i,j] = A[i,k] * B[k,j] represents a tensor
contraction where the sum over 'k' as well as the loops over 'i' and 'j' are
implicit. Further examples of tensor contractions are: C[i0,j0,j1] = A[i0,k0] * B[j1,k0,j0];
C[i0,j0,j1,i1] = A[i0,k0,i1] * B[j1,k0,j0]; C[i0,j0,j1,i1] = A[k0,i0,k1,i1] * B[k1,j1,k0,j0] ...

Current version: **v0.1.1**

# Key Features
--------------

* TCCG generates high-performance vectorized C code
* TCCG generates code based on three different approaches:
    * GEMM-like Tensor-Tensor Multiplication (GETT): This novel approach to tensor contractions is at the core of our latest publication (see below).
    * Transpose-Transpose-GEMM-Transpose (TTGT)
    * Loops-over-GEMM (LoG)
* Shared-memory parallelism
    * Work in progress: GETT
    * Fully supported: TTGT, LoG
* Support for single- and double-precision
* Auto-Fine-Tuning:
    * Automatically explores a search space of promising implementation candidates
    * The fastest candidate will be selected and returned automatically
    * A performance model guides the search
    * The search space can be limited by the user (via the --maxImplementations=N command line argument)
* Support for multiple instruction sets:
    * AVX2: GETT, TTGT, LoG
    * AVX512: GETT, TTGT, LoG (experimental)
    * CUDA: TTGT, LoG

# Advantages of GETT
---------
GETT's advantages are manifold:
    * GETT-based code is *fully vectorized* and *exploits the cache hierarchy*.
        * Sub-tensors are packed into the caches as needed. Thus, GETT avoids the explicit transposition overhead incurred by TTGT.
    * The *stride-one index is preserved* while packing the sub-tensors into a specified level of the cache hierarchy.
    * *No additional workspace* is required (except for small buffers which fit into the caches).
    * The *arithmetic intensity is retained* for any given tensor contraction.

While GETT exhibits excellent performance across a wide range of tensor contractions, its performance for bandwidth-bound tensor contractions is especially outstanding.

For further information, please see our [(paper)](https://arxiv.org/abs/1607.00145).

# Requirements
--------------

In order to use TCCG, a working C compiler and some BLAS library (e.g., Intel's MKL) as well as the [Tensor Transposition Compiler](https://github.com/HPAC/TTC) (TTC) are required:

* Intel's ICC (>= v15.0, recommended) or g++ (>= v4.8, experimental) 
* Some BLAS library (e.g., [BLIS](https://github.com/flame/blis), [ATLAS](http://math-atlas.sourceforge.net/))
* Tensor Transposition Compiler
* Python (tested with v2.7.5 and v2.7.9)


# Install
---------

1. Clone the repository into a desired directory and change to that location:

    git clone https://github.com/HPAC/tccg.git
    cd tccg

2. Install TCCG:

    python setup.py install --user

3. Export the TCCG_ROOT environment variable (add to your .bashrc):

    export TCCG_ROOT=`pwd`

4. Setup the your BLAS library within the $TCCG_ROOT/config.cfg (default: mkl).

5. You might have to add the installed location to your PATH environment variable:
   
    export PATH=$PATH:~/.local/bin


# Getting Started
-----------------

Please run **tccg --help** to get an overview of TCCG's parameters.

Here is an exemplary input file to TCCG:

    C[a,b,i,j] = A[i,m,a] * B[m,j,b]
    a = 24
    b = 24
    i = 24
    j = 24
    m = 24

TCCG command line arguments: 

    tccg --arch=avx2 --numThreads=1 --floatType=s example.tccg

Further exmaples (.tccg files) can be generated via:

    python bechmark/benchmark.py

# Benchmark
-----------

TCCG provides a [benchmark for tensor contractions](https://github.com/HPAC/tccg/blob/master/benchmark/).

    python benchmark.py

This will generate the input files (.tccg) for TCCG for each of the test-cases within the benchmark.
The tensor contractions within the benchmark are collected from four different publications to cover a broad range of use cases (see paper, Sec. 7.1); this being said, we don't claim that this benchmark is exhaustive in any sense.
If you think that the benchmark is missing certain tensor contractions or sizes, please feel free to contribute to the benchmark.

Since this benchmark may evolve over time and to make comparisons easier, please refer to the current version of the benchmark.

Benchmark version: **v0.1**

# Current Limitations of GETT
--------------
The product of the sizes corresponding to the free indices of each input tensor needs to be a
multiple of 24. This limitation will be lifted in a future version of GETT.

# Citation
-----------
In case you want to refer to TCCG as part of a research paper, please cite the following
article [(pdf)](https://arxiv.org/abs/1607.00145):
```
@article{tccg2016a,
   author      = {Paul Springer and Paolo Bientinesi},
   title       = {{Design of a high-performance GEMM-like Tensor-Tensor Multiplication}},
   archivePrefix = "arXiv",
   eprint = {1607.00145},
   primaryClass = "quant-ph",
   journal     = {CoRR},
   year        = {2016},
   issue_date  = {July 2016},
   url         = {http://arxiv.org/abs/1607.00145}
}
``` 

# Changelog
-----------
V0.1.1:
   * Improved performance for TTGT (based on improvements to [TTC](https://github.com/HPAC/TTC))

# Feedback & Contributions
-----------
We are happy for any feedback or feature requests. Please contact springer@aices.rwth-aachen.de.

We also welcome any contributions to the code base or the benchmark.
