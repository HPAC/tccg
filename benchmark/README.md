# Tensor Contraction Benchmark #

The tensor contractions within the benchmark are collected from four different publications to cover a broad range of use cases (see paper, Sec. 7.1); this being said, we don't claim that this benchmark is exhaustive in any sense.
If you think that the benchmark is missing certain tensor contractions or sizes, please feel free to contribute to the benchmark.

Since this benchmark may evolve over time and to make comparisons easier, please refer to the current version of the benchmark.

To generate the input files (.tccg) corresponding to all the test-cases within the benchmark please run:

    python benchmark.py

In addition to this benchmark you can also generate further tensor contractions via:

    python transC.py

The examples generated via transC.py correspond to those of Fig. 10 of our publication and highlight the performance differences between GETT, TTGT and LoG for this specific TC.

# Performance Results
---------------------
We evaluated the performance of TCCG on a single core of an Intel Xeon E5-2680 v3 CPU based on the Intel Haswell
microarchitecture. ECC is enabled, and both Intel Speedstep and Intel TurboBoost
are disabled for all measurements. The C++ compiler of choice is Intel's icpc
16.0.1 20151021 with flags -O3 -xhost -mkl.

![ttc](https://github.com/HPAC/tccg/blob/master/benchmark/GETT_LoG_TTGT.png)

![ttc](https://github.com/HPAC/tccg/blob/master/benchmark/benchmark.png)

![ttc](https://github.com/HPAC/tccg/blob/master/benchmark/transC.png)

Benchmark version: **v0.1**
