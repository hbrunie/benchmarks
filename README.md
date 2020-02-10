# benchmarks
small benchmarks
TODO: dump relError and speedup against different compiler (GNU, INTEL, LLVM) different option (fpmodel, ffast-math, O0 O1 O2 O3), different architecture (Haswell, KNL, NVIDIA V100) check vectorization is done with compiler.
Each time speedup is for expf (and \_\_expf for cuda) compare to exp.
Speedup with:
  - double b = (double) expf((float) a); // double a;
  - double b = (double) expf(a); // float a;
  - float b = expf((float) a); // double a;
  - float b = expf(a); // float a;
For different range: pow(2,x) with x in -100 ... +100 ( fine grain around 0, then coarse) 0:1 1:2 2:3 3:4 4:8 8:16 16:32 ... same with negative values.

The idea is to show the tradeoffs between loss of accuracy (relative error get bigger) and gain of performance (speedup).
Show that it depends a lot on the context (vectorization, range of values, architecture, compiler).
