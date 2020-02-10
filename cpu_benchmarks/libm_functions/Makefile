all: gnu intel
OPTIM = -O2
gnu: main.cpp
	g++ main.cpp $(OPTIM) -o $@

intel: main.cpp
	icpc -qopt-report=5 -xMIC-AVX512 -S main.cpp $(OPTIM) -o $@.s #-no-vec
	icpc -qopt-report=5 -xMIC-AVX512 main.cpp $(OPTIM) -o $@ #-no-vec
