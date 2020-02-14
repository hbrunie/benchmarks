all: gnu.exe-ref gnu.exe-O1 gnu.exe-O2 gnu.exe-O3 intel.exe-ref intel.exe-O1 intel.exe-O2 intel.exe-O3  intel.exe-conservative  intel.exe-agressive  intel.exe-super-agressive

OPTIONS= -std=c++17 -g
## None
OPTIM_ALL0 = -O0
## First degree
OPTIM_ALL1 = -O1
## Second degree
OPTIM_ALL2 = -O2
## Third degree
OPTIM_ALL3 = -O3
INTEL_REPORT= -qopt-report-annotate -qopt-report=5
## Conservative
OPTIM_INTEL1 = -O3 $(INTEL_REPORT) -fp-model precise -fp-model source
## Agressive
OPTIM_INTEL2 = -O3 $(INTEL_REPORT)
## Super Agressive
OPTIM_INTEL3 = -O3 $(INTEL_REPORT) -fp-model fast=2

gnu.exe-ref: main.cpp foolCompiler.cpp
	g++ $^ $(OPTIONS) $(OPTIM_ALL0) -o $@
gnu.exe-O1: main.cpp foolCompiler.cpp
	g++ $^ $(OPTIONS) $(OPTIM_ALL1) -o $@
gnu.exe-O2: main.cpp foolCompiler.cpp
	g++ $^ $(OPTIONS) $(OPTIM_ALL2) -o $@
gnu.exe-O3: main.cpp foolCompiler.cpp
	g++ $^ $(OPTIONS) $(OPTIM_ALL3) -o $@

intel.exe-ref: main.cpp foolCompiler.cpp
	icpc $^ $(OPTIONS) $(OPTIM_ALL0) -o $@
intel.exe-O1: main.cpp foolCompiler.cpp
	icpc $^ $(OPTIONS) $(OPTIM_ALL1) -o $@
intel.exe-O2: main.cpp foolCompiler.cpp
	icpc $^ $(OPTIONS) $(OPTIM_ALL2) -o $@
intel.exe-O3: main.cpp foolCompiler.cpp
	icpc $^ $(OPTIONS) $(OPTIM_ALL3) -o $@
intel.exe-conservative: main.cpp foolCompiler.cpp
	icpc $^ $(OPTIONS) $(OPTIM_INTEL1) -o $@
intel.exe-agressive: main.cpp foolCompiler.cpp
	icpc $^ $(OPTIONS) $(OPTIM_INTEL2) -o $@
intel.exe-super-agressive: main.cpp foolCompiler.cpp
	icpc $^ $(OPTIONS) $(OPTIM_INTEL3) -o $@

clean:
	rm -f *.o

cleanall:
	rm -f intel.exe* gnu.exe* rm *.annot *.optrpt

