all: gnu.exe intel.exe

OPTIONS= -std=c++17
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

gnu.exe: main.cpp foolCompiler.cpp
	g++ $^ $(OPTIONS) $(OPTIM_ALL0) -o $@-ref
	g++ $^ $(OPTIONS) $(OPTIM_ALL1) -o $@-O1
	g++ $^ $(OPTIONS) $(OPTIM_ALL2) -o $@-O2
	g++ $^ $(OPTIONS) $(OPTIM_ALL3) -o $@-O3

intel.exe: main.cpp foolCompiler.cpp
	icpc $^ $(OPTIONS) $(OPTIM_ALL0) -o $@-ref
	icpc $^ $(OPTIONS) $(OPTIM_ALL1) -o $@-O1
	icpc $^ $(OPTIONS) $(OPTIM_ALL2) -o $@-O2
	icpc $^ $(OPTIONS) $(OPTIM_ALL3) -o $@-O3
	icpc $^ $(OPTIONS) $(OPTIM_INTEL1) -o $@-conservative
	icpc $^ $(OPTIONS) $(OPTIM_INTEL2) -o $@-agressive
	icpc $^ $(OPTIONS) $(OPTIM_INTEL3) -o $@-super-agressive

clean:
	rm -f *.o

cleanall:
	rm -f intel.exe* gnu.exe* rm main.cpp.annot main.optrpt

