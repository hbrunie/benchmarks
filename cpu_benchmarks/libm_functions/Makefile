all: gnu.exe intel.exe
OPTIM = -O1
OPTIM = -O2
OPTIM =  -std=c++17 -O0
OPTIM = -fp-model precise -fp-model source -std=c++17 -O3
OPTIM = -fp-model fast=2 -std=c++17 -O3
OPTIM = -std=c++17 -O0

gnu.exe: main.cpp
	g++ main.cpp $(OPTIM) -o $@

intel.exe: main.cpp
	icpc  main.cpp $(OPTIM) -o $@ #-no-vec

clean:
	rm -f *.o

cleanall:
	rm -f intel.exe gnu.exe
