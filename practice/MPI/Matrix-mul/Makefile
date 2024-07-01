CXX = mpicxx
CXXFLAGS = -O2 -std=c++11

TARGET=main

$(TARGET): $(TARGET).cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

run-default: $(TARGET)
	mpirun -np 4 ./$(TARGET) 1
	mpirun -np 4 ./$(TARGET) 2
	mpirun -np 4 ./$(TARGET) 3
	mpirun -np 4 ./$(TARGET) 4

.PHONY: gen check

check: Check.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^
	./check

gen: Generator.cu
	nvcc -O3 -o gen $^
	./gen 1 1024 1024 1024
	./gen 2 8192 8192 8192
	./gen 3 8192 16384 8192
	./gen 4 16384 16384 8192