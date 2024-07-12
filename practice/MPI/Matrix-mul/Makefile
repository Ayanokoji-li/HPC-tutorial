CXX = mpicxx
CXXFLAGS = -O2 -std=c++11

TARGET=main
Data_Dir = ./attachment


$(TARGET): $(TARGET).cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

run-default: $(TARGET)
	mpirun -np 4 ./$(TARGET) 1
	mpirun -np 4 ./$(TARGET) 2
	mpirun -np 4 ./$(TARGET) 3
	mpirun -np 4 ./$(TARGET) 4

.PHONY: gen check clean

check: Check.cpp
	mkdir -p $(Data_Dir)
	$(CXX) $(CXXFLAGS) -o $@ $^
	./check

gen: Generator.cu
	nvcc -O3 -o gen $^
	mkdir -p $(Data_Dir)
	./gen 1 1024 1024 1024
	./gen 2 8192 8192 8192
	./gen 3 8192 16384 8192
	./gen 4 16384 16384 8192

clean:
	rm -f $(TARGET) check gen
	rm -rf ./attachment