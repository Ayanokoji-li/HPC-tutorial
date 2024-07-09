CXX=mpicxx
CXXFLAGS=-O2 -std=c++11

TARGET=main

$(TARGET): $(TARGET).cpp
	$(CXX) $(CXXFLAGS) -o $@ $^


.PHONY: run-default, clean

run-default: $(TARGET)
	mpirun -np 4 ./$(TARGET) 10 5000000

clean:
	rm -f $(TARGET)