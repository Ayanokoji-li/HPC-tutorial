CXX = nvcc
CXXFLAGS = -O2 -Xcompiler -fopenmp -std=c++11

TARGET = main

$(TARGET): $(TARGET).cu
	$(CXX) $(CXXFLAGS) -o $@ $^

.PHONY: run-default clean

run: $(TARGET)
	./$(TARGET) 20000 20000 10

clean:
	rm -f $(TARGET)