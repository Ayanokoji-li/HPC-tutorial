CXX = nvcc
CXXFLAGS = -O2 -std=c++11

TARGET = main

$(TARGET): $(TARGET).cu
	$(CXX) $(CXXFLAGS) -o $@ $^

.PHONY: run-default clean

run: $(TARGET)
	./$(TARGET) 2000 1000

clean:
	rm -f $(TARGET)