CXX = mpicxx
CXXFLAGS = -O2 -std=c++11

main: main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

.PHONY: getAns run-default check clean

run-default: main
	mpirun -np 4 ./main

getAns: Ans.cpp
	$(CXX) $(CXXFLAGS) -fopenmp -o $@ $^
	./getAns

check: Check.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^
	./check

clean:
	rm -f main getAns check ans.bin pi.bin