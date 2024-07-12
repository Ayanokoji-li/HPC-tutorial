# Requirement

Need to support MPI

# How To Use

You can modify `main.cpp` and use `make` to compile.

Program `main` should launch with args `<loop times> <num per rank>`

`make run-default` to run code with 4 processes with args `10 5000000`

If the answer of `Yours_Scan` is wrong, it will print Incorrect!

Otherwise it will print the average time after loop times of `Yours_Scan`, `Naive_Scan` and MPI implementation.

# Note

This practice uses [THU HPC exp2](https://lab.cs.tsinghua.edu.cn/hpc/doc/assignments/2.mpi_allreduce/) as reference. If any question, please email me.