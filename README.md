# Distributed vec X matrix
Code for Final Project from COL334 (Intro to Parallel and Distributed Systems)

### Notes on Running
The code runs on any server where CUDA and MPICH has been installed. For more info, look at compile.sh. If you want to learn more about the inner workings of the code, please look at the attached report.

The entire code has been written as a giant blob in one file (something I did to make MPI and CUDA work easily) but it's all fairly readable and modular.


To run, first compile the code as ./compile.sh and then run as 

```
./run.sh <input_file> <output_file>
```

