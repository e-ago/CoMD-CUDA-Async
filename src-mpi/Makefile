# Makefile for CUDA version

SHELL = /bin/bash

### Specify CUDA path here if necessary
CUDA_HOME ?= $(CUDA_PATH)

### CoMD can be built in either double or single precision and with or
### without MPI.  Select desired precision and MPI here.

# double precision (ON/OFF)
DOUBLE_PRECISION = ON
# MPI for parallel (ON/OFF)
DO_MPI = ON
# Maximum number of atoms that can be stored in a link cell
MAXATOMS = 64

# This enables assertions within device kernels as well as sanity-checks
# scattered within the C code. (ON/OFF)
DEBUG = OFF

### Set your desired C compiler and any necessary flags.  Note that CoMD
### uses some c99 features.  You can also set flags for optimization and
### specify paths to include files that the compiler can't find on its
### own.  If you need any -L or -l switches to get C standard libraries
### (such as -lm for the math library) put them in C_LIB.
CC = mpicc
CCX = mpic++
#ASYNC change ... was mpicc
CFLAGS = -std=gnu11 -Wno-unused-result -DMAXATOMS=$(MAXATOMS)
#-std=c99
#-std=c++11
OPTFLAGS =
# -05 
#-g -O5
INCLUDES =
C_LIB = -lm -lstdc++

### If you need to specify include paths, library paths, or link flags
### for MPI, put them here.  Put both -L and -l switches into MPI_LIB.

CUB_INCLUDE= -I$(PREFIX)/src/cub/cub
MPI_INCLUDE= -I$(MPI_HOME)/include
MPI_LIB= -L/lib64 -L/lib  
#DGX-1 required
#-I/usr/local/Cluster-Apps/openmpi/gnu/1.10.3/include/openmpi/opal/mca/hwloc/hwloc191/hwloc/include -I/usr/local/Cluster-Apps/openmpi/gnu/1.10.3/include/openmpi/opal/mca/event/libevent2021/libevent -I/usr/local/Cluster-Apps/openmpi/gnu/1.10.3/include/openmpi/opal/mca/event/libevent2021/libevent/include -I/usr/local/Cluster-Apps/openmpi/gnu/1.10.3/include -I/usr/local/Cluster-Apps/openmpi/gnu/1.10.3/include/openmpi
#use those includes shown by 'mpicc --show'

#ASYNC
MP_INCLUDE= -I$(PREFIX)/include ${MPI_INCLUDE} ${CUB_INCLUDE}
MP_LIB= -L$(PREFIX)/lib -lmp -lgdsync -lgdrapi -lcuda -libverbs

### A place to specify any other include or library switches your
### platform requires.
OTHER_LIB = -L$(CUDA_HOME)/lib64 -lcudart
OTHER_INCLUDE = -I$(CUDA_HOME)/include ${MPI_INCLUDE} ${CUB_INCLUDE}

### CUDA targets
# set up nvcc compiler path
NVCC = $(CUDA_HOME)/bin/nvcc

# nvcc-specific options
NVCC_FLAGS += -std=c++11 -DMAXATOMS=$(MAXATOMS)
NVCC_FLAGS += --ptxas-options=-v -lineinfo #-g -G
# add support for various cuda architectures below
#NVCC_FLAGS += -gencode arch=compute_20,code=sm_20
#NVCC_FLAGS += -gencode arch=compute_35,code=sm_35
#NVCC_FLAGS += -gencode arch=compute_52,code=sm_52
#DGX required
NVCC_FLAGS += -gencode arch=compute_60,code=sm_60

# add precision flag for CUDA targets
ifeq ($(DOUBLE_PRECISION), ON)
NVCC_FLAGS += -DCOMD_DOUBLE
else
NVCC_FLAGS += -DCOMD_SINGLE
endif

ifeq ($(DEBUG), ON)
CFLAGS += -DDEBUG -O0 -g
NVCC_FLAGS += -O0 -DDEBUG
else
# specify optimization flags
CFLAGS += -DNDEBUG ${OPTFLAGS}
NVCC_FLAGS += -O2 -DNDEBUG
endif

#########################################
### Below here, it is pitch black.  
### You are likely to be eaten by a grue.
##########################################

# clear all suffixes
.SUFFIXES:
# list only those that we use 
.SUFFIXES: .c .cc .cu .o

.PHONY: DEFAULT clean distclean depend

BIN_DIR=../bin

# Check for double precision
ifeq ($(DOUBLE_PRECISION), ON)
CFLAGS += -DCOMD_DOUBLE
else
CFLAGS += -DCOMD_SINGLE
endif

# Set executable name and add includes & libraries for MPI if needed.
CoMD_VARIANT = CoMD-cuda
ifeq ($(DO_MPI), ON)
CoMD_VARIANT = CoMD-cuda-mpi
INCLUDES += ${MPI_INCLUDE} ${MP_INCLUDE} ${CUB_INCLUDE}
CFLAGS += -DDO_MPI -DUSE_ASYNC ${MP_INCLUDE} ${CUB_INCLUDE}
CPPFLAGS += -DDO_MPI -DUSE_ASYNC ${MP_INCLUDE} ${CUB_INCLUDE}

LDFLAGS += ${MPI_LIB} ${MP_LIB} -lnvToolsExt
NVCC_FLAGS += ${MPI_INCLUDE} ${MP_INCLUDE} ${CUB_INCLUDE} -DDO_MPI -DUSE_ASYNC -lineinfo
endif
CoMD_EXE = ${BIN_DIR}/${CoMD_VARIANT}

LDFLAGS += ${C_LIB} ${OTHER_LIB}
CFLAGS  += ${INCLUDES} ${OTHER_INCLUDE} -D_NVPROF_NVTX -DUSE_ASYNC
CPPFLAGS += ${INCLUDES} ${OTHER_INCLUDE} -D_NVPROF_NVTX -DUSE_ASYNC
# c files to be compiled with mpicc
SOURCES=$(wildcard *.c)
OBJECTS=$(SOURCES:.c=.o)

SOURCESCC=$(wildcard *.cc)
OBJECTSCC=$(SOURCESCC:.cc=.cc.o)

# cuda files to be compiled with nvcc
CU_SOURCES=$(wildcard *.cu)
CU_OBJECTS=$(CU_SOURCES:.cu=.cu.o)

DEFAULT: ${CoMD_EXE}

%.cu.o: %.cu
	${NVCC} -c $< ${NVCC_FLAGS} -o $@

%.cc.o: %.cc
	${CC} -c $< ${CPPFLAGS} -o $@

%.o: %.c
	${CC} ${CFLAGS} -c $< -o $@

${CoMD_EXE}: ${BIN_DIR} CoMD_info.h ${OBJECTS} ${OBJECTSCC} ${CU_OBJECTS}
	${CCX} ${CFLAGS} -o ${CoMD_EXE} ${OBJECTS} ${OBJECTSCC} ${CU_OBJECTS} ${LDFLAGS}

CoMD_info.h: Makefile
	./generate_info_header ${CoMD_VARIANT} "$(CC)" "$(CFLAGS)" "$(LDFLAGS)"

${BIN_DIR}:
	@if [ ! -d ${BIN_DIR} ]; then mkdir -p ${BIN_DIR} ; fi

clean:
	rm -f *.o CoMD_info.h .depend

distclean: clean
	rm -f ${CoMD_EXE} .depend.bak
	rm -rf html latex

depend:
	touch .depend
	makedepend -f .depend -Y. --$(CFLAGS)-- $(SOURCES) 2> /dev/null
	makedepend -a -f .depend -Y. -o.cu.o --$(CFLAGS)-- $(CU_SOURCES) 2> /dev/null

-include .depend
