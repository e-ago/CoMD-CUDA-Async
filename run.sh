#!/usr/bin/env bash

function run() {
    local A=$1
    local B=$2
    local C=$3
    local NP=$4
    shift 4
    local PAR=$@
    date
    (
        echo; echo; \

        OMPI_params="$OMPI_params --mca btl openib,self"
        OMPI_params="$OMPI_params --mca btl_openib_want_cuda_gdr 1"
        OMPI_params="$OMPI_params --mca btl_openib_warn_default_gid_prefix 0"
        OMPI_params="$OMPI_params --mca btl_openib_verbose 1"

        $MPI_HOME/bin/mpirun $OMPI_params \
        \
        -x ASYNC_USE_ASYNC=0 \
        -x ASYNC_ENABLE_DEBUG=0 \
   	-x COMM_USE_COMM=$A  -x COMM_USE_ASYNC=$B   -x COMM_USE_GPU_COMM=$C \
        \
        -x MP_ENABLE_DEBUG=0 \
        -x GDS_ENABLE_DEBUG=0 \
        -x ENABLE_DEBUG_MSG=0 \
        \
        -x MLX5_DEBUG_MASK=0 \
        -x MLX5_FREEZE_ON_ERROR_CQE=0 \
        \
        -x MP_DBREC_ON_GPU=0 \
        -x MP_RX_CQ_ON_GPU=0 \
        -x MP_TX_CQ_ON_GPU=0 \
        \
        -x MP_EVENT_ASYNC=0 \
        \
        -x GDS_DISABLE_WRITE64=0           \
        -x GDS_SIMULATE_WRITE64=0         \
        -x GDS_DISABLE_INLINECOPY=0       \
        -x GDS_DISABLE_WEAK_CONSISTENCY=0 \
        -x GDS_DISABLE_MEMBAR=0         \
         -x CUDA_PASCAL_FORCE_40_BIT=1 \
        --map-by node  -np $NP ./wrapper.sh ./bin/CoMD-cuda-mpi $PAR ) 2>&1 | tee -a run.log
        #nvprof -o nvprof-kernel.%q{OMPI_COMM_WORLD_RANK}.nvprof
    date
}

set -x

echo "CWD=$PWD"

#run 1 0 0 0 2 -e -x 16 -y 16 -z 16 -i 2 -m thread_atom_nl $@ &> out_2proc_2x_16384.txt
#run 1 0 0 0 16 -e -x 198 -y 99 -z 99 -i 16 -m thread_atom_nl $@ &> out_16proc_16x_big.txt
#run 0 0 1 1 2 -e -i 2 -j 1 -k 1 -x 40 -y 40 -z 40 &> out_2proc_2x_40.txt
#run 0 0 1 1 32 -e -i 4 -j 4 -k 2 -x 80 -y 80 -z 80 &> out_32proc_4x_4y_2z_80_comm.txt

#MPI test
run 0 0 0 2 -e -i 2 -j 1 -k 1 -x 80 -y 80 -z 80  &> out_2proc_80_mpi.txt
run 0 0 0 4 -e -i 2 -j 2 -k 1 -x 80 -y 80 -z 80  &> out_4proc_80_mpi.txt
#run 0 0 0 8 -e -i 2 -j 2 -k 2 -x 80 -y 80 -z 80  &> out_8proc_80_mpi.txt
#run 0 0 0 16 -e -i 4 -j 2 -k 2 -x 80 -y 80 -z 80  &> out_16proc_80_mpi.txt

#Comm Sync test
run 1 0 0 2 -e -i 2 -j 1 -k 1 -x 80 -y 80 -z 80 &> out_2proc_80_sync.txt
run 1 0 0 4 -e -i 2 -j 2 -k 1 -x 80 -y 80 -z 80 &> out_4proc_80_sync.txt
#run 1 0 0 8 -e -i 2 -j 2 -k 2 -x 80 -y 80 -z 80 &> out_8proc_80_sync.txt
#run 1 0 0 16 -e -i 4 -j 2 -k 2 -x 80 -y 80 -z 80 &> out_16proc_80_sync.txt

#Comm Async test
run 1 1 0 2 -e -i 2 -j 1 -k 1 -x 80 -y 80 -z 80 &> out_2proc_80_async.txt
run 1 1 0 4 -e -i 2 -j 2 -k 1 -x 80 -y 80 -z 80 &> out_4proc_80_async.txt
#run 1 1 0 8 -e -i 2 -j 2 -k 2 -x 80 -y 80 -z 80 &> out_8proc_80_async.txt
#run 1 1 0 16 -e -i 4 -j 2 -k 2 -x 80 -y 80 -z 80 &> out_16proc_80_async.txt

#Comm GPU test
run 1 1 1 2 -e -i 2 -j 1 -k 1 -x 80 -y 80 -z 80 &> out_2proc_80_gpu.txt
run 1 1 1 4 -e -i 2 -j 2 -k 1 -x 80 -y 80 -z 80 &> out_4proc_80_gpu.txt
#run 1 1 1 8 -e -i 2 -j 2 -k 2 -x 80 -y 80 -z 80 &> &> out_8proc_80_gpu.txt
#run 1 1 1 16 -e -i 4 -j 2 -k 2 -x 80 -y 80 -z 80 &> &> out_16proc_80_gpu.txt
