#ifndef __KI_FUNC_H_
#define __KI_FUNC_H_

//#if 1
#include "../cub/cub/thread/thread_load.cuh"
#include "comm.h"
#include <mp_device.cuh>

//using namespace cub;

#ifndef ACCESS_ONCE_COMM
#define ACCESS_ONCE_COMM(V)                          \
    (*(volatile typeof (V) *)&(V))
#endif

//const int large_number = 1<<10;
#define TOT_SCHEDS 128
#define TOT_TYPES 3

typedef struct sched_info {
  mp::sem32_t sema;
  unsigned int block;
  unsigned int done[TOT_TYPES];
} sched_info_t;

__device__ sched_info_t scheds[TOT_SCHEDS];

__global__ void scheds_init()
{
  int j = threadIdx.x;
  assert(gridDim.x == 1);
  assert(blockDim.x >= TOT_SCHEDS);
  if (j < TOT_SCHEDS) {
    scheds[j].sema.sem = 0;
    scheds[j].sema.value = 1;
    scheds[j].block = 0;
    for (int i = 0; i < TOT_TYPES; ++i)
      scheds[j].done[i] = 0;
  }
}

__device__ static inline unsigned int elect_block(sched_info &sched)
{
  unsigned int ret;
  const int n_grids = gridDim.x; // BUG: account for .y and .z
  __shared__ unsigned int block;
  if (0 == threadIdx.x) {
    // 1st guy gets 0
    block = atomicInc(&sched.block, n_grids);
  }
  __syncthreads();
  ret = block;
  return ret;
}

__device__ static inline unsigned int elect_one(sched_info &sched, int grid_size, int idx)
{
  unsigned int ret;
  __shared__ unsigned int last;
  assert(idx < TOT_TYPES);
  if (0 == threadIdx.x) {
    // 1st guy gets 0
    // ((old >= val) ? 0 : (old+1))
    last = atomicInc(&sched.done[idx], grid_size);

  }
  __syncthreads();
  ret = last;
  return ret;
}



__device__ void LoadForceBuffer_KI(
  ForceMsg *buf, int nCells, int *gpu_cells, 
  SimGpu sim, int *cell_indices,
  int blockId, int gridSize)
{
  int tid = blockId * blockDim.x + threadIdx.x;
  int iCell = tid / MAXATOMS;
  int iAtom = tid % MAXATOMS;

  if (iCell < nCells) {
    int iBox = gpu_cells[iCell];
    int ii = iBox * MAXATOMS + iAtom;

    if (iAtom < sim.boxes.nAtoms[iBox])
    {
      int nBuf = cell_indices[iCell] + iAtom;
      buf[nBuf].dfEmbed = sim.eam_pot.dfEmbed[ii];
    }
  }
/*
  int tid = blockId * blockDim.x + threadIdx.x;
  int iCell, iAtom, iBox, ii, nBuf;

  iCell = tid / MAXATOMS;
  iAtom = tid % MAXATOMS;

//  while(iCell < nCells)
  if(iCell < nCells)
  {
    iBox = gpu_cells[iCell];
    ii = iBox * MAXATOMS + iAtom;

    if (iAtom < sim.boxes.nAtoms[iBox])
    {
      nBuf = cell_indices[iCell] + iAtom;
      buf[nBuf].dfEmbed = sim.eam_pot.dfEmbed[ii];
    }

  //  tid += gridSize*blockDim.x;
  //  iCell = tid / MAXATOMS;
  //  iAtom = tid % MAXATOMS;
  }
*/
}


__device__ void UnloadForceBuffer_KI(
  ForceMsg *buf, int nCells, int *gpu_cells, 
  SimGpu sim, int *cell_indices, int blockId, int gridSize)
{

  int tid = blockId * blockDim.x + threadIdx.x;
  int iCell = tid / MAXATOMS;
  int iAtom = tid % MAXATOMS;

  if (iCell < nCells) {
    int iBox = gpu_cells[iCell];
    int ii = iBox * MAXATOMS + iAtom;

    if (iAtom < sim.boxes.nAtoms[iBox])
    {
      int nBuf = cell_indices[iCell] + iAtom;
      sim.eam_pot.dfEmbed[ii] = buf[nBuf].dfEmbed;
    }
  }
/*
  int tid = blockId * blockDim.x + threadIdx.x;
  int iCell, iAtom, iBox, ii, nBuf;

  iCell = tid / MAXATOMS;
  iAtom = tid % MAXATOMS;

  //while(iCell < nCells)
  if(iCell < nCells)
  {
    iBox = gpu_cells[iCell];
    ii = iBox * MAXATOMS + iAtom;

    if (iAtom < sim.boxes.nAtoms[iBox])
    {
      nBuf = cell_indices[iCell] + iAtom;
      sim.eam_pot.dfEmbed[ii] = buf[nBuf].dfEmbed;
    }

    //tid += gridSize*blockDim.x;
    //iCell = tid / MAXATOMS;
    //iAtom = tid % MAXATOMS;
  }
*/
}


 /*
        ForceMsg *buf, int nCells, int *gpu_cells, 
        SimGpu sim, int *cell_indices,
        int blockId, int gridSize)
      
      int tid = block * blockDim.x + threadIdx.x;
      int iCell = tid / MAXATOMS;
      int iAtom = tid % MAXATOMS;

      if (iCell < nCellsM) {
        int iBox = sendCellListM[iCell];
        int ii = iBox * MAXATOMS + iAtom;

        if (iAtom < sGpu.boxes.nAtoms[iBox])
        {
          int nBuf = natoms_buf_sendM[iCell] + iAtom;
          ((ForceMsg*)sendBufM)[nBuf].dfEmbed = sGpu.eam_pot.dfEmbed[ii];
        }
      }
      */

__global__ void exchangeData_Force_KI(
  char *sendBufM_h, char *sendBufP_h,
  char *sendBufM_d, char *sendBufP_d,
  int sendSizeM, int sendSizeP,
  char *recvBufM, char *recvBufP, 
  int nCellsM, int nCellsP, 
  int *sendCellListM, int *sendCellListP, int *recvCellListM, int *recvCellListP,
  SimGpu sGpu, 
  int *natoms_buf_sendM, int *natoms_buf_sendP, int *natoms_buf_recvM, int *natoms_buf_recvP,
  int grid0, int grid1, int sched_id, struct comm_dev_descs *pdescs)
{
  assert(sched_id >= 0 && sched_id < TOT_SCHEDS);
  assert(gridDim.x >= grid0+grid1+1);

  sched_info_t &sched = scheds[sched_id];
  int block = elect_block(sched);
  int tid_local = threadIdx.x;

  //First block wait
  if(block == 0)
  {
    assert(blockDim.x >= pdescs->n_wait);
    
    if (threadIdx.x < pdescs->n_wait) {
      //printf("WAIT sched_id=%d, block=%d blockIdx.x=%d threadIdx.x=%d, pdescs->n_wait=%d\n", sched_id, block, blockIdx.x, threadIdx.x, pdescs->n_wait);
      mp::device::mlx5::wait(pdescs->wait[threadIdx.x]);
      mp::device::mlx5::signal(pdescs->wait[threadIdx.x]);
    }
    
    __syncthreads();

    if (0 == threadIdx.x) {
      // signal other blocks
      ACCESS_ONCE_COMM(sched.done[2]) = 1;   
    }
  }
  else
  {
    block--;
    if (block < grid0)
    {
      LoadForceBuffer_KI((ForceMsg*)sendBufM_d, nCellsM, sendCellListM, sGpu, natoms_buf_sendM, block, grid0);

      // elect last block to wait
      int last_block = elect_one(sched, grid0, 0); //__syncthreads(); inside
      if (0 == threadIdx.x)
          __threadfence();

      if (last_block == grid0-1)
      {
        while(1)
        {
          sendBufM_h[tid_local] = sendBufM_d[tid_local];
          tid_local += blockDim.x;
          if(tid_local >= sendSizeM) break;
        }
        __syncthreads();

        if(threadIdx.x == 0)
          mp::device::mlx5::send(pdescs->tx[threadIdx.x]);
      }
    }
    else
    {
      block -= grid0;
      if (block < grid0)
      {
        LoadForceBuffer_KI((ForceMsg*)sendBufP_d, nCellsP, sendCellListP, sGpu, natoms_buf_sendP, block, grid0);
/*
        int tid = block * blockDim.x + threadIdx.x;
        int iCell = tid / MAXATOMS;
        int iAtom = tid % MAXATOMS;

        if (iCell < nCellsM) {
          int iBox = sendCellListP[iCell];
          int ii = iBox * MAXATOMS + iAtom;

          if (iAtom < sGpu.boxes.nAtoms[iBox])
          {
            int nBuf = natoms_buf_sendP[iCell] + iAtom;
            ((ForceMsg*)sendBufP)[nBuf].dfEmbed = sGpu.eam_pot.dfEmbed[ii];
          }
        }
*/
        // elect last block to wait
        int last_block = elect_one(sched, grid0, 1); //__syncthreads(); inside
        if (0 == threadIdx.x)
            __threadfence();

        if (last_block == grid0-1)
        {
          while(1)
          {
            sendBufP_h[tid_local] = sendBufP_d[tid_local];
            tid_local += blockDim.x;
            if(tid_local >= sendSizeP) break;
          }
          __syncthreads();

          if(threadIdx.x == 0)
            mp::device::mlx5::send(pdescs->tx[threadIdx.x]);
        }
      }
      else 
      {
        // use other blocks to wait and unpack
        block -= grid0;
        if (0 <= block && block < grid1) {

          if (0 == threadIdx.x)
            while (cub::ThreadLoad<cub::LOAD_CG>(&sched.done[2]) < 1); // { __threadfence_block(); }

          __syncthreads();

          // execute sub-task
          UnloadForceBuffer_KI((ForceMsg*)recvBufP, nCellsP, recvCellListP, sGpu, natoms_buf_recvP, block, grid1);
          UnloadForceBuffer_KI((ForceMsg*)recvBufM, nCellsM, recvCellListM, sGpu, natoms_buf_recvM, block, grid1);
        }
      }

    }

  }
}

#if 0



else
  {
    block--;
    if (block < grid0)
    {
      LoadForceBuffer_KI((ForceMsg*)sendBufM, nCellsM, sendCellListM, sGpu, natoms_buf_sendM, block, grid0);

      // elect last block to wait
      int last_block = elect_one(sched, grid0, 0); //__syncthreads(); inside
      if (0 == threadIdx.x)
          __threadfence();

      if (last_block == grid0-1) 
      {
/*        if(
          (typeSend == 0 && threadIdx.x < pdescs->n_tx) ||
          (typeSend == 1 && threadIdx.x == (pdescs->n_tx-1))
        )
  */  
        if(threadIdx.x == 0)
          mp::device::mlx5::send(pdescs->tx[threadIdx.x]);
      }

      LoadForceBuffer_KI((ForceMsg*)sendBufP, nCellsP, sendCellListP, sGpu, natoms_buf_sendP, block, grid0);
      if (last_block == grid0-1) 
      {
        if(threadIdx.x == 1)
          mp::device::mlx5::send(pdescs->tx[threadIdx.x]);
      }
    }
    else 
    {
      // use other blocks to wait and unpack
      block -= grid0;
      if (0 <= block && block < grid1) {

        if (0 == threadIdx.x)
          while (cub::ThreadLoad<cub::LOAD_CG>(&sched.done[1]) < 1); // { __threadfence_block(); }

        __syncthreads();

        // execute sub-task
        UnloadForceBuffer_KI((ForceMsg*)recvBufP, nCellsP, recvCellListP, sGpu, natoms_buf_recvP, block, grid1);
        UnloadForceBuffer_KI((ForceMsg*)recvBufM, nCellsM, recvCellListM, sGpu, natoms_buf_recvM, block, grid1);
      }
    }

__global__ void localData_Force_KI(
  char *sendBufM, char *sendBufP, char *recvBufM, char *recvBufP, 
  int nCellsM, int nCellsP, 
  int *sendCellListM, int *sendCellListP, int *recvCellListM, int *recvCellListP,
  SimFlat *s, 
  int *natoms_buf_sendM, int *natoms_buf_sendP, int *natoms_buf_recvM, int *natoms_buf_recvP,
  int grid0, int grid1, int sched_id, struct comm_dev_descs *pdescs)
{
  assert(sched_id >= 0 && sched_id < TOT_SCHEDS);
  assert(gridDim.x >= max_grid01+grid2+1);

  sched_info_t &sched = scheds[sched_id];
  int block = elect_block(sched);
  
  if (block < grid0)
  {
    LoadForceBuffer_KI((ForceMsg*)recvBufP, nCellsM, sendCellListM, s->gpu, natoms_buf_sendM, block, grid0);
    LoadForceBuffer_KI((ForceMsg*)recvBufM, nCellsP, sendCellListP, s->gpu, natoms_buf_sendP, block, grid0);
    
    int last_block = elect_one(sched, grid0, 0); //__syncthreads(); inside
    if (last_block == grid0-1) 
    {
      if (0 == threadIdx.x)
       ACCESS_ONCE_COMM(sched.done[0]) = 1;
      
      __syncthreads();
    }
  }
  else 
  {
    block -= grid0;
    if (0 <= block && block < grid1) {

      if (0 == threadIdx.x)
          while (cub::ThreadLoad<cub::LOAD_CG>(&sched.done[0]) < 1); // { __threadfence_block(); }

        __syncthreads();

      UnloadForceBuffer_KI((ForceMsg*)recvBufM, nCellsM, recvCellListM, s->gpu, natoms_buf_recvM,block, grid1);
      UnloadForceBuffer_KI((ForceMsg*)recvBufP, nCellsP, recvCellListP, s->gpu, natoms_buf_recvP, block, grid1);
    }
  }
}
#endif

#endif