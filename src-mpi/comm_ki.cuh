#ifndef __KI_FUNC_H_
#define __KI_FUNC_H_

//#if 1
#include <thread/thread_load.cuh>
#include "comm.h"
#include <mp/device.cuh>

//using namespace cub;

#ifndef ACCESS_ONCE_COMM
#define ACCESS_ONCE_COMM(V)                          \
    (*(volatile typeof (V) *)&(V))
#endif

//const int large_number = 1<<10;
#define TOT_SCHEDS 128
#define TOT_TYPES 3

typedef struct sched_info {
  //mp::sem32_t sema;
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
    //scheds[j].sema.sem = 0;
    //scheds[j].sema.value = 1;
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
  volatile char *sendBufM_h, volatile char *sendBufP_h,
  volatile char *sendBufM_d, volatile char *sendBufP_d,
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
      LoadForceBuffer_KI((ForceMsg*)sendBufM_h, nCellsM, sendCellListM, sGpu, natoms_buf_sendM, block, grid0);

      // elect last block to wait
      int last_block = elect_one(sched, grid0, 0); //__syncthreads(); inside
      if (0 == threadIdx.x)
          __threadfence();

      if (last_block == grid0-1)
      {
/*        
        if (0 == threadIdx.x)
          printf("lastblockM: %d, grid0: %d, blockIdx: %d blockDimx:%d\n", last_block, grid0, blockIdx.x, blockDim.x);
*/
/*
        //Why this is not working and the P buffer is working?
        int tid_local = threadIdx.x;
        while(1)
        {
          sendBufM_h[tid_local] = sendBufM_d[tid_local];
          tid_local += blockDim.x;
          if(tid_local >= sendSizeM) break;
        }
        __threadfence_block();
*/
        __syncthreads();
        //__threadfence_system();

        if(threadIdx.x == 0)
          mp::device::mlx5::send(pdescs->tx[threadIdx.x]);
      }
    }
    else
    {
      block -= grid0;
      if (block < grid0)
      {
        LoadForceBuffer_KI((ForceMsg*)sendBufP_h, nCellsP, sendCellListP, sGpu, natoms_buf_sendP, block, grid0);

        // elect last block to wait
        int last_block = elect_one(sched, grid0, 1); //__syncthreads(); inside
        if (0 == threadIdx.x)
            __threadfence();

        if (last_block == grid0-1)
        {
/*          
           if (0 == threadIdx.x)
            printf("lastblockP: %d, grid0: %d, blockIdx: %d blockDimx:%d\n",  last_block, grid0, blockIdx.x, blockDim.x);
*/
/*          
          int tid_local = threadIdx.x;
          while(1)
          {
            sendBufP_h[tid_local] = sendBufP_d[tid_local];
            tid_local += blockDim.x;
            if(tid_local >= sendSizeP) break;
          }
          __threadfence_block();
  */
          __syncthreads();

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


__device__ void LoadAtomsBufferPacked_KI(AtomMsgSoA compactAtoms, int *cellIDs, SimGpu sim_gpu, int *cellOffset, real_t shift_x, real_t shift_y, real_t shift_z)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int iCell, iAtom, ii, iBox, iBuf;

  iCell = tid / MAXATOMS;
  iAtom = tid % MAXATOMS;
  assert(iCell < sim_gpu.boxes.nLocalBoxes + 1 && iCell >= 0);

  iBox = cellIDs[iCell];
  ii = iBox * MAXATOMS + iAtom;

  if (iAtom < sim_gpu.boxes.nAtoms[iBox]) 
  {
    iBuf = cellOffset[iCell] + iAtom;

    // coalescing writes: structure of arrays
    compactAtoms.gid[iBuf] = sim_gpu.atoms.gid[ii];
    compactAtoms.type[iBuf] = sim_gpu.atoms.iSpecies[ii];
    compactAtoms.rx[iBuf] = sim_gpu.atoms.r.x[ii] + shift_x;
    compactAtoms.ry[iBuf] = sim_gpu.atoms.r.y[ii] + shift_y;
    compactAtoms.rz[iBuf] = sim_gpu.atoms.r.z[ii] + shift_z;
    compactAtoms.px[iBuf] = sim_gpu.atoms.p.x[ii];
    compactAtoms.py[iBuf] = sim_gpu.atoms.p.y[ii];
    compactAtoms.pz[iBuf] = sim_gpu.atoms.p.z[ii];
  }
}

__global__ void exchangeData_Atoms_KI(
    AtomMsgSoA compactAtomsM, AtomMsgSoA compactAtomsP, 
    int *d_cellListM, int *d_cellListP,
    int* d_cellOffsetsM, int* d_cellOffsetsP,
    real_t shiftM_x, real_t shiftM_y, real_t shiftM_z,
    real_t shiftP_x, real_t shiftP_y, real_t shiftP_z,
    int gridM, int gridP, SimGpu sim_gpu, int sched_id, struct comm_dev_descs *pdescs)
{
  assert(sched_id >= 0 && sched_id < TOT_SCHEDS);
  assert(gridDim.x >= gridM+gridP+1);

  sched_info_t &sched = scheds[sched_id];
  int block = elect_block(sched);
  
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
  }
  else
  {
    block--;
    if (block < gridM)
    {
      LoadAtomsBufferPacked_KI(compactAtomsM, d_cellListM, sim_gpu, d_cellOffsetsM, shiftM_x, shiftM_y, shiftM_z);

      // elect last block to wait
      int last_block = elect_one(sched, gridM, 0); //__syncthreads(); inside
      if (0 == threadIdx.x)
          __threadfence();

      if (last_block == gridM-1 && threadIdx.x == 0)
          mp::device::mlx5::send(pdescs->tx[threadIdx.x]);
    }
    else
    {
      block -= gridM;
      if (block < gridP)
      {
        LoadAtomsBufferPacked_KI(compactAtomsP, d_cellListP, sim_gpu, d_cellOffsetsP, shiftP_x, shiftP_y, shiftP_z);

        // elect last block to wait
        int last_block = elect_one(sched, gridP, 1); //__syncthreads(); inside
        if (0 == threadIdx.x)
            __threadfence();

        if (last_block == gridP-1 && threadIdx.x == 1)
            mp::device::mlx5::send(pdescs->tx[threadIdx.x]);
      }
    }
  }
}

#endif