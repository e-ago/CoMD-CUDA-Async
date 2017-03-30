#ifndef GPU_KERNELS_H
#define GPU_KERNELS_H

#include "CoMDTypes.h"
#include "gpu_types.h"
#include <cuda_runtime.h>


#ifdef __cplusplus
extern "C" {
#endif
	
void ljForceGpu(SimGpu * sim, int interpolation, int num_cells, int * cells_list,real_t plcutoff, int method);

void updateNeighborsGpu(SimGpu sim, int * temp);
void updateNeighborsGpuAsync(SimGpu sim, int * temp, int nCells, int * cellList, cudaStream_t stream);
void eamForce1Gpu(SimGpu sim, int method, int spline);
void eamForce2Gpu(SimGpu sim, int method, int spline);
void eamForce3Gpu(SimGpu sim, int method, int spline);

// latency hiding opt
void eamForce1GpuAsync(SimGpu sim, AtomListGpu atoms_list, int num_cells, int *cells_list, int method, cudaStream_t stream, int spline);
void eamForce2GpuAsync(SimGpu sim, AtomListGpu atoms_list, int num_cells, int *cells_list, int method, cudaStream_t stream, int spline);
void eamForce3GpuAsync(SimGpu sim, AtomListGpu atoms_list, int num_cells, int *cells_list, int method, cudaStream_t stream, int spline);


void emptyNeighborListGpu(SimGpu * sim, int boundaryFlag);

int compactCellsGpu(char* work_d, int nCells, int *d_cellList, SimGpu sim_gpu, int* d_cellOffsets, int * d_workScan, real3_old shift, cudaStream_t stream);

#ifdef USE_ASYNC
int loadAtomsBufferFromGpu_Async(char * sendBuf, int * sendSize,
                          char* d_compactAtoms, int nCells, 
                          int *d_cellList, SimGpu sim_gpu, int* d_cellOffsets, 
                          int * d_workScan, real3_old shift, cudaStream_t stream, int type);

int loadAtomsBufferFromGpu_Comm(char * sendBuf, int * sendSize,
                          char* d_compactAtoms, int nCells,
                          int *d_cellList, SimGpu sim_gpu, int* d_cellOffsets,
                          int * d_workScan, real3_old shift, cudaStream_t stream, int type);

int unloadAtomsBufferToGpu_Async(char *buf, int sendSize, SimFlat *sim, char *gpu_buf, cudaStream_t stream, int typeMem); //cudaStream_t stream2,  cudaEvent_t * event_copy)
int unloadAtomsBufferToGpu_Comm(char *buf, int sendSize, SimFlat *sim, char *gpu_buf, cudaStream_t stream, int typeMem); //cudaStream_t stream2,  cudaEvent_t * event_copy)

void loadForceBufferFromGpu_Async(char *buf, int bufSize, int nCells, int *cellList, int *natoms_buf, int *partial_sums, SimFlat *s, char *gpu_buf, cudaStream_t stream);
void loadForceBufferFromGpu_Comm(char *buf, int bufSize, int nCells, int *cellList, int *natoms_buf, int *partial_sums, SimFlat *s, char *gpu_buf, cudaStream_t stream);

void unloadForceBufferToGpu_Async(char *buf, int bufSize, int nCells, int *cellList, int *natoms_buf, int *partial_sums, SimFlat *s, char *gpu_buf, cudaStream_t stream, int type);
void unloadForceBufferToGpu_Comm(char *buf, int bufSize, int nCells, int *cellList, int *natoms_buf, int *partial_sums, SimFlat *s, char *gpu_buf, cudaStream_t stream, int type);

void unloadForceScanCells(int nCells, int *cellList, int *natoms_buf, 
  int *partial_sums, SimFlat *s, cudaStream_t stream);

int neighborListUpdateRequiredGpu_Async(SimGpu* sim, HaloExchange* haloExchange, int num, int iAxis);

void exchangeDataForceGpu_KI(
  char *sendBufM_h, char *sendBufP_h, 
  char *sendBufM_d, char *sendBufP_d, 
  int sendSizeM, int sendSizeP,
  char *recvBufM, char *recvBufP, 
  int nCellsM, int nCellsP, 
  int *sendCellListM, int *sendCellListP, int *recvCellListM, int *recvCellListP,
  SimFlat *s, 
  int *natoms_buf_sendM, int *natoms_buf_sendP, int *natoms_buf_recvM, int *natoms_buf_recvP,
  int *partial_sums_sendM, int *partial_sums_sendP, int *partial_sums_recvM, int *partial_sums_recvP,
  cudaStream_t stream, int iAxis, int rankM, int rankP);

#endif

void unloadAtomsBufferToGpu(char *buf, int nBuf, SimFlat *s, char *gpu_buf, cudaStream_t stream);
void loadForceBufferFromGpu(char *buf, int *nbuf, int nCells, int *cellList, int *natoms_buf, int *partial_sums, SimFlat *s, char *gpu_buf, cudaStream_t stream);
void unloadForceBufferToGpu(char *buf, int nBuf, int nCells, int *cellList, int *natoms_buf, int *partial_sums, SimFlat *s, char *gpu_buf, cudaStream_t stream);

void initHashTableGpu(HashTableGpu* hashTable, int nMaxEntries);

void getAtomMsgSoAPtr(char* const buffer, AtomMsgSoA *atomMsg, int n);

void buildNeighborListGpu(SimGpu* sim, int method, int boundaryFlag); //TODO rename flag (REFACTORING)
int neighborListUpdateRequiredGpu(SimGpu* sim);
int pairlistUpdateRequiredGpu(SimGpu* sim);

// computes local potential and kinetic energies
void computeEnergy(SimFlat *sim, real_t *eLocal);

void advanceVelocityGpu(SimGpu sim, real_t dt);
void advancePositionGpu(SimGpu* sim, real_t dt);

void buildAtomListGpu(SimFlat *sim, cudaStream_t stream);
void updateLinkCellsGpu(SimFlat *sim);
void sortAtomsGpu(SimFlat *sim, cudaStream_t stream);

int neighborListUpdateRequiredGpu(SimGpu* sim);
void updateNeighborsGpuAsync(SimGpu sim, int *temp, int num_cells, int *cell_list, cudaStream_t stream);

void emptyHashTableGpu(HashTableGpu* hashTable);
#ifdef __cplusplus
}
#endif
#endif
