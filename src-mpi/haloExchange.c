/// \file
/// Communicate halo data such as "ghost" atoms with neighboring tasks.
/// In addition to ghost atoms, the EAM potential also needs to exchange
/// some force information.  Hence this file implements both an atom
/// exchange and a force exchange, each with slightly different
/// properties due to their different roles.
/// 
/// The halo exchange in CoMD 1.1 takes advantage of the Cartesian domain
/// decomposition as well as the link cell structure to quickly
/// determine what data needs to be sent.  
///
/// This halo exchange implementation is able to send data to all 26
/// neighboring tasks using only 6 messages.  This is accomplished by
/// sending data across the x-faces, then the y-faces, and finally
/// across the z-faces.  Some of the data that was received from the
/// x-faces is included in the y-face sends and so on.  This
/// accumulation of data allows data to reach edge neighbors and corner
/// neighbors by a two or three step process.
///
/// The advantage of this type of structured halo exchange is that it
/// minimizes the number of MPI messages to send, and maximizes the size
/// of those messages.
///
/// The disadvantage of this halo exchange is that it serializes message
/// traffic.  Only two messages can be in flight at once. The x-axis
/// messages must be received and processed before the y-axis messages
/// can begin.  Architectures with low message latency and many off node
/// network links would likely benefit from alternate halo exchange
/// strategies that send independent messages to each neighbor task.

#include "haloExchange.h"

#include <assert.h>
#include <stdio.h>

#include "CoMDTypes.h"
#include "decomposition.h"
#include "parallel.h"
#include "defines.h"
#include "linkCells.h"
#include "hashTable.h"
#include "neighborList.h"
#include "eam.h"
#include "memUtils.h"
#include "performanceTimers.h"

#include "gpu_kernels.h"

#define MAX(A,B) ((A) > (B) ? (A) : (B))

/// Don't change the order of the faces in this enum.
enum HaloFaceOrder {HALO_X_MINUS, HALO_X_PLUS,
                    HALO_Y_MINUS, HALO_Y_PLUS,
                    HALO_Z_MINUS, HALO_Z_PLUS};

/// Don't change the order of the axes in this enum.
enum HaloAxisOrder {HALO_X_AXIS, HALO_Y_AXIS, HALO_Z_AXIS};

static HaloExchange* initHaloExchange(Domain* domain);
static void exchangeData(HaloExchange* haloExchange, void* data, int iAxis);

static int* mkAtomCellList(LinkCell* boxes, enum HaloFaceOrder iFace, const int nCells);
static int loadAtomsBuffer(void* vparms, void* data, int face, char* charBuf);
static void unloadAtomsBuffer(void* vparms, void* data, int face, int bufSize, char* charBuf);
static void destroyAtomsExchange(void* vparms);

static int* mkForceSendCellList(LinkCell* boxes, int face, int nCells);
static int* mkForceRecvCellList(LinkCell* boxes, int face, int nCells);
static int loadForceBuffer(void* vparms, void* data, int face, char* charBuf);
static int loadForceBufferCpu(void* vparms, void* data, int face, char* charBuf);
static void unloadForceBuffer(void* vparms, void* data, int face, int bufSize, char* charBuf);
static void unloadForceBufferCpu(void* vparms, void* data, int face, int bufSize, char* charBuf);
static void destroyForceExchange(void* vparms);
static int sortAtomsById(const void* a, const void* b);
static void exchangeData_Force_KI(HaloExchange* haloExchange, void* data, int iAxis, 
   comm_request_t * recv_requests, comm_request_t * send_requests, comm_request_t * ready_requests,
   int * sendSizeM, int * sendSizeP,  int * recvSizeM,  int * recvSizeP);

//ASYNC
static void exchangeData_Atom_Comm(HaloExchange* haloExchange, void* data, int iAxis, 
   comm_request_t * recv_requests, comm_request_t * send_requests, comm_request_t * ready_requests, int type);

static void exchangeData_Atom_Async(HaloExchange* haloExchange, void* data, int iAxis, 
   comm_request_t * recv_requests, comm_request_t * send_requests, comm_request_t * ready_requests, int type);

static void exchangeData_Force_Comm(HaloExchange* haloExchange, void* data, int iAxis, 
   comm_request_t * recv_requests,
   comm_request_t * send_requests,
   comm_request_t * ready_requests, int * sM, int* sP, int* rM, int* rP, int type);

static void exchangeData_Force_Async(HaloExchange* haloExchange, void* data, int iAxis, 
   comm_request_t * recv_requests,
   comm_request_t * send_requests,
   comm_request_t * ready_requests, int * sM, int* sP, int* rM, int* rP, int type);


/// \details
/// When called in proper sequence by redistributeAtoms, the atom halo
/// exchange helps serve three purposes:
/// - Send ghost atom data to neighbor tasks.
/// - Shift atom coordinates by the global simulation size when they cross
///   periodic boundaries.  This shift is performed in loadAtomsBuffer.
/// - Transfer ownership of atoms between tasks as the atoms move across
///   spatial domain boundaries.  This transfer of ownership occurs in
///   two places.  The former owner gives up ownership when
///   updateLinkCells moves a formerly local atom into a halo link cell.
///   The new owner accepts ownership when unloadAtomsBuffer calls
///   putAtomInBox to place a received atom into a local link cell.
///
/// This constructor does the following:
///
/// - Sets the bufCapacity to hold the largest possible number of atoms
///   that can be sent across a face.
/// - Initialize function pointers to the atom-specific versions
/// - Sets the number of link cells to send across each face.
/// - Builds the list of link cells to send across each face.  As
///   explained in the comments for mkAtomCellList, this list must
///   include any link cell, local or halo, that could possibly contain
///   an atom that needs to be sent across the face.  Atoms that need to
///   be sent include "ghost atoms" that are located in local link
///   cells that correspond to halo link cells on receiving tasks as well as
///   formerly local atoms that have just moved into halo link cells and
///   need to be sent to the rank that owns the spatial domain the atom
///   has moved into.
/// - Sets a coordinate shift factor for each face to account for
///   periodic boundary conditions.  For most faces the factor is zero.
///   For faces on the +x, +y, or +z face of the simulation domain
///   the factor is -1.0 (to shift the coordinates by -1 times the
///   simulation domain size).  For -x, -y, and -z faces of the
///   simulation domain, the factor is +1.0.
///
/// \see redistributeAtoms

#ifndef CUDACHECK
#define __CUDACHECK(stmt, cond_str)         \
    do {                \
        cudaError_t result = (stmt);                                    \
        if (cudaSuccess != result) {                                    \
          fprintf(stderr, "[%d] [%d] Assertion \"%s != cudaSuccess\" failed at %s:%d error=%d(%s)\n", \
                  getpid(), mpi_comm_rank, cond_str, __FILE__, __LINE__, result, cudaGetErrorString(result)); \
          exit(EXIT_FAILURE);                                           \
        }                                                               \
    } while (0)

#define CUDACHECK(stmt) __CUDACHECK(stmt, #stmt)
#endif

#define SIZE_BYTES 0

HaloExchange* initAtomHaloExchange(Domain* domain, LinkCell* boxes)
{
   HaloExchange* hh = initHaloExchange(domain);
   
   int size0 = (boxes->gridSize[1]+2)*(boxes->gridSize[2]+2);
   int size1 = (boxes->gridSize[0]+2)*(boxes->gridSize[2]+2);
   int size2 = (boxes->gridSize[0]+2)*(boxes->gridSize[1]+2);
   int maxSize = MAX(size0, size1);
   maxSize = MAX(size1, size2);
   hh->bufCapacity = maxSize*2*MAXATOMS*sizeof(AtomMsg);
   
   hh->sendBufM = (char*)comdMalloc(hh->bufCapacity);
   hh->sendBufP = (char*)comdMalloc(hh->bufCapacity);
   hh->recvBufP = (char*)comdMalloc(hh->bufCapacity);
   hh->recvBufM = (char*)comdMalloc(hh->bufCapacity);

   // pin memory
   cudaHostRegister(hh->sendBufM, hh->bufCapacity, 0);
   cudaHostRegister(hh->sendBufP, hh->bufCapacity, 0);
   cudaHostRegister(hh->recvBufP, hh->bufCapacity, 0);
   cudaHostRegister(hh->recvBufM, hh->bufCapacity, 0);
   
   // -----------------------------
   if(comm_use_comm())
   {
      int sendSize =  SIZE_BYTES+hh->bufCapacity;
      int recvSize =  SIZE_BYTES+hh->bufCapacity;

      //printf("----> ATOMS. Recv Size: %d Send Size: %d\n", recvSize, sendSize);
      
      hh->sendBufM_Async = (char**) calloc(3, sizeof(char*));
      CUDACHECK(cudaMallocHost((void**)&(hh->sendBufM_Async[0]), sendSize));
      CUDACHECK(cudaMallocHost((void**)&(hh->sendBufM_Async[1]), sendSize));
      CUDACHECK(cudaMallocHost((void**)&(hh->sendBufM_Async[2]), sendSize));

      hh->sendBufP_Async = (char**)calloc(3, sizeof(char*));
      CUDACHECK(cudaMallocHost((void**)&(hh->sendBufP_Async[0]), sendSize));
      CUDACHECK(cudaMallocHost((void**)&(hh->sendBufP_Async[1]), sendSize));
      CUDACHECK(cudaMallocHost((void**)&(hh->sendBufP_Async[2]), sendSize));

      hh->recvBufM_Async = (char**)calloc(3, sizeof(char*));
      CUDACHECK(cudaMallocHost((void**)&(hh->recvBufM_Async[0]), recvSize));
      CUDACHECK(cudaMallocHost((void**)&(hh->recvBufM_Async[1]), recvSize));
      CUDACHECK(cudaMallocHost((void**)&(hh->recvBufM_Async[2]), recvSize));

      hh->recvBufP_Async = (char**)calloc(3, sizeof(char*));
      CUDACHECK(cudaMallocHost((void**)&(hh->recvBufP_Async[0]), recvSize));
      CUDACHECK(cudaMallocHost((void**)&(hh->recvBufP_Async[1]), recvSize));
      CUDACHECK(cudaMallocHost((void**)&(hh->recvBufP_Async[2]), recvSize));

      hh->d_recvBufM_Async = (char**)calloc(3, sizeof(char*));
      CUDACHECK(cudaMalloc((void**)&(hh->d_recvBufM_Async[0]), recvSize));
      CUDACHECK(cudaMalloc((void**)&(hh->d_recvBufM_Async[1]), recvSize));
      CUDACHECK(cudaMalloc((void**)&(hh->d_recvBufM_Async[2]), recvSize));
     
      hh->d_recvBufP_Async = (char**)calloc(3, sizeof(char*));
      CUDACHECK(cudaMalloc((void**)&(hh->d_recvBufP_Async[0]), recvSize));
      CUDACHECK(cudaMalloc((void**)&(hh->d_recvBufP_Async[1]), recvSize));
      CUDACHECK(cudaMalloc((void**)&(hh->d_recvBufP_Async[2]), recvSize));
     
      hh->d_sendBufM_Async = (char**)calloc(3, sizeof(char*));
      CUDACHECK(cudaMalloc((void**)&(hh->d_sendBufM_Async[0]), sendSize));
      CUDACHECK(cudaMalloc((void**)&(hh->d_sendBufM_Async[1]), sendSize));
      CUDACHECK(cudaMalloc((void**)&(hh->d_sendBufM_Async[2]), sendSize));

      hh->d_sendBufP_Async = (char**)calloc(3, sizeof(char*));
      CUDACHECK(cudaMalloc((void**)&(hh->d_sendBufP_Async[0]), sendSize));
      CUDACHECK(cudaMalloc((void**)&(hh->d_sendBufP_Async[1]), sendSize));
      CUDACHECK(cudaMalloc((void**)&(hh->d_sendBufP_Async[2]), sendSize));

      hh->regSendM = (comm_reg_t*)calloc(3, /* num recvs */3*sizeof(comm_reg_t));
      hh->regSendP = (comm_reg_t*)calloc(3, /* num recvs */3*sizeof(comm_reg_t));
      hh->regRecvM = (comm_reg_t*)calloc(3, /* num recvs */3*sizeof(comm_reg_t));
      hh->regRecvP = (comm_reg_t*)calloc(3, /* num recvs */3*sizeof(comm_reg_t));   
   }

// -----------------------------

   hh->loadBuffer = loadAtomsBuffer;
   hh->unloadBuffer = unloadAtomsBuffer;
   hh->destroy = destroyAtomsExchange;

   hh->hashTable = initHashTable((boxes->nTotalBoxes - boxes->nLocalBoxes) * MAXATOMS * 2);

   AtomExchangeParms* parms = (AtomExchangeParms*)comdMalloc(sizeof(AtomExchangeParms));

   parms->nCells[HALO_X_MINUS] = 2*(boxes->gridSize[1]+2)*(boxes->gridSize[2]+2);
   parms->nCells[HALO_Y_MINUS] = 2*(boxes->gridSize[0]+2)*(boxes->gridSize[2]+2);
   parms->nCells[HALO_Z_MINUS] = 2*(boxes->gridSize[0]+2)*(boxes->gridSize[1]+2);
   parms->nCells[HALO_X_PLUS]  = parms->nCells[HALO_X_MINUS];
   parms->nCells[HALO_Y_PLUS]  = parms->nCells[HALO_Y_MINUS];
   parms->nCells[HALO_Z_PLUS]  = parms->nCells[HALO_Z_MINUS];

   for (int ii=0; ii<6; ++ii) {
      parms->cellList[ii] = mkAtomCellList(boxes, (enum HaloFaceOrder)ii, parms->nCells[ii]);
	  
      // copy cell list to gpu
      cudaMalloc((void**)&parms->cellListGpu[ii], parms->nCells[ii] * sizeof(int));
      cudaMemcpy(parms->cellListGpu[ii], parms->cellList[ii], parms->nCells[ii] * sizeof(int), cudaMemcpyHostToDevice);
  
   }
   // allocate scan buf
   int size = boxes->nLocalBoxes+1;
   if (size % 256 != 0) size = ((size + 255)/256)*256;

   int partial_size = size/256 + 1;
   if (partial_size % 256 != 0) partial_size = ((partial_size + 255)/256)*256;

   cudaMalloc((void**)&parms->d_natoms_buf, size * sizeof(int));
   parms->h_natoms_buf = (int*) malloc( size * sizeof(int));
   cudaMalloc((void**)&parms->d_partial_sums, partial_size * sizeof(int));

   for (int ii=0; ii<6; ++ii)
   {
      parms->pbcFactor[ii] = (real_t*)comdMalloc(3*sizeof(real_t));
      for (int jj=0; jj<3; ++jj)
         parms->pbcFactor[ii][jj] = 0.0;
   }
   int* procCoord = domain->procCoord; //alias
   int* procGrid  = domain->procGrid; //alias
   if (procCoord[HALO_X_AXIS] == 0)                       parms->pbcFactor[HALO_X_MINUS][HALO_X_AXIS] = +1.0;
   if (procCoord[HALO_X_AXIS] == procGrid[HALO_X_AXIS]-1) parms->pbcFactor[HALO_X_PLUS][HALO_X_AXIS]  = -1.0;
   if (procCoord[HALO_Y_AXIS] == 0)                       parms->pbcFactor[HALO_Y_MINUS][HALO_Y_AXIS] = +1.0;
   if (procCoord[HALO_Y_AXIS] == procGrid[HALO_Y_AXIS]-1) parms->pbcFactor[HALO_Y_PLUS][HALO_Y_AXIS]  = -1.0;
   if (procCoord[HALO_Z_AXIS] == 0)                       parms->pbcFactor[HALO_Z_MINUS][HALO_Z_AXIS] = +1.0;
   if (procCoord[HALO_Z_AXIS] == procGrid[HALO_Z_AXIS]-1) parms->pbcFactor[HALO_Z_PLUS][HALO_Z_AXIS]  = -1.0;
   
   hh->type = 0;
   hh->parms = parms;
   return hh;
}

/// The force exchange is considerably simpler than the atom exchange.
/// In the force case we only need to exchange data that is needed to
/// complete the force calculation.  Since the atoms have not moved we
/// only need to send data from local link cells and we are guaranteed
/// that the same atoms exist in the same order in corresponding halo
/// cells on remote tasks.  The only tricky part is the size of the
/// plane of local cells that needs to be sent grows in each direction.
/// This is because the y-axis send must send some of the data that was
/// received from the x-axis send, and the z-axis must send some data
/// from the y-axis send.  This accumulation of data to send is
/// responsible for data reaching neighbor cells that share only edges
/// or corners.
///
/// \see eam.c for an explanation of the requirement to exchange
/// force data.
HaloExchange* initForceHaloExchange(Domain* domain, LinkCell* boxes, int useGPU)
{
   HaloExchange* hh = initHaloExchange(domain);

   if(useGPU){
      hh->loadBuffer = loadForceBuffer;
      hh->unloadBuffer = unloadForceBuffer;
   }else{
      hh->loadBuffer = loadForceBufferCpu;
      hh->unloadBuffer = unloadForceBufferCpu;
   }
   hh->destroy = destroyForceExchange;

   int size0 = (boxes->gridSize[1])*(boxes->gridSize[2]);
   int size1 = (boxes->gridSize[0]+2)*(boxes->gridSize[2]);
   int size2 = (boxes->gridSize[0]+2)*(boxes->gridSize[1]+2);
   int maxSize = MAX(size0, size1);
   maxSize = MAX(size1, size2);
   hh->bufCapacity = (maxSize)*MAXATOMS*sizeof(ForceMsg);
   hh->sendBufM = (char*)comdMalloc(hh->bufCapacity);
   hh->sendBufP = (char*)comdMalloc(hh->bufCapacity);
   hh->recvBufP = (char*)comdMalloc(hh->bufCapacity);
   hh->recvBufM = (char*)comdMalloc(hh->bufCapacity);
   
   // pin memory
   cudaHostRegister(hh->sendBufM, hh->bufCapacity, 0);
   cudaHostRegister(hh->sendBufP, hh->bufCapacity, 0);
   cudaHostRegister(hh->recvBufP, hh->bufCapacity, 0);
   cudaHostRegister(hh->recvBufM, hh->bufCapacity, 0);

// -----------------------------
   if(comm_use_comm())
   {
      int sendSize =  SIZE_BYTES+hh->bufCapacity;
      int recvSize =  SIZE_BYTES+hh->bufCapacity;

    //  printf("INIT: MyRANK: %d, sendSize: %d, recvSize: %d\n", getMyRank(), sendSize, recvSize);

      hh->sendBufM_Async = (char**) calloc(3, sizeof(char*));
      CUDACHECK(cudaMallocHost((void**)&(hh->sendBufM_Async[0]), sendSize));
      CUDACHECK(cudaMallocHost((void**)&(hh->sendBufM_Async[1]), sendSize));
      CUDACHECK(cudaMallocHost((void**)&(hh->sendBufM_Async[2]), sendSize));

      hh->sendBufP_Async = (char**)calloc(3, sizeof(char*));
      CUDACHECK(cudaMallocHost((void**)&(hh->sendBufP_Async[0]), sendSize));
      CUDACHECK(cudaMallocHost((void**)&(hh->sendBufP_Async[1]), sendSize));
      CUDACHECK(cudaMallocHost((void**)&(hh->sendBufP_Async[2]), sendSize));

      hh->recvBufM_Async = (char**)calloc(3, sizeof(char*));
      CUDACHECK(cudaMallocHost((void**)&(hh->recvBufM_Async[0]), recvSize));
      CUDACHECK(cudaMallocHost((void**)&(hh->recvBufM_Async[1]), recvSize));
      CUDACHECK(cudaMallocHost((void**)&(hh->recvBufM_Async[2]), recvSize));

      hh->recvBufP_Async = (char**)calloc(3, sizeof(char*));
      CUDACHECK(cudaMallocHost((void**)&(hh->recvBufP_Async[0]), recvSize));
      CUDACHECK(cudaMallocHost((void**)&(hh->recvBufP_Async[1]), recvSize));
      CUDACHECK(cudaMallocHost((void**)&(hh->recvBufP_Async[2]), recvSize));

      hh->d_recvBufM_Async = (char**)calloc(3, sizeof(char*));
      CUDACHECK(cudaMalloc((void**)&(hh->d_recvBufM_Async[0]), recvSize));
      CUDACHECK(cudaMalloc((void**)&(hh->d_recvBufM_Async[1]), recvSize));
      CUDACHECK(cudaMalloc((void**)&(hh->d_recvBufM_Async[2]), recvSize));
     
      hh->d_recvBufP_Async = (char**)calloc(3, sizeof(char*));
      CUDACHECK(cudaMalloc((void**)&(hh->d_recvBufP_Async[0]), recvSize));
      CUDACHECK(cudaMalloc((void**)&(hh->d_recvBufP_Async[1]), recvSize));
      CUDACHECK(cudaMalloc((void**)&(hh->d_recvBufP_Async[2]), recvSize));

      hh->d_sendBufM_Async = (char**)calloc(3, sizeof(char*));
      CUDACHECK(cudaMalloc((void**)&(hh->d_sendBufM_Async[0]), sendSize));
      CUDACHECK(cudaMalloc((void**)&(hh->d_sendBufM_Async[1]), sendSize));
      CUDACHECK(cudaMalloc((void**)&(hh->d_sendBufM_Async[2]), sendSize));

      hh->d_sendBufP_Async = (char**)calloc(3, sizeof(char*));
      CUDACHECK(cudaMalloc((void**)&(hh->d_sendBufP_Async[0]), sendSize));
      CUDACHECK(cudaMalloc((void**)&(hh->d_sendBufP_Async[1]), sendSize));
      CUDACHECK(cudaMalloc((void**)&(hh->d_sendBufP_Async[2]), sendSize));

      hh->regSendM = (comm_reg_t*)calloc(3, /* num recvs */3*sizeof(comm_reg_t));
      hh->regSendP = (comm_reg_t*)calloc(3, /* num recvs */3*sizeof(comm_reg_t));
      hh->regRecvM = (comm_reg_t*)calloc(3, /* num recvs */3*sizeof(comm_reg_t));
      hh->regRecvP = (comm_reg_t*)calloc(3, /* num recvs */3*sizeof(comm_reg_t));         
   }

// -----------------------------

   ForceExchangeParms* parms = (ForceExchangeParms*)comdMalloc(sizeof(ForceExchangeParms));

   parms->nCells[HALO_X_MINUS] = (boxes->gridSize[1]  )*(boxes->gridSize[2]  );
   parms->nCells[HALO_Y_MINUS] = (boxes->gridSize[0]+2)*(boxes->gridSize[2]  );
   parms->nCells[HALO_Z_MINUS] = (boxes->gridSize[0]+2)*(boxes->gridSize[1]+2);
   parms->nCells[HALO_X_PLUS]  = parms->nCells[HALO_X_MINUS];
   parms->nCells[HALO_Y_PLUS]  = parms->nCells[HALO_Y_MINUS];
   parms->nCells[HALO_Z_PLUS]  = parms->nCells[HALO_Z_MINUS];

   for (int ii=0; ii<6; ++ii)
   {
      parms->sendCells[ii] = mkForceSendCellList(boxes, ii, parms->nCells[ii]);
      parms->recvCells[ii] = mkForceRecvCellList(boxes, ii, parms->nCells[ii]);

      // copy cell list to gpu
      cudaMalloc((void**)&parms->sendCellsGpu[ii], parms->nCells[ii] * sizeof(int));
      cudaMalloc((void**)&parms->recvCellsGpu[ii], parms->nCells[ii] * sizeof(int));
      cudaMemcpy(parms->sendCellsGpu[ii], parms->sendCells[ii], parms->nCells[ii] * sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(parms->recvCellsGpu[ii], parms->recvCells[ii], parms->nCells[ii] * sizeof(int), cudaMemcpyHostToDevice);

      // allocate temp buf
      int size = parms->nCells[ii]+1;
      if (size % 256 != 0) size = ((size + 255)/256)*256;
      cudaMalloc((void**)&parms->natoms_buf[ii], size * sizeof(int));
      cudaMalloc((void**)&parms->partial_sums[ii], size * sizeof(int));
//      cudaMalloc((void**)&parms->partial_sums[ii], (size/256 + 1) * sizeof(int)); ---> bug for scan!!!

       //elenago
      //printf("RANK[%d], cudaMalloc ii: %d, size: %d\n", getMyRank(), ii, size);
      cudaMalloc((void**)&parms->natoms_buf_send[ii], size * sizeof(int));
      cudaMalloc((void**)&parms->natoms_buf_recv[ii], size * sizeof(int));

//      if(getMyRank() == 0)
//         printf("partial_sums: %d elems, ii: %d, size: %d\n", (size/256 + 1), ii, size);

      cudaMalloc((void**)&parms->partial_sums_send[ii], size * sizeof(int));
      cudaMalloc((void**)&parms->partial_sums_recv[ii], size * sizeof(int));
   }
   
   hh->hashTable = NULL;
   hh->type = 1;
   hh->parms = parms;

   return hh;
}

void destroyHaloExchange(HaloExchange** haloExchange)
{
   (*haloExchange)->destroy((*haloExchange)->parms);
   if((*haloExchange)->hashTable);
     destroyHashTable(&((*haloExchange)->hashTable));
   free((*haloExchange)->sendBufM);
   free((*haloExchange)->sendBufP);
   free((*haloExchange)->recvBufP);
   free((*haloExchange)->recvBufM);

   free(*haloExchange);
   *haloExchange = NULL;
}

static int sizeMsgM[3];
static int sizeMsgP[3];
static int sizeMsgIndex=0;
static int sizeMsgIndexP=0;
static int sizeMsgIndexM=0;
static int typeAtomExchange=0;

void haloExchange_comm(HaloExchange* haloExchange, void* data)
{
	char * recvBufP, * recvBufM;
	enum HaloFaceOrder faceM, faceP;
	int maxSendSize =  haloExchange->bufCapacity+SIZE_BYTES;
	int maxRecvSize =  haloExchange->bufCapacity+SIZE_BYTES;
	int nCellsM, nCellsP, nbrRankM, nbrRankP;
	SimFlat* sim = (SimFlat*) data;

	comm_request_t  recv_requests[6];
	comm_request_t  send_requests[6];
	comm_request_t  ready_requests[6];

	sizeMsgIndex=0;
	sizeMsgIndexP=0;
	sizeMsgIndexM=0;

	//type == 0 for atom-exchange
	if(haloExchange->type == 0)
	{
		PUSH_RANGE("setupAtom", 1);

		AtomExchangeParms* parms = (AtomExchangeParms*) haloExchange->parms;

		for (int iAxis=0; iAxis<3; ++iAxis)
		{
			faceM = (enum HaloFaceOrder)(2*iAxis);
			faceP = (enum HaloFaceOrder)(faceM+1);

			nCellsM = parms->nCells[faceM];
			nCellsP = parms->nCells[faceP];

			nbrRankM = haloExchange->nbrRank[faceM];
			nbrRankP = haloExchange->nbrRank[faceP];

			recvBufP = (char*)haloExchange->recvBufP_Async[iAxis];
			recvBufM = (char*)haloExchange->recvBufM_Async[iAxis];

			if(getMyRank() != nbrRankP)
			{
				comm_irecv(recvBufP,
			              maxRecvSize,
			              MPI_CHAR,
			              &(haloExchange->regRecvP[iAxis]),
			              nbrRankP,
			              &(recv_requests[(2*iAxis)]));

				if (comm_use_comm() && comm_use_async())
					comm_send_ready_on_stream(nbrRankP, &ready_requests[(2*iAxis)], sim->boundary_stream);
				else if (comm_use_comm())
					comm_send_ready(nbrRankP, &ready_requests[(2*iAxis)]);
			}

			if(getMyRank() != nbrRankM)
			{
				comm_irecv(recvBufM,
			              maxRecvSize,
			              MPI_CHAR,
			              &(haloExchange->regRecvM[iAxis]),
			              nbrRankM,
			              &(recv_requests[(2*iAxis)+1]));

				if (comm_use_comm() && comm_use_async())
					comm_send_ready_on_stream(nbrRankM, &ready_requests[(2*iAxis)+1], sim->boundary_stream);
				else if (comm_use_comm())
					comm_send_ready(nbrRankM, &ready_requests[(2*iAxis)+1]);
			}
		}
		POP_RANGE;

		if(comm_use_async())
		{
			for (int iAxis=0; iAxis<3; ++iAxis)
			{
				PUSH_RANGE("exchangeAtomAsync", 2);
				exchangeData_Atom_Async(haloExchange, data, iAxis, recv_requests+(2*iAxis),
										send_requests+(2*iAxis), ready_requests+(2*iAxis), 2);
				POP_RANGE;
				
			}
		}
		else if(comm_use_comm())
		{
			for (int iAxis=0; iAxis<3; ++iAxis)
			{
				PUSH_RANGE("exchangeAtomComm", 2);
				exchangeData_Atom_Comm(haloExchange, data, iAxis, recv_requests+(2*iAxis),
										send_requests+(2*iAxis), ready_requests+(2*iAxis), 2); 
				POP_RANGE;
			}
		}
	}
	else
	{
		ForceExchangeParms* parms = (ForceExchangeParms*) haloExchange->parms;
		int sendSizeM[3], sendSizeP[3],  recvSizeM[3],  recvSizeP[3]; 
		comm_dev_descs_t descs = NULL;

		// if (comm_use_comm() && comm_use_gpu_comm()) 
		// 	get_curr_descs_req();

		PUSH_RANGE("setupForce", 1);

		for (int iAxis=0; iAxis<3; ++iAxis)
		{
			faceM = (enum HaloFaceOrder)(2*iAxis);
			faceP = (enum HaloFaceOrder)(faceM+1);

			nCellsM = parms->nCells[faceM];
			nCellsP = parms->nCells[faceP];

			nbrRankM = haloExchange->nbrRank[faceM];
			nbrRankP = haloExchange->nbrRank[faceP];

			recvBufP = (char*)haloExchange->recvBufP_Async[iAxis];
			recvBufM = (char*)haloExchange->recvBufM_Async[iAxis];

			sendSizeM[iAxis] = sizeMsgM[sizeMsgIndexM]*sizeof(ForceMsg);
			sendSizeP[iAxis] = sizeMsgP[sizeMsgIndexP]*sizeof(ForceMsg);
			recvSizeM[iAxis] = sizeMsgM[sizeMsgIndexM]*sizeof(ForceMsg);
			recvSizeP[iAxis] = sizeMsgP[sizeMsgIndexP]*sizeof(ForceMsg);

			sizeMsgIndexP++;
			sizeMsgIndexM++;

			if(getMyRank() != nbrRankP)
			{
				comm_irecv(recvBufP,
					recvSizeP[iAxis],
					MPI_CHAR,
					&(haloExchange->regRecvP[iAxis]),
					nbrRankP,
					&(recv_requests[(2*iAxis)]));

				if (comm_use_comm() && comm_use_async())
					comm_send_ready_on_stream(nbrRankP, &ready_requests[(2*iAxis)], sim->boundary_stream);
				else if (comm_use_comm())
					comm_send_ready(nbrRankP, &ready_requests[(2*iAxis)]);

				// if (comm_use_comm() && comm_use_gpu_comm())
				// 	comm_prepare_wait_all(1, &(recv_requests[(2*iAxis)]));
			}

			if(getMyRank() != nbrRankM)
			{
				comm_irecv(recvBufM,
					recvSizeM[iAxis],
					MPI_CHAR,
					&(haloExchange->regRecvM[iAxis]),
					nbrRankM,
					&(recv_requests[(2*iAxis)+1]));

				if (comm_use_comm() && comm_use_async())
					comm_send_ready_on_stream(nbrRankM, &ready_requests[(2*iAxis)+1], sim->boundary_stream);
				else if (comm_use_comm())
					comm_send_ready(nbrRankM, &ready_requests[(2*iAxis)+1]);

				// if (comm_use_comm() && comm_use_gpu_comm())
				// 	comm_prepare_wait_all(1, &(recv_requests[(2*iAxis)+1]));
			}

			#if 0
			if (comm_use_comm() && comm_use_gpu_comm())
			{
				if(getMyRank() != nbrRankM)
				{
					comm_prepare_isend(haloExchange->sendBufM_Async[iAxis],
						sendSizeM[iAxis],
						MPI_CHAR,
						&(haloExchange->regSendM[iAxis]),
						nbrRankM,
						&(send_requests[2*iAxis]));
				}

				if(getMyRank() != nbrRankP)
				{
					comm_prepare_isend(haloExchange->sendBufP_Async[iAxis],
						sendSizeP[iAxis],
						MPI_CHAR,
						&(haloExchange->regSendP[iAxis]),
						nbrRankP,
						&(send_requests[(2*iAxis)+1]));
				}

				update_curr_descs_pointer();
			}
			#endif
		}

		POP_RANGE;

		for (int iAxis=0; iAxis<3; ++iAxis)
		{
			if (comm_use_comm() && comm_use_gpu_comm())
			{
				printf("ERROR: KI model not supported yet\n");
				// exchangeData_Force_KI(haloExchange, data, iAxis, 
				// 	recv_requests+(2*iAxis), send_requests+(2*iAxis), ready_requests+(2*iAxis),
				// 	sendSizeM, sendSizeP,  recvSizeM,  recvSizeP);
			}
			else if (comm_use_comm() && comm_use_async())
			{
				PUSH_RANGE("exchangeForceAsync", 2);
				
				exchangeData_Force_Async(haloExchange, data, iAxis, 
					recv_requests+(2*iAxis), send_requests+(2*iAxis), ready_requests+(2*iAxis),
					sendSizeM, sendSizeP,  recvSizeM,  recvSizeP, 2);
			
				POP_RANGE;
			}
			else if (comm_use_comm())
			{
				PUSH_RANGE("exchangeForceComm", 2);

				exchangeData_Force_Comm(haloExchange, data, iAxis, 
					recv_requests+(2*iAxis), send_requests+(2*iAxis), ready_requests+(2*iAxis),
					sendSizeM, sendSizeP,  recvSizeM,  recvSizeP, 2);
				
				POP_RANGE;
			}
		}

		if (comm_use_comm() && (comm_use_async() || comm_use_gpu_comm()) )
			comm_progress();
	}
}


void haloExchange_MPI(HaloExchange* haloExchange, void* data)
{
   for (int iAxis=0; iAxis<3; ++iAxis)
      exchangeData(haloExchange, data, iAxis);
}

void haloExchange(HaloExchange* haloExchange, void* data)
{  
	#ifdef COMMUNICATION_TIMERS
		cudaDeviceSynchronize();
	#endif

	startTimer(commHaloTimer);
	if(comm_use_comm())
	{
		haloExchange_comm(haloExchange, data);
		if(comm_use_comm())
			comm_flush();
	}
	else //MPI
		haloExchange_MPI(haloExchange, data);
	
	#ifdef COMMUNICATION_TIMERS
		cudaDeviceSynchronize();
	#endif
	
	stopTimer(commHaloTimer);
}

/// Base class constructor.
HaloExchange* initHaloExchange(Domain* domain)
{
   HaloExchange* hh = (HaloExchange*)comdMalloc(sizeof(HaloExchange));

   // Rank of neighbor task for each face.
   hh->nbrRank[HALO_X_MINUS] = processorNum(domain, -1,  0,  0);
   hh->nbrRank[HALO_X_PLUS]  = processorNum(domain, +1,  0,  0);
   hh->nbrRank[HALO_Y_MINUS] = processorNum(domain,  0, -1,  0);
   hh->nbrRank[HALO_Y_PLUS]  = processorNum(domain,  0, +1,  0);
   hh->nbrRank[HALO_Z_MINUS] = processorNum(domain,  0,  0, -1);
   hh->nbrRank[HALO_Z_PLUS]  = processorNum(domain,  0,  0, +1);
   hh->bufCapacity = 0; // will be set by sub-class.

   return hh;
}


#include <sys/time.h>

#define TIMER_DEF(n)     struct timeval temp_1_##n={0,0}, temp_2_##n={0,0}
#define TIMER_START(n)   gettimeofday(&temp_1_##n, (struct timezone*)0)
#define TIMER_STOP(n)    gettimeofday(&temp_2_##n, (struct timezone*)0)
#define TIMER_ELAPSED(n) ((temp_2_##n.tv_sec-temp_1_##n.tv_sec)*1.e6+(temp_2_##n.tv_usec-temp_1_##n.tv_usec))

//elenago
//#ifdef _NVPROF_NVTX

#include "nvToolsExt.h"

#define COMM_VERT    1
#define COMM_HORIZ  2
#define SCAN_COL    3
#define APPEND_ROWS 4
#define ALL_REDUCE  5
#define EX_SCAN      6
#define SEND      7
#define RECEIVE      8
#define OTHER     9


#define PUSH_RANGE(name,cid)                                                                 \
   do {                                                                                                  \
     const uint32_t colors[] = {                                                             \
            0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff, 0xff000000, 0xff0000ff, 0x55ff3300, 0xff660000, 0x66330000  \
      };                                                                                                 \
      const int num_colors = sizeof(colors)/sizeof(colors[0]);                \
      int color_id = cid%num_colors;                                                   \
    nvtxEventAttributes_t eventAttrib = {0};                                  \
    eventAttrib.version = NVTX_VERSION;                                             \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;                      \
    eventAttrib.colorType = NVTX_COLOR_ARGB;                                  \
    eventAttrib.color = colors[color_id];                                        \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;                     \
    eventAttrib.message.ascii = name;                                               \
    nvtxRangePushEx(&eventAttrib);                                                  \
   } while(0)

#define POP_RANGE do { nvtxRangePop(); } while(0)

/*
#else
#define PUSH_RANGE(name,cid) {}
#define POP_RANGE {}
#endif
*/

void exchangeData_Atom_Comm(
   HaloExchange* haloExchange, void* data, int iAxis, 
   comm_request_t * recv_requests, comm_request_t * send_requests, comm_request_t * ready_requests, int type)
{
   enum HaloFaceOrder faceM = (enum HaloFaceOrder)(2*iAxis);
   enum HaloFaceOrder faceP = (enum HaloFaceOrder)(faceM+1);

   int nbrRankM = haloExchange->nbrRank[faceM];
   int nbrRankP = haloExchange->nbrRank[faceP];

   char* sendBufM, * sendBufP, * recvBufP, * recvBufM;
   int typeM = 0, typeP = 0;

   sendBufM = (char*)haloExchange->sendBufM_Async[iAxis];      
   sendBufP = (char*)haloExchange->sendBufP_Async[iAxis];      

   if((getMyRank() == nbrRankM))
   {
      typeM=1;
      recvBufM = (char*)haloExchange->d_recvBufM_Async[iAxis];
   }
   else
      recvBufM = (char*)haloExchange->recvBufM_Async[iAxis];

   if((getMyRank() == nbrRankP))
   {
      typeP=1;
      recvBufP = (char*)haloExchange->d_recvBufP_Async[iAxis];
   }
   else
      recvBufP = (char*)haloExchange->recvBufP_Async[iAxis];

   int sendSize = 0; //SIZE_BYTES+haloExchange->bufCapacity;

   AtomExchangeParms* parms = (AtomExchangeParms*) haloExchange->parms;
   SimFlat* sim = (SimFlat*) data;
   real_t* pbcFactorM = parms->pbcFactor[faceM];
   real_t* pbcFactorP = parms->pbcFactor[faceP];

   real3_old shiftM;
   real3_old shiftP;

   shiftM[0] = pbcFactorM[0] * sim->domain->globalExtent[0];
   shiftM[1] = pbcFactorM[1] * sim->domain->globalExtent[1];
   shiftM[2] = pbcFactorM[2] * sim->domain->globalExtent[2];

   shiftP[0] = pbcFactorP[0] * sim->domain->globalExtent[0];
   shiftP[1] = pbcFactorP[1] * sim->domain->globalExtent[1];
   shiftP[2] = pbcFactorP[2] * sim->domain->globalExtent[2];

   int nCellsM = parms->nCells[faceM];
   int* cellListGpuM = parms->cellListGpu[faceM];

   int nCellsP = parms->nCells[faceP];
   int* cellListGpuP = parms->cellListGpu[faceP];

   if(getMyRank() == nbrRankM)
   {
      PUSH_RANGE("Atom M", 1);

      loadAtomsBufferFromGpu_Comm(recvBufP, &sizeMsgM[iAxis], sim->gpu_atoms_buf, nCellsM, 
                                 parms->cellListGpu[faceM], sim->gpu, parms->d_natoms_buf, 
                                 parms->d_partial_sums, shiftM, sim->boundary_stream, typeP);
      POP_RANGE;

      if(getMyRank() != nbrRankP)
         printf("Warning M! My rank: %d, RankM: %d RankP: %d\n", getMyRank(), nbrRankM, nbrRankP);
   }
   
   if(getMyRank() == nbrRankP)
   {
      PUSH_RANGE("Atom P", 2);

      loadAtomsBufferFromGpu_Comm(recvBufM, &sizeMsgP[iAxis], sim->gpu_atoms_buf, nCellsP, 
                                 parms->cellListGpu[faceP], sim->gpu, parms->d_natoms_buf, 
                                 parms->d_partial_sums, shiftP, sim->boundary_stream, typeM);

      POP_RANGE;

      if(getMyRank() != nbrRankM)
         printf("Warning P! My rank: %d, RankM: %d RankP: %d\n", getMyRank(), nbrRankM, nbrRankP);
   }

   if((getMyRank() != nbrRankM) && (getMyRank() != nbrRankP))
   {

      PUSH_RANGE("Atom Both", 3);

      loadAtomsBufferFromGpu_Comm(sendBufM, &sizeMsgM[iAxis], sim->gpu_atoms_buf, nCellsM, 
                                 parms->cellListGpu[faceM], sim->gpu, parms->d_natoms_buf, 
                                 parms->d_partial_sums, shiftM, sim->boundary_stream, typeM);

      comm_isend(sendBufM, 
                  (sizeMsgM[iAxis] * sizeof(AtomMsg)),
                  MPI_CHAR,
                  &(haloExchange->regSendM[iAxis]),
                  nbrRankM,
                  &send_requests[0]);

      //-------- COMPUTE & SEND faceP
      loadAtomsBufferFromGpu_Comm(sendBufP, &sizeMsgP[iAxis], sim->gpu_atoms_buf, nCellsP, 
                                 parms->cellListGpu[faceP], sim->gpu, parms->d_natoms_buf, 
                                 parms->d_partial_sums, shiftP, sim->boundary_stream, typeP);

      comm_isend(sendBufP,
                  (sizeMsgP[iAxis] * sizeof(AtomMsg)),
                  MPI_CHAR,
                  &(haloExchange->regSendP[iAxis]),
                  nbrRankP,
                  &send_requests[1]);

     //-------- Wait recv on stream
      comm_wait_all(2, recv_requests);
      comm_wait_all(2, send_requests);

      POP_RANGE;
   }

   //-------- Unload M
   unloadAtomsBufferToGpu_Comm(recvBufM, sizeMsgM[iAxis], sim, sim->gpu_atoms_buf, sim->boundary_stream, typeM); //haloExchange->stream_copy, , haloExchange->event_copy);     /*sizeMsgP[sizeMsgIndexP] = */ //sizeMsgIndexP++;
   //-------- Unload P
   unloadAtomsBufferToGpu_Comm(recvBufP, sizeMsgP[iAxis], sim, sim->gpu_atoms_buf, sim->boundary_stream, typeP); //haloExchange->stream_copy, typeP, haloExchange->event_copy);
}

void exchangeData_Atom_Async(
   HaloExchange* haloExchange, void* data, int iAxis, 
   comm_request_t * recv_requests, 
   comm_request_t * send_requests, 
   comm_request_t * ready_requests, int type)
{
	enum HaloFaceOrder faceM = (enum HaloFaceOrder)(2*iAxis);
	enum HaloFaceOrder faceP = (enum HaloFaceOrder)(faceM+1);

	int nbrRankM = haloExchange->nbrRank[faceM];
	int nbrRankP = haloExchange->nbrRank[faceP];

	char* sendBufM, * sendBufP, * recvBufP, * recvBufM;
	int typeM = 0, typeP = 0;

	sendBufM = (char*)haloExchange->sendBufM_Async[iAxis];      
	sendBufP = (char*)haloExchange->sendBufP_Async[iAxis];      

	if((getMyRank() == nbrRankM))
	{
		typeM=1;
		recvBufM = (char*)haloExchange->d_recvBufM_Async[iAxis];
	}
	else
		recvBufM = (char*)haloExchange->recvBufM_Async[iAxis];

	if((getMyRank() == nbrRankP))
	{
		typeP=1;
		recvBufP = (char*)haloExchange->d_recvBufP_Async[iAxis];
	}
	else
		recvBufP = (char*)haloExchange->recvBufP_Async[iAxis];

	int sendSize = 0; //SIZE_BYTES+haloExchange->bufCapacity;

	AtomExchangeParms* parms = (AtomExchangeParms*) haloExchange->parms;
	SimFlat* sim = (SimFlat*) data;
	real_t* pbcFactorM = parms->pbcFactor[faceM];
	real_t* pbcFactorP = parms->pbcFactor[faceP];

	real3_old shiftM;
	real3_old shiftP;

	shiftM[0] = pbcFactorM[0] * sim->domain->globalExtent[0];
	shiftM[1] = pbcFactorM[1] * sim->domain->globalExtent[1];
	shiftM[2] = pbcFactorM[2] * sim->domain->globalExtent[2];

	shiftP[0] = pbcFactorP[0] * sim->domain->globalExtent[0];
	shiftP[1] = pbcFactorP[1] * sim->domain->globalExtent[1];
	shiftP[2] = pbcFactorP[2] * sim->domain->globalExtent[2];

	int nCellsM = parms->nCells[faceM];
	int* cellListGpuM = parms->cellListGpu[faceM];

	int nCellsP = parms->nCells[faceP];
	int* cellListGpuP = parms->cellListGpu[faceP];

	if(getMyRank() == nbrRankM)
	{
		PUSH_RANGE("Atom M", 1);

		loadAtomsBufferFromGpu_Async(recvBufP, &sizeMsgM[iAxis], sim->gpu_atoms_buf, nCellsM, 
			parms->cellListGpu[faceM], sim->gpu, parms->d_natoms_buf, 
			parms->d_partial_sums, shiftM, sim->boundary_stream, typeP);

		POP_RANGE;

		if(getMyRank() != nbrRankP)
			printf("Warning M! My rank: %d, RankM: %d RankP: %d\n", getMyRank(), nbrRankM, nbrRankP);
	}

	if(getMyRank() == nbrRankP)
	{
		PUSH_RANGE("Atom P", 2);

		loadAtomsBufferFromGpu_Async(recvBufM, &sizeMsgP[iAxis], sim->gpu_atoms_buf, nCellsP, 
			parms->cellListGpu[faceP], sim->gpu, parms->d_natoms_buf, 
			parms->d_partial_sums, shiftP, sim->boundary_stream, typeM);

		POP_RANGE;

		if(getMyRank() != nbrRankM)
			printf("Warning P! My rank: %d, RankM: %d RankP: %d\n", getMyRank(), nbrRankM, nbrRankP);
	}

	if((getMyRank() != nbrRankM) && (getMyRank() != nbrRankP))
	{
		PUSH_RANGE("Atom Both", 3);

		loadAtomsBufferFromGpu_Async(sendBufM, &sizeMsgM[iAxis], sim->gpu_atoms_buf, nCellsM, 
			parms->cellListGpu[faceM], sim->gpu, parms->d_natoms_buf, 
			parms->d_partial_sums, shiftM, sim->boundary_stream, typeM);

		comm_wait_ready_on_stream(nbrRankM, sim->boundary_stream);
		comm_isend_on_stream(sendBufM, 
			(sizeMsgM[iAxis] * sizeof(AtomMsg)),
			MPI_CHAR,
			&(haloExchange->regSendM[iAxis]),
			nbrRankM,
			&send_requests[0],
			sim->boundary_stream);

		//-------- COMPUTE & SEND faceP
		loadAtomsBufferFromGpu_Async(sendBufP, &sizeMsgP[iAxis], sim->gpu_atoms_buf, nCellsP, 
			parms->cellListGpu[faceP], sim->gpu, parms->d_natoms_buf, 
			parms->d_partial_sums, shiftP, sim->boundary_stream, typeP);

		comm_wait_ready_on_stream(nbrRankP, sim->boundary_stream);
		comm_isend_on_stream(sendBufP, 
			(sizeMsgP[iAxis] * sizeof(AtomMsg)),
			MPI_CHAR,
			&(haloExchange->regSendP[iAxis]),
			nbrRankP,
			&send_requests[1],
			sim->boundary_stream);

		//-------- Wait recv on stream
		comm_wait_all_on_stream(2, recv_requests, sim->boundary_stream);

		POP_RANGE;
	}

	//-------- Unload M
	unloadAtomsBufferToGpu_Async(recvBufM, sizeMsgM[iAxis], sim, sim->gpu_atoms_buf, sim->boundary_stream, typeM); //haloExchange->stream_copy, , haloExchange->event_copy);     /*sizeMsgP[sizeMsgIndexP] = */ //sizeMsgIndexP++;
	//-------- Unload P
	unloadAtomsBufferToGpu_Async(recvBufP, sizeMsgP[iAxis], sim, sim->gpu_atoms_buf, sim->boundary_stream, typeP); //haloExchange->stream_copy, typeP, haloExchange->event_copy);

	if((getMyRank() != nbrRankM) && (getMyRank() != nbrRankP))
	{
		comm_wait_all_on_stream(2, send_requests, sim->boundary_stream);
		comm_wait_all_on_stream(2, ready_requests, sim->boundary_stream);
	}	
}

void exchangeData_Force_Comm(HaloExchange* haloExchange, void* data, int iAxis, 
   comm_request_t * recv_requests, comm_request_t * send_requests, comm_request_t * ready_requests,
   int * sendSizeM, int * sendSizeP,  int * recvSizeM,  int * recvSizeP, int typeFunc)
{
	enum HaloFaceOrder faceM = (enum HaloFaceOrder)(2*iAxis);
	enum HaloFaceOrder faceP = (enum HaloFaceOrder)(faceM+1);

	int nbrRankM = haloExchange->nbrRank[faceM];
	int nbrRankP = haloExchange->nbrRank[faceP];

	char* sendBufM, * sendBufP, * recvBufP, * recvBufM;
	sendBufM = (char*)haloExchange->sendBufM_Async[iAxis];
	sendBufP = (char*)haloExchange->sendBufP_Async[iAxis];
	recvBufM = (char*)haloExchange->recvBufM_Async[iAxis];
	recvBufP = (char*)haloExchange->recvBufP_Async[iAxis];

	int typeM=0, typeP=0;

	if((getMyRank() == nbrRankM))
	{
		typeP = 1;
		recvBufP = haloExchange->d_recvBufP_Async[iAxis];
		// sendBufP = haloExchange->d_sendBufP_Async[iAxis];
	}
	if((getMyRank() == nbrRankP))
	{
		typeM=1;
		recvBufM = haloExchange->d_recvBufM_Async[iAxis];
		//sendBufM = haloExchange->d_sendBufM_Async[iAxis];
	}

	ForceExchangeParms* parms = (ForceExchangeParms*) haloExchange->parms;
	SimFlat* sim = (SimFlat*) data;

	int nCellsM = parms->nCells[faceM];
	int* cellListGpuM = parms->sendCellsGpu[faceM];

	int nCellsP = parms->nCells[faceP];
	int* cellListGpuP = parms->sendCellsGpu[faceP];

	//Only send
	if( typeFunc == 0 || typeFunc == 2)
	{
		if(getMyRank() ==  nbrRankM)
		{
			PUSH_RANGE("nbrRankM", 1);

			loadForceBufferFromGpu_Comm(recvBufP, sendSizeM[iAxis], nCellsM, parms->sendCellsGpu[faceM], 
				parms->natoms_buf_send[faceM], parms->partial_sums[faceM],
				sim, sim->gpu_force_buf, sim->boundary_stream);

			if(getMyRank() != nbrRankP)
				printf("Warning M! My rank: %d, RankM: %d RankP: %d\n", getMyRank(), nbrRankM, nbrRankP);

			POP_RANGE;
		}

		if((getMyRank() == nbrRankP))
		{
			PUSH_RANGE("nbrRankP", 2);

			if(getMyRank() != nbrRankM)
				printf("Warning P! My rank: %d, RankM: %d RankP: %d\n", getMyRank(), nbrRankM, nbrRankP);

			loadForceBufferFromGpu_Comm(recvBufM, sendSizeP[iAxis], nCellsP, parms->sendCellsGpu[faceP], 
				parms->natoms_buf_send[faceP], parms->partial_sums[faceP],
				sim, sim->gpu_force_buf, sim->boundary_stream);

			POP_RANGE;
		}

		if((getMyRank() != nbrRankM) && (getMyRank() != nbrRankP))
		{
			PUSH_RANGE("BOTH", 3);

			loadForceBufferFromGpu_Comm(sendBufM, sendSizeM[iAxis], nCellsM, parms->sendCellsGpu[faceM], 
				/* natoms_buf_send */
				parms->natoms_buf_send[faceM], parms->partial_sums[faceM],
				sim, sim->gpu_force_buf, sim->boundary_stream);

			comm_isend(sendBufM, 
				sendSizeM[iAxis],
				MPI_CHAR,
				&(haloExchange->regSendM[iAxis]),
				nbrRankM,
				&send_requests[0]);


			loadForceBufferFromGpu_Comm(sendBufP, sendSizeP[iAxis], nCellsP, parms->sendCellsGpu[faceP], 
				/* natoms_buf_send */
				parms->natoms_buf_send[faceP], parms->partial_sums[faceP],
				sim, sim->gpu_force_buf, sim->boundary_stream);


			//-------- COMPUTE & SEND faceP
			comm_isend(sendBufP, 
				sendSizeP[iAxis],
				MPI_CHAR,
				&(haloExchange->regSendP[iAxis]),
				nbrRankP,
				&send_requests[1]);
			POP_RANGE;
		}
	}
	if(typeFunc == 1 || typeFunc == 2)
	{
		unloadForceScanCells(nCellsM, parms->recvCellsGpu[faceM], 
			/* natoms_buf_recv */
			parms->natoms_buf_recv[faceM], parms->partial_sums[faceM], 
			sim, sim->boundary_stream);


		unloadForceScanCells(nCellsP, parms->recvCellsGpu[faceP], 
			/* natoms_buf_recv */
			parms->natoms_buf_recv[faceP], parms->partial_sums[faceP], 
			sim, sim->boundary_stream);
		
		PUSH_RANGE("UNLOAD", 4);
		
		//-------- Wait recv on stream
		if((getMyRank() != nbrRankM) && (getMyRank() != nbrRankP))
			comm_wait_all(2, recv_requests);

		//-------- Unload P
		unloadForceBufferToGpu_Comm(recvBufP, recvSizeP[iAxis], nCellsP, parms->recvCellsGpu[faceP], 
			/* natoms_buf_recv */
			parms->natoms_buf_recv[faceP], parms->partial_sums[faceP], 
			sim, sim->gpu_force_buf, sim->boundary_stream, typeP);

		//-------- Unload M
		unloadForceBufferToGpu_Comm(recvBufM, recvSizeM[iAxis], nCellsM, parms->recvCellsGpu[faceM], 
			/* natoms_buf_recv */
			parms->natoms_buf_recv[faceM], parms->partial_sums[faceM], 
			sim, sim->gpu_force_buf, sim->boundary_stream, typeM);   

		if((getMyRank() != nbrRankM) && (getMyRank() != nbrRankP))
			comm_wait_all(2, send_requests);

		POP_RANGE;
	}    
}


void exchangeData_Force_Async(HaloExchange* haloExchange, void* data, int iAxis, 
   comm_request_t * recv_requests, 
   comm_request_t * send_requests, 
   comm_request_t * ready_requests,
   int * sendSizeM, int * sendSizeP,  
   int * recvSizeM,  int * recvSizeP, int typeFunc)
{
	enum HaloFaceOrder faceM = (enum HaloFaceOrder)(2*iAxis);
	enum HaloFaceOrder faceP = (enum HaloFaceOrder)(faceM+1);

	int nbrRankM = haloExchange->nbrRank[faceM];
	int nbrRankP = haloExchange->nbrRank[faceP];

	char* sendBufM, * sendBufP, * recvBufP, * recvBufM;
	sendBufM = (char*)haloExchange->sendBufM_Async[iAxis];
	sendBufP = (char*)haloExchange->sendBufP_Async[iAxis];
	recvBufM = (char*)haloExchange->recvBufM_Async[iAxis];
	recvBufP = (char*)haloExchange->recvBufP_Async[iAxis];

	int typeM=0, typeP=0;

	if((getMyRank() == nbrRankM))
	{
		typeP = 1;
		recvBufP = haloExchange->d_recvBufP_Async[iAxis];
		// sendBufP = haloExchange->d_sendBufP_Async[iAxis];
	}
	if((getMyRank() == nbrRankP))
	{
		typeM=1;
		recvBufM = haloExchange->d_recvBufM_Async[iAxis];
		//sendBufM = haloExchange->d_sendBufM_Async[iAxis];
	}

	ForceExchangeParms* parms = (ForceExchangeParms*) haloExchange->parms;
	SimFlat* sim = (SimFlat*) data;

	int nCellsM = parms->nCells[faceM];
	int* cellListGpuM = parms->sendCellsGpu[faceM];

	int nCellsP = parms->nCells[faceP];
	int* cellListGpuP = parms->sendCellsGpu[faceP];

	//Only send
	if( typeFunc == 0 || typeFunc == 2)
	{
		if(getMyRank() ==  nbrRankM)
		{
			PUSH_RANGE("nbrRankM", 3);

			loadForceBufferFromGpu_Async(recvBufP, sendSizeM[iAxis], nCellsM, parms->sendCellsGpu[faceM], 
				parms->natoms_buf_send[faceM], parms->partial_sums[faceM],
				sim, sim->gpu_force_buf, sim->boundary_stream);

			if(getMyRank() != nbrRankP)
				printf("Warning M! My rank: %d, RankM: %d RankP: %d\n", getMyRank(), nbrRankM, nbrRankP);
				/*
				if((getMyRank() != nbrRankP))
				{
					loadForceBufferFromGpu_Async(
						sendBufP, sendSizeP[iAxis], nCellsP, cellListGpuP, 
						parms->natoms_buf_send[faceP], parms->partial_sums[faceP],
						sim, sim->gpu_force_buf, sim->boundary_stream);

					comm_isend_on_stream(sendBufP, 
						sendSizeP[iAxis],
						MPI_CHAR,
						&(haloExchange->regSendP[iAxis]),
						nbrRankP,
						&send_requests[1],
						sim->boundary_stream);
				}
				*/

			POP_RANGE;
		}

		if((getMyRank() == nbrRankP))
		{
			PUSH_RANGE("nbrRankP", 3);

			if(getMyRank() != nbrRankM)
				printf("Warning P! My rank: %d, RankM: %d RankP: %d\n", getMyRank(), nbrRankM, nbrRankP);

			/*
			if((getMyRank() != nbrRankM))
			{
				loadForceBufferFromGpu_Async(sendBufM, sendSizeM[iAxis], nCellsM, cellListGpuM, 
					parms->natoms_buf_send[faceM], parms->partial_sums[faceM],
					sim, sim->gpu_force_buf, sim->boundary_stream);

				comm_isend_on_stream(sendBufM, 
					sendSizeM[iAxis],
					MPI_CHAR,
					&(haloExchange->regSendM[iAxis]),
					nbrRankM,
					&send_requests[0],
					sim->boundary_stream);
			}
			*/
			loadForceBufferFromGpu_Async(recvBufM, sendSizeP[iAxis], nCellsP, parms->sendCellsGpu[faceP], 
				parms->natoms_buf_send[faceP], parms->partial_sums[faceP],
				sim, sim->gpu_force_buf, sim->boundary_stream);

			POP_RANGE;
		}

		if((getMyRank() != nbrRankM) && (getMyRank() != nbrRankP))
		{
			PUSH_RANGE("Force Both", 3);

			loadForceBufferFromGpu_Async(sendBufM, sendSizeM[iAxis], nCellsM, parms->sendCellsGpu[faceM], 
				/* natoms_buf_send */ 
				parms->natoms_buf_send[faceM], parms->partial_sums[faceM],
				sim, sim->gpu_force_buf, sim->boundary_stream);

			comm_wait_ready_on_stream(nbrRankM, sim->boundary_stream);
			comm_isend_on_stream(sendBufM, 
				sendSizeM[iAxis],
				MPI_CHAR,
				&(haloExchange->regSendM[iAxis]),
				nbrRankM,
				&send_requests[0],
				sim->boundary_stream);


			loadForceBufferFromGpu_Async(sendBufP, sendSizeP[iAxis], nCellsP, parms->sendCellsGpu[faceP], 
				/* natoms_buf_send */ 
				parms->natoms_buf_send[faceP], parms->partial_sums[faceP],
				sim, sim->gpu_force_buf, sim->boundary_stream);


			//-------- COMPUTE & SEND faceP
			comm_wait_ready_on_stream(nbrRankP, sim->boundary_stream);
			comm_isend_on_stream(sendBufP, 
				sendSizeP[iAxis],
				MPI_CHAR,
				&(haloExchange->regSendP[iAxis]),
				nbrRankP,
				&send_requests[1],
				sim->boundary_stream);

			POP_RANGE;
		}
	}

	if(typeFunc == 1 || typeFunc == 2)
	{
		PUSH_RANGE("unload scan", 4);

		unloadForceScanCells(nCellsM, parms->recvCellsGpu[faceM], 
			/* natoms_buf_recv */
			parms->natoms_buf_recv[faceM], parms->partial_sums[faceM], 
			sim, sim->boundary_stream);


		unloadForceScanCells(nCellsP, parms->recvCellsGpu[faceP], 
			/* natoms_buf_recv */
			parms->natoms_buf_recv[faceP], parms->partial_sums[faceP], 
			sim, sim->boundary_stream);
		
		POP_RANGE;

		//-------- Wait recv on stream
		PUSH_RANGE("wait recv", 5);
		if((getMyRank() != nbrRankM) && (getMyRank() != nbrRankP))
			comm_wait_all_on_stream(2, recv_requests, sim->boundary_stream);
		POP_RANGE;

		//-------- Unload P
		PUSH_RANGE("unload gpu", 6);
		unloadForceBufferToGpu_Async(recvBufP, recvSizeP[iAxis], nCellsP, parms->recvCellsGpu[faceP], 
			/* natoms_buf_recv */
			parms->natoms_buf_recv[faceP], parms->partial_sums[faceP], 
			sim, sim->gpu_force_buf, sim->boundary_stream, typeP);

		//-------- Unload M
		unloadForceBufferToGpu_Async(recvBufM, recvSizeM[iAxis], nCellsM, parms->recvCellsGpu[faceM], 
			/* natoms_buf_recv */
			parms->natoms_buf_recv[faceM], parms->partial_sums[faceM], 
			sim, sim->gpu_force_buf, sim->boundary_stream, typeM);  
		POP_RANGE;

		PUSH_RANGE("wait sr", 7);
		if((getMyRank() != nbrRankM) && (getMyRank() != nbrRankP))
		{
			comm_wait_all_on_stream(2, send_requests, sim->boundary_stream);
			comm_wait_all_on_stream(2, ready_requests, sim->boundary_stream);
		}

		POP_RANGE;
	}    
}

void exchangeData_Force_KI(HaloExchange* haloExchange, void* data, int iAxis, 
   comm_request_t * recv_requests, comm_request_t * send_requests, comm_request_t * ready_requests,
   int * sendSizeM, int * sendSizeP,  int * recvSizeM,  int * recvSizeP)
{
   enum HaloFaceOrder faceM = (enum HaloFaceOrder)(2*iAxis);
   enum HaloFaceOrder faceP = (enum HaloFaceOrder)(faceM+1);

   int nbrRankM = haloExchange->nbrRank[faceM];
   int nbrRankP = haloExchange->nbrRank[faceP];

   char * sendBufM, * sendBufP, * sendBufM_d, * sendBufP_d, * recvBufP, * recvBufM;
   sendBufM = (char*)haloExchange->sendBufM_Async[iAxis];
   sendBufP = (char*)haloExchange->sendBufP_Async[iAxis];
   sendBufM_d = (char*)haloExchange->d_sendBufM_Async[iAxis];
   sendBufP_d = (char*)haloExchange->d_sendBufP_Async[iAxis];
   recvBufM = (char*)haloExchange->recvBufM_Async[iAxis];
   recvBufP = (char*)haloExchange->recvBufP_Async[iAxis];
   int typeM=0, typeP=0;

   if((getMyRank() == nbrRankM))
   {
      typeP = 1;
      recvBufP = haloExchange->d_recvBufP_Async[iAxis];
     // sendBufP = haloExchange->d_sendBufP_Async[iAxis];
   }
  
   if((getMyRank() == nbrRankP))
   {
      typeM=1;
      recvBufM = haloExchange->d_recvBufM_Async[iAxis];
      //sendBufM = haloExchange->d_sendBufM_Async[iAxis];
   }
  
   ForceExchangeParms* parms = (ForceExchangeParms*) haloExchange->parms;
   SimFlat* sim = (SimFlat*) data;

   int nCellsM = parms->nCells[faceM];
   int* cellListGpuM = parms->sendCellsGpu[faceM];

   int nCellsP = parms->nCells[faceP];
   int* cellListGpuP = parms->sendCellsGpu[faceP];
   
   if(getMyRank() ==  nbrRankM)
   {
      PUSH_RANGE("nbrRankM", 1);

      loadForceBufferFromGpu_Async(recvBufP, sendSizeM[iAxis], nCellsM, parms->sendCellsGpu[faceM], 
                                       parms->natoms_buf_send[faceM], parms->partial_sums[faceM],
                                       sim, sim->gpu_force_buf, sim->boundary_stream);
      POP_RANGE;
   }
      
   if((getMyRank() == nbrRankP))
   {
      PUSH_RANGE("nbrRankP", 2);

      loadForceBufferFromGpu_Async(recvBufM, sendSizeP[iAxis], nCellsP, parms->sendCellsGpu[faceP], 
                                    parms->natoms_buf_send[faceP], parms->partial_sums[faceP],
                                    sim, sim->gpu_force_buf, sim->boundary_stream);

      POP_RANGE;
   }

   if((getMyRank() != nbrRankM) && (getMyRank() != nbrRankP))
   {
      PUSH_RANGE("DIFFERENTS", 1);

      exchangeDataForceGpu_KI(
        sendBufM, sendBufP,
        sendBufM_d, sendBufP_d, 
        sendSizeM[iAxis], sendSizeP[iAxis],
        recvBufM, recvBufP,
        nCellsM, nCellsP, 
        parms->sendCellsGpu[faceM], parms->sendCellsGpu[faceP], parms->recvCellsGpu[faceM], parms->recvCellsGpu[faceP],
        sim, 
        parms->natoms_buf_send[faceM], parms->natoms_buf_send[faceP], parms->natoms_buf_recv[faceM], parms->natoms_buf_recv[faceP],
        parms->partial_sums_send[faceM], parms->partial_sums_send[faceP], parms->partial_sums_recv[faceM], parms->partial_sums_recv[faceP],
        sim->boundary_stream, iAxis, nbrRankM, nbrRankP);

      POP_RANGE;
   }

   if((getMyRank() ==  nbrRankM) && (getMyRank() ==  nbrRankP))
   {
      unloadForceScanCells(nCellsM, parms->recvCellsGpu[faceM], 
                                    /* natoms_buf_recv */parms->natoms_buf_recv[faceM], parms->partial_sums[faceM], 
                                    sim, sim->boundary_stream);


      unloadForceScanCells(nCellsP, parms->recvCellsGpu[faceP], 
                                    /* natoms_buf_recv */parms->natoms_buf_recv[faceP], parms->partial_sums[faceP], 
                                    sim, sim->boundary_stream);
      //WAIT
      PUSH_RANGE("UNLOAD", 4);
      //-------- Wait recv on stream
      if((getMyRank() != nbrRankM) && (getMyRank() != nbrRankP))
         comm_wait_all_on_stream(2, recv_requests, sim->boundary_stream);

      //UNPACK
      //-------- Unload P
      unloadForceBufferToGpu_Async(recvBufP, recvSizeP[iAxis], nCellsP, parms->recvCellsGpu[faceP], 
                                    /* natoms_buf_recv */parms->natoms_buf_recv[faceP], parms->partial_sums[faceP], 
                                    sim, sim->gpu_force_buf, sim->boundary_stream, typeP);

       //-------- Unload M
      unloadForceBufferToGpu_Async(recvBufM, recvSizeM[iAxis], nCellsM, parms->recvCellsGpu[faceM], 
                                    /* natoms_buf_recv */parms->natoms_buf_recv[faceM], parms->partial_sums[faceM], 
                                    sim, sim->gpu_force_buf, sim->boundary_stream, typeM);   
   

      if((getMyRank() != nbrRankM) && (getMyRank() != nbrRankP))
         comm_wait_all_on_stream(2, send_requests, sim->boundary_stream);

      POP_RANGE;
   }
}

/// This is the function that does the heavy lifting for the
/// communication of halo data.  It is called once for each axis and
/// sends and receives two message.  Loading and unloading of the
/// buffers is in the hands of the sub-class virtual functions.
///
/// \param [in] iAxis     Axis index.
/// \param [in, out] data Pointer to data that will be passed to the load and
///                       unload functions
void exchangeData(HaloExchange* haloExchange, void* data, int iAxis)
{
	enum HaloFaceOrder faceM = (enum HaloFaceOrder)(2*iAxis);
	enum HaloFaceOrder faceP = (enum HaloFaceOrder)(faceM+1);

	char* sendBufM = haloExchange->sendBufM;
	char* sendBufP = haloExchange->sendBufP;
	char* recvBufP = haloExchange->recvBufP;
	char* recvBufM = haloExchange->recvBufM;

	PUSH_RANGE("MPIload", 3);
	int nSendM = haloExchange->loadBuffer(haloExchange->parms, data, faceM, sendBufM);
	int nSendP = haloExchange->loadBuffer(haloExchange->parms, data, faceP, sendBufP);
	POP_RANGE;

	int nbrRankM = haloExchange->nbrRank[faceM];
	int nbrRankP = haloExchange->nbrRank[faceP];

	int nRecvM, nRecvP;

	PUSH_RANGE("MPI send", 4);
	nRecvP = sendReceiveParallel(sendBufM, nSendM, nbrRankM, recvBufP, haloExchange->bufCapacity, nbrRankP);
	nRecvM = sendReceiveParallel(sendBufP, nSendP, nbrRankP, recvBufM, haloExchange->bufCapacity, nbrRankM);
	POP_RANGE;

	PUSH_RANGE("MPIunload", 5);
	haloExchange->unloadBuffer(haloExchange->parms, data, faceM, nRecvM, recvBufM);
	haloExchange->unloadBuffer(haloExchange->parms, data, faceP, nRecvP, recvBufP);
	POP_RANGE;
}

/// Make a list of link cells that need to be sent across the specified
/// face.  For each face, the list must include all cells, local and
/// halo, in the first two planes of link cells.  Halo cells must be
/// included in the list of link cells to send since local atoms may
/// have moved from local cells into halo cells on this time step.
/// (Actual remote atoms should have been deleted, so the halo cells
/// should contain only these few atoms that have just crossed.)
/// Sending these atoms will allow them to be reassigned to the task
/// that covers the spatial domain they have moved into.
///
/// Note that link cell grid coordinates range from -1 to gridSize[iAxis].
/// \see initLinkCells for an explanation link cell grid coordinates.
///
/// \param [in] boxes  Link cell information.
/// \param [in] iFace  Index of the face data will be sent across.
/// \param [in] nCells Number of cells to send.  This is used for a
///                    consistency check.
/// \return The list of cells to send.  Caller is responsible to free
/// the list.
int* mkAtomCellList(LinkCell* boxes, enum HaloFaceOrder iFace, const int nCells)
{
   int* list = (int*)comdMalloc(nCells*sizeof(int));
   int xBegin = -1;
   int xEnd   = boxes->gridSize[0]+1;
   int yBegin = -1;
   int yEnd   = boxes->gridSize[1]+1;
   int zBegin = -1;
   int zEnd   = boxes->gridSize[2]+1;

   if (iFace == HALO_X_MINUS) xEnd = xBegin+2;
   if (iFace == HALO_X_PLUS)  xBegin = xEnd-2;
   if (iFace == HALO_Y_MINUS) yEnd = yBegin+2;
   if (iFace == HALO_Y_PLUS)  yBegin = yEnd-2;
   if (iFace == HALO_Z_MINUS) zEnd = zBegin+2;
   if (iFace == HALO_Z_PLUS)  zBegin = zEnd-2;

   int count = 0;
   for (int ix=xBegin; ix<xEnd; ++ix)
      for (int iy=yBegin; iy<yEnd; ++iy)
         for (int iz=zBegin; iz<zEnd; ++iz)
            list[count++] = getBoxFromTuple(boxes, ix, iy, iz);
   assert(count == nCells);
   return list;
}

/// The loadBuffer function for a halo exchange of atom data.  Iterates
/// link cells in the cellList and load any atoms into the send buffer.
/// This function also shifts coordinates of the atoms by an appropriate
/// factor if they are being sent across a periodic boundary.
///
/// \see HaloExchangeSt::loadBuffer for an explanation of the loadBuffer
/// parameters.
int loadAtomsBuffer(void* vparms, void* data, int face, char* charBuf)
{
   AtomExchangeParms* parms = (AtomExchangeParms*) vparms;
   SimFlat* sim = (SimFlat*) data;
   
   real_t* pbcFactor = parms->pbcFactor[face];
   real3_old shift;
   shift[0] = pbcFactor[0] * sim->domain->globalExtent[0];
   shift[1] = pbcFactor[1] * sim->domain->globalExtent[1];
   shift[2] = pbcFactor[2] * sim->domain->globalExtent[2];
   
   int nCells = parms->nCells[face];

   int nTotalAtomsCellList = 0;
//   if(sim->method == 3 ||sim->method == 4)
   if(sim->method == CPU_NL)
   {
           int* cellList = parms->cellList[face];

           AtomMsg* buf = (AtomMsg*) charBuf;
           for (int iCell=0; iCell<nCells; ++iCell)
           {
                   int iBox = cellList[iCell];
                   int iOff = iBox*MAXATOMS;
                   for (int ii=iOff; ii<iOff+sim->boxes->nAtoms[iBox]; ++ii)
                   {
                           buf[nTotalAtomsCellList].gid  = sim->atoms->gid[ii];
                           buf[nTotalAtomsCellList].type = sim->atoms->iSpecies[ii];
                           buf[nTotalAtomsCellList].rx = sim->atoms->r.x[ii] + shift[0];
                           buf[nTotalAtomsCellList].ry = sim->atoms->r.y[ii] + shift[1];
                           buf[nTotalAtomsCellList].rz = sim->atoms->r.z[ii] + shift[2];
                           buf[nTotalAtomsCellList].px = sim->atoms->p.x[ii];
                           buf[nTotalAtomsCellList].py = sim->atoms->p.y[ii];
                           buf[nTotalAtomsCellList].pz = sim->atoms->p.z[ii];
                           ++nTotalAtomsCellList;
                   }
           }
//           printf("cpu: %d\n",nTotalAtomsCellList);
   }else{
           int* d_cellList = parms->cellListGpu[face];

           nTotalAtomsCellList = compactCellsGpu(sim->gpu_atoms_buf, nCells, d_cellList, sim->gpu,  parms->d_natoms_buf, parms->d_partial_sums,shift,sim->boundary_stream);
//           printf("gpu: %d\n",nTotalAtomsCellList);
            if((face%2) == 0)
            {
//               if(getMyRank() == 0) printf("RANK[%d], msgM, nTotalAtomsCellList: %d, sizeMsgIndex: %d face: %d\n",  getMyRank(), nTotalAtomsCellList, sizeMsgIndex, face);
               sizeMsgM[sizeMsgIndexM] = nTotalAtomsCellList;
               sizeMsgIndexM++;
            }
            else
            {
//               if(getMyRank() == 0) printf("RANK[%d], msgP, nTotalAtomsCellList: %d, sizeMsgIndex: %d face: %d\n",  getMyRank(), nTotalAtomsCellList, sizeMsgIndex, face);
               sizeMsgP[sizeMsgIndexP] = nTotalAtomsCellList;
               sizeMsgIndexP++;
            }

           cudaMemcpyAsync(charBuf, (void*)(sim->gpu_atoms_buf), nTotalAtomsCellList * sizeof(AtomMsg), cudaMemcpyDeviceToHost,sim->boundary_stream);
           cudaStreamSynchronize(sim->boundary_stream);
   }
   return nTotalAtomsCellList*sizeof(AtomMsg);
}

/// The unloadBuffer function for a halo exchange of atom data.
/// Iterates the receive buffer and places each atom that was received
/// into the link cell that corresponds to the atom coordinate.  Note
/// that this naturally accomplishes transfer of ownership of atoms that
/// have moved from one spatial domain to another.  Atoms with
/// coordinates in local link cells automatically become local
/// particles.  Atoms that are owned by other ranks are automatically
/// placed in halo kink cells.
/// \see HaloExchangeSt::unloadBuffer for an explanation of the
/// unloadBuffer parameters.
/// @param face Not used for this function. The only reason we keep it is to match the unloadForcesBuffer declaration
void unloadAtomsBuffer(void* vparms, void* data, int face, int bufSize, char* charBuf)
{
   AtomExchangeParms* parms = (AtomExchangeParms*) vparms;
   SimFlat* sim = (SimFlat*) data;
   assert(bufSize % sizeof(AtomMsg) == 0);

   int nBuf = bufSize / sizeof(AtomMsg);

   if(sim->method == CPU_NL)
   {

           AtomMsg* buf = (AtomMsg*) charBuf;
           //   const int nlUpdateRequired = neighborListUpdateRequired(s->atoms->neighborList,s->boxes,s->atoms);
           const int nlUpdateRequired = neighborListUpdateRequired(sim);
           for (int ii=0; ii<nBuf; ++ii)
           {
                   int gid   = buf[ii].gid;
                   int type  = buf[ii].type;
                   real_t rx = buf[ii].rx;
                   real_t ry = buf[ii].ry;
                   real_t rz = buf[ii].rz;
                   real_t px = buf[ii].px;
                   real_t py = buf[ii].py;
                   real_t pz = buf[ii].pz;
                   if(nlUpdateRequired){
                           int iOff = putAtomInBox(sim->boxes, sim->atoms, gid, type, rx, ry, rz, px, py, pz);

                           if(iOff >= (MAXATOMS*sim->boxes->nLocalBoxes))
                                   hashTablePut(sim->atomExchange->hashTable, iOff); //remember iOff only for particles which are mapped to haloCells
                           else //putting particle to local 
                                   sim->atoms->neighborList->updateLinkCellsRequired = 1; 
                   }else{
                           int iOff = hashTableGet(sim->atomExchange->hashTable);
                           updateAtomInBoxAt(sim->boxes, sim->atoms, gid, type, rx, ry, rz, px, py, pz,iOff);
                   }
           }
   }else{
      unloadAtomsBufferToGpu(charBuf, nBuf, sim, sim->gpu_atoms_buf, sim->boundary_stream);   
   }
}

void destroyAtomsExchange(void* vparms)
{
   AtomExchangeParms* parms = (AtomExchangeParms*) vparms;

   for (int ii=0; ii<6; ++ii)
   {
      free(parms->pbcFactor[ii]);
      free(parms->cellList[ii]);
      cudaFree(parms->cellListGpu[ii]);
   }
   cudaFree(parms->d_natoms_buf);
   cudaFree(parms->d_partial_sums);
   free(parms->h_natoms_buf);
}

/// Make a list of link cells that need to send data across the
/// specified face.  Note that this list must be compatible with the
/// corresponding recv list to ensure that the data goes to the correct
/// atoms.
///
/// \see initLinkCells for information about the conventions for grid
/// coordinates of link cells.
int* mkForceSendCellList(LinkCell* boxes, int face, int nCells)
{
   int* list = (int*)comdMalloc(nCells*sizeof(int));
   int xBegin, xEnd, yBegin, yEnd, zBegin, zEnd;

   int nx = boxes->gridSize[0];
   int ny = boxes->gridSize[1];
   int nz = boxes->gridSize[2];
   switch(face)
   {
     case HALO_X_MINUS:
      xBegin=0;    xEnd=1;    yBegin=0;    yEnd=ny;   zBegin=0;    zEnd=nz;
      break;
     case HALO_X_PLUS:
      xBegin=nx-1; xEnd=nx;   yBegin=0;    yEnd=ny;   zBegin=0;    zEnd=nz;
      break;
     case HALO_Y_MINUS:
      xBegin=-1;   xEnd=nx+1; yBegin=0;    yEnd=1;    zBegin=0;    zEnd=nz;
      break;
     case HALO_Y_PLUS:
      xBegin=-1;   xEnd=nx+1; yBegin=ny-1; yEnd=ny;   zBegin=0;    zEnd=nz;
      break;
     case HALO_Z_MINUS:
      xBegin=-1;   xEnd=nx+1; yBegin=-1;   yEnd=ny+1; zBegin=0;    zEnd=1;
      break;
     case HALO_Z_PLUS:
      xBegin=-1;   xEnd=nx+1; yBegin=-1;   yEnd=ny+1; zBegin=nz-1; zEnd=nz;
      break;
     default:
      assert(1==0);
   }
   
   int count = 0;
   for (int ix=xBegin; ix<xEnd; ++ix)
      for (int iy=yBegin; iy<yEnd; ++iy)
         for (int iz=zBegin; iz<zEnd; ++iz)
            list[count++] = getBoxFromTuple(boxes, ix, iy, iz);
   
   assert(count == nCells);
   return list;
}

/// Make a list of link cells that need to receive data across the
/// specified face.  Note that this list must be compatible with the
/// corresponding send list to ensure that the data goes to the correct
/// atoms.
///
/// \see initLinkCells for information about the conventions for grid
/// coordinates of link cells.
int* mkForceRecvCellList(LinkCell* boxes, int face, int nCells)
{
   int* list = (int*)comdMalloc(nCells*sizeof(int));
   int xBegin, xEnd, yBegin, yEnd, zBegin, zEnd;

   int nx = boxes->gridSize[0];
   int ny = boxes->gridSize[1];
   int nz = boxes->gridSize[2];
   switch(face)
   {
     case HALO_X_MINUS:
      xBegin=-1; xEnd=0;    yBegin=0;  yEnd=ny;   zBegin=0;  zEnd=nz;
      break;
     case HALO_X_PLUS:
      xBegin=nx; xEnd=nx+1; yBegin=0;  yEnd=ny;   zBegin=0;  zEnd=nz;
      break;
     case HALO_Y_MINUS:
      xBegin=-1; xEnd=nx+1; yBegin=-1; yEnd=0;    zBegin=0;  zEnd=nz;
      break;
     case HALO_Y_PLUS:
      xBegin=-1; xEnd=nx+1; yBegin=ny; yEnd=ny+1; zBegin=0;  zEnd=nz;
      break;
     case HALO_Z_MINUS:
      xBegin=-1; xEnd=nx+1; yBegin=-1; yEnd=ny+1; zBegin=-1; zEnd=0;
      break;
     case HALO_Z_PLUS:
      xBegin=-1; xEnd=nx+1; yBegin=-1; yEnd=ny+1; zBegin=nz; zEnd=nz+1;
      break;
     default:
      assert(1==0);
   }
   
   int count = 0;
   for (int ix=xBegin; ix<xEnd; ++ix)
      for (int iy=yBegin; iy<yEnd; ++iy)
         for (int iz=zBegin; iz<zEnd; ++iz)
            list[count++] = getBoxFromTuple(boxes, ix, iy, iz);
   
   assert(count == nCells);
   return list;
}



/// The unloadBuffer function for a force exchange.
/// Data is received in an order that naturally aligns with the atom
/// storage so it is simple to put the data where it belongs.
///
/// \see HaloExchangeSt::unloadBuffer for an explanation of the
/// unloadBuffer parameters.
void unloadForceBufferCpu(void* vparms, void* vdata, int face, int bufSize, char* charBuf)
{
   ForceExchangeParms* parms = (ForceExchangeParms*) vparms;
   ForceExchangeData* data = (ForceExchangeData*) vdata;
   ForceMsg* buf = (ForceMsg*) charBuf;
   assert(bufSize % sizeof(ForceMsg) == 0);
   
   int nCells = parms->nCells[face];
   int* cellList = parms->recvCells[face];
   int iBuf = 0;
   for (int iCell=0; iCell<nCells; ++iCell)
   {
      int iBox = cellList[iCell];
      int iOff = iBox*MAXATOMS;
      for (int ii=iOff; ii<iOff+data->boxes->nAtoms[iBox]; ++ii)
      {
         data->dfEmbed[ii] = buf[iBuf].dfEmbed;
         ++iBuf;
      }
   }
   assert(iBuf == bufSize/ sizeof(ForceMsg));
}

/// The loadBuffer function for a force exchange.
/// Iterate the send list and load the derivative of the embedding
/// energy with respect to the local density into the send buffer.
///
/// \see HaloExchangeSt::loadBuffer for an explanation of the loadBuffer
/// parameters.
int loadForceBufferCpu(void* vparms, void* vdata, int face, char* charBuf)
{
   ForceExchangeParms* parms = (ForceExchangeParms*) vparms;
   ForceExchangeData* data = (ForceExchangeData*) vdata;
   ForceMsg* buf = (ForceMsg*) charBuf;
   
   int nCells = parms->nCells[face];
   int* cellList = parms->sendCells[face];
   int nBuf = 0;
   for (int iCell=0; iCell<nCells; ++iCell)
   {
      int iBox = cellList[iCell];
      int iOff = iBox*MAXATOMS;
      for (int ii=iOff; ii<iOff+data->boxes->nAtoms[iBox]; ++ii)
      {
         buf[nBuf].dfEmbed = data->dfEmbed[ii];
         ++nBuf;
      }
   }
   return nBuf*sizeof(ForceMsg);
}
/// The loadBuffer function for a force exchange.
/// Iterate the send list and load the derivative of the embedding
/// energy with respect to the local density into the send buffer.
///
/// \see HaloExchangeSt::loadBuffer for an explanation of the loadBuffer
/// parameters.
int loadForceBuffer(void* vparms, void* vdata, int face, char* charBuf)
{
   ForceExchangeParms* parms = (ForceExchangeParms*) vparms;
   SimFlat* s = (SimFlat*) vdata;
   
   int nCells = parms->nCells[face];
   int* cellListGpu = parms->sendCellsGpu[face];
   int nBuf = 0;

   loadForceBufferFromGpu(charBuf, &nBuf, nCells, cellListGpu, parms->natoms_buf[face], parms->partial_sums[face], s, s->gpu_force_buf, s->boundary_stream);

   return nBuf*sizeof(ForceMsg);
}

/// The unloadBuffer function for a force exchange.
/// Data is received in an order that naturally aligns with the atom
/// storage so it is simple to put the data where it belongs.
///
/// \see HaloExchangeSt::unloadBuffer for an explanation of the
/// unloadBuffer parameters.
void unloadForceBuffer(void* vparms, void* vdata, int face, int bufSize, char* charBuf)
{
   ForceExchangeParms* parms = (ForceExchangeParms*) vparms;
   SimFlat* s = (SimFlat*) vdata;
   assert(bufSize % sizeof(ForceMsg) == 0);
   
   int nCells = parms->nCells[face];
   int* cellListGpu = parms->recvCellsGpu[face];
   int nBuf = bufSize / sizeof(ForceMsg);
  
   unloadForceBufferToGpu(charBuf, nBuf, nCells, cellListGpu, parms->natoms_buf[face], parms->partial_sums[face], s, s->gpu_force_buf, s->boundary_stream);   
}

void destroyForceExchange(void* vparms)
{
   ForceExchangeParms* parms = (ForceExchangeParms*) vparms;

   for (int ii=0; ii<6; ++ii)
   {
      free(parms->sendCells[ii]);
      free(parms->recvCells[ii]);
      cudaFree(parms->sendCellsGpu[ii]);
      cudaFree(parms->recvCellsGpu[ii]);
      cudaFree(parms->natoms_buf[ii]);
      //elenago
      cudaFree(parms->natoms_buf_send[ii]);
      cudaFree(parms->natoms_buf_recv[ii]);
      cudaFree(parms->partial_sums[ii]);
   }
}

/// \details
/// The force exchange assumes that the atoms are in the same order in
/// both a given local link cell and the corresponding remote cell(s).
/// However, the atom exchange does not guarantee this property,
/// especially when atoms cross a domain decomposition boundary and move
/// from one task to another.  Trying to maintain the atom order during
/// the atom exchange would immensely complicate that code.  Instead, we
/// just sort the atoms after the atom exchange.
void sortAtomsInCell(Atoms* atoms, LinkCell* boxes, int iBox)
{
   int nAtoms = boxes->nAtoms[iBox];

#if defined(_WIN32) || defined(_WIN64)
   AtomMsg *tmp = (AtomMsg*)malloc(sizeof(AtomMsg) * nAtoms);
#else
   AtomMsg tmp[nAtoms];
#endif

   int begin = iBox*MAXATOMS;
   int end = begin + nAtoms;
   for (int ii=begin, iTmp=0; ii<end; ++ii, ++iTmp)
   {
      tmp[iTmp].gid  = atoms->gid[ii];
      tmp[iTmp].type = atoms->iSpecies[ii];
      tmp[iTmp].rx =   atoms->r.x[ii];
      tmp[iTmp].ry =   atoms->r.y[ii];
      tmp[iTmp].rz =   atoms->r.z[ii];
      tmp[iTmp].px =   atoms->p.x[ii];
      tmp[iTmp].py =   atoms->p.y[ii];
      tmp[iTmp].pz =   atoms->p.z[ii];
   }
   qsort(&tmp, nAtoms, sizeof(AtomMsg), sortAtomsById);
   for (int ii=begin, iTmp=0; ii<end; ++ii, ++iTmp)
   {
      atoms->gid[ii]   = tmp[iTmp].gid;
      atoms->iSpecies[ii] = tmp[iTmp].type;
      atoms->r.x[ii]  = tmp[iTmp].rx;
      atoms->r.y[ii]  = tmp[iTmp].ry;
      atoms->r.z[ii]  = tmp[iTmp].rz;
      atoms->p.x[ii]  = tmp[iTmp].px;
      atoms->p.y[ii]  = tmp[iTmp].py;
      atoms->p.z[ii]  = tmp[iTmp].pz;
   }

#if defined(_WIN32) || defined(_WIN64)
   free(tmp);
#endif
}

///  A function suitable for passing to qsort to sort atoms by gid.
///  Because every atom in the simulation is supposed to have a unique
///  id, this function checks that the atoms have different gids.  If
///  that assertion ever fails it is a sign that something has gone
///  wrong elsewhere in the code.
int sortAtomsById(const void* a, const void* b)
{
   int aId = ((AtomMsg*) a)->gid;
   int bId = ((AtomMsg*) b)->gid;
   assert(aId != bId);

   if (aId < bId)
      return -1;
   return 1;
}

