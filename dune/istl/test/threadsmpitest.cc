// 1D FEM scheme for equation -u''=f with Dirichlet boundary conditions to test MPI and threads

#if HAVE_CONFIG_H
#include "config.h"
#endif

#define WORLDDIM 1
#define GRIDDIM 1

// includes
#include <iostream>
#include <vector>
#include <functional>
#include <string>
#include <thread>

#include <dune/common/fvector.hh>
#include <dune/common/dynmatrix.hh>
#include <dune/common/dynvector.hh>
#include <dune/common/enumset.hh>

#include <dune/common/parallel/mpicollectivecommunication.hh>
#include <dune/common/parallel/indexset.hh>
#include <dune/common/parallel/plocalindex.hh>
#include <dune/common/parallel/remoteindices.hh>
#include <dune/common/parallel/interface.hh>
#include <dune/common/parallel/communicator.hh>

// problem definition
const double x0Global(0.0);
const double lengthGrid(0.5);
inline double uExact(double&& x){return x*x+1;}
inline double f(double&& x){return 2.0;}

// threads control
#define NUM_THREADS 4
#define GRID_ELEMENTS_PER_THREAD 2

//debug flag
#define DEBUG_THREADS_TEST 1

// basis functions
inline double phi0(double&& x){return 1.0-x;}
inline double phi1(double&& x){return x;}
inline double derphi0(double&& x){return -1.0;}
inline double derphi1(double&& x){return 1.0;}

// assemble function
template<typename M,typename V,typename G,typename F>
void assemble(size_t tid,M& A_thr,V& b_thr,G& grid,size_t numGridElements,F& phi,F& derphi){

    std::vector<std::vector<double>> A_entity(2,std::vector<double>(2,0.0));
    std::vector<double> b_entity(2,0.0);

    size_t startElem(tid*numGridElements);
    size_t endElem((tid+1)*numGridElements);
    for(size_t elem=startElem;elem!=endElem;++elem){

        // assemble A_enity and b_entity
        for(size_t i=0;i!=2;++i){

            b_entity[i]=0.0;
            // using trapezoid rule
            b_entity[i]+=0.5*(phi[i](0.0)*f(0.0));
            b_entity[i]+=0.5*(phi[i](1.0)*f(1.0));
            b_entity[i]*=(grid[elem+1]-grid[elem]);

            for(size_t j=0;j!=2;++j){
                A_entity[i][j]=0.0;
                // using trapezoid rule
                A_entity[i][j]+=0.5*(derphi[i](0.0)*derphi[j](0.0));
                A_entity[i][j]+=0.5*(derphi[i](1.0)*derphi[j](1.0));
                A_entity[i][j]/=(grid[elem+1]-grid[elem]);
            }

        }

        // add entity contributions to A_thr and b_thr
        for(size_t i=0;i!=2;++i){
            b_thr[elem-startElem+i]+=b_entity[i];
            for(size_t j=0;j!=2;++j){
                A_thr[elem-startElem+i][elem-startElem+j]+=A_entity[i][j];
            }
        }

    }

}

// push function
template<typename M,typename V>
void push(size_t tid,M& A_thr,V& b_thr,M& A,V& b){

  size_t dim(b_thr.size());
  size_t offset((dim-1)*tid);
  for(size_t i=0;i!=dim;++i){
    b[i+offset]+=b_thr[i];
    for(size_t j=0;j!=dim;++j) A[i+offset][j+offset]+=A_thr[i][j];
  }

}

// some printing routines for debugging
// parallel sync print of a value for all the processes
template<typename T,typename C>
void printAll(std::string&& str,T& value,C& comm){
  #ifdef DEBUG_THREADS_TEST
  #if DEBUG_THREADS_TEST
  for(size_t i=0;i!=comm.size();++i){
    if(comm.rank()==i) std::cout<<str<<" (rank "<<comm.rank()<<")\t"<<value<<std::endl;
    comm.barrier();
  }
  #endif
  #endif
}

// parallel sync print of a value for only 1 process (default rank=0)
template<typename T,typename C>
void printOne(std::string&& str,T& value,C& comm,size_t rank=0){
  #ifdef DEBUG_THREADS_TEST
  #if DEBUG_THREADS_TEST
  if(comm.rank()==rank) std::cout<<str<<"\t"<<value<<std::endl;
  comm.barrier();
  #endif
  #endif
}

// parallel sync print of a vector for all the processes
template<typename T,typename C>
void printAll(std::string&& str,std::vector<T>& v,C& comm){
  #ifdef DEBUG_THREADS_TEST
  #if DEBUG_THREADS_TEST
  for(size_t i=0;i!=comm.size();++i){
    if(comm.rank()==i){
      std::cout<<str<<" (rank "<<comm.rank()<<")\t";
      for(typename std::vector<T>::iterator it=v.begin();it!=v.end();++it) std::cout<<*it<<" ";
      std::cout<<std::endl;
    }
    comm.barrier();
  }
  #endif
  #endif
}

// parallel sync print of a vector for only 1 process (default rank=0)
template<typename T,typename C>
void printOne(std::string&& str,std::vector<T>& v,C& comm,size_t rank=0){
  #ifdef DEBUG_THREADS_TEST
  #if DEBUG_THREADS_TEST
  if(comm.rank()==rank){
    std::cout<<str<<"\t";
    for(typename std::vector<T>::iterator it=v.begin();it!=v.end();++it) std::cout<<*it<<" ";
    std::cout<<std::endl;
  }
  comm.barrier();
  #endif
  #endif
}

int main(int argc,char** argv){

  // init MPI
  MPI_Init(&argc,&argv);
  Dune::CollectiveCommunication<MPI_Comm> comm(MPI_COMM_WORLD);

  // get size and rank
  const size_t size(comm.size());
  const size_t rank(comm.rank());

  // local geometry definition
  typedef Dune::FieldVector<double,WORLDDIM> CoordType;
  CoordType x0(x0Global+rank*lengthGrid);
  CoordType x1(x0+lengthGrid);

  // number of threads
  const size_t numThreads((NUM_THREADS<2?2:NUM_THREADS));

  // number of grid elements managed by each thread
  const size_t numGridElementsPerThread((GRID_ELEMENTS_PER_THREAD<1?1:GRID_ELEMENTS_PER_THREAD));

  // crate grid
  const size_t numNodes(numThreads*numGridElementsPerThread+1);
  const double deltax(lengthGrid/(numNodes-1));
  typedef std::vector<CoordType> GridType;
  GridType grid(numNodes,x0);

  for(size_t i=1;i!=numNodes;++i) grid[i]+=(deltax*i);
  printAll("Local grid",grid,comm);
  printOne("","",comm);

  // define parallel local index and parallel index set
  enum flags{owner,ghost};
  typedef Dune::ParallelLocalIndex<flags> LocalIndexType;
  typedef Dune::ParallelIndexSet<size_t,LocalIndexType,numNodes> ParallelIndexType;

  // create parallel index set sis
  ParallelIndexType pis;
  const size_t firstGlobalIdx((numNodes-1)*rank);
  const size_t lastGlobalIdx(firstGlobalIdx+numNodes-1);

  pis.beginResize();
  for(size_t i=firstGlobalIdx;i!=(lastGlobalIdx-1);++i) pis.add(i,LocalIndexType(i-firstGlobalIdx,owner));
  if(rank!=(size-1)) pis.add(lastGlobalIdx,LocalIndexType(lastGlobalIdx-firstGlobalIdx,ghost));
  else pis.add(lastGlobalIdx,LocalIndexType(lastGlobalIdx-firstGlobalIdx,owner));
  pis.endResize();

  printAll("Parallel index set",pis,comm);
  printOne("","",comm);

  // create remote indicx set ris
  typedef Dune::RemoteIndices<ParallelIndexType> RemoteIndicesType;
  RemoteIndicesType ris(pis,pis,MPI_COMM_WORLD);
  ris.rebuild<true>();

  // create interface
  Dune::EnumItem<flags,ghost> ghostFlags;
  Dune::EnumItem<flags,owner> ownerFlags;

  typedef Dune::Interface InterfaceType;
  InterfaceType interface(MPI_COMM_WORLD);
  interface.build(ris,ownerFlags,ghostFlags);

  // set color (each row contains all the thread with the same color)
  size_t numColors(2); // 0 when tid is even, 1 when tid is odd
  std::vector<std::vector<size_t>> colors(numColors);
  for(size_t i=0;i!=2;++i){
    colors[i].resize((numThreads+1*(1-i))/2);
    for(size_t j=0;j!=colors[i].size();++j) colors[i][j]=j*2+1*i;
  }

  printOne("Threads with color 0:",colors[0],comm);
  printOne("Threads with color 1:",colors[1],comm);
  printOne("","",comm);

  // allocate stiffness matrix A, RHS vector b and solution vector x
  typedef Dune::DynamicMatrix<double> StiffnessMatrixType;
  StiffnessMatrixType A(numNodes,numNodes,0.0);

  typedef Dune::DynamicVector<double> VectorType;
  VectorType b(numNodes,0.0);
  VectorType x(numNodes,0.0);

  // allocate local stiffness matrices A_thr, local RHS vectors b_thr and local solution vectors x_thr
  // here local means that is not shared among threads
  std::vector<StiffnessMatrixType> A_thr(numThreads,StiffnessMatrixType(numGridElementsPerThread+1,numGridElementsPerThread+1,0.0));
  std::vector<VectorType> b_thr(numThreads,VectorType(numGridElementsPerThread+1,0.0));
  std::vector<VectorType> x_thr(numThreads,VectorType(numGridElementsPerThread+1,0.0));

  // basis functions
  typedef std::function<double(double&&)> FunctionType;

  std::vector<FunctionType> phi(2);
  phi[0]=phi0;
  phi[1]=phi1;

  std::vector<FunctionType> derphi(2);
  derphi[0]=derphi0;
  derphi[1]=derphi1;

  // launch a group of threads to assemble the local stiffness matrices and the local RHS vectors
  // here local means that is not shared among threads
  std::vector<std::thread> thr(numThreads);

  for(size_t i=0;i!=numThreads;++i) thr[i]=std::thread(assemble<StiffnessMatrixType,VectorType,GridType,std::vector<FunctionType>>,i,std::ref(A_thr[i]),std::ref(b_thr[i]),std::ref(grid),numGridElementsPerThread,std::ref(phi),std::ref(derphi));
  for(size_t i=0;i!=numThreads;++i) thr[i].join();

  // launch a group of threads to update the stiffness matrix and the RHS with the values computed with asseble()
  for(size_t i=0;i!=numColors;++i){
    for(size_t j=0;j!=colors[i].size();++j) thr[j]=std::thread(push<StiffnessMatrixType,VectorType>,colors[i][j],std::ref(A_thr[i]),std::ref(b_thr[i]),std::ref(A),std::ref(b));
    for(size_t j=0;j!=colors[i].size();++j) thr[j].join();
  }

  // finalize MPI
  MPI_Finalize();

  return 0;

}
