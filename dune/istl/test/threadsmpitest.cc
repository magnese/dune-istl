// 1D FEM scheme for equation -u''=f with Dirichlet boundary conditions to test MPI and threads

#if HAVE_CONFIG_H
#include "config.h"
#endif

#define WORLDDIM 1
#define GRIDDIM 1

#include <iostream>
#include <vector>
#include <functional>
#include <string>
#include <thread>

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
#define DEBUG_FLAG 1

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

  const size_t startElem(tid*numGridElements);
  const size_t endElem((tid+1)*numGridElements);
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

  const size_t dim(b_thr.size());
  const size_t offset((dim-1)*tid);
  for(size_t i=0;i!=dim;++i){
    b[i+offset]+=b_thr[i];
    for(size_t j=0;j!=dim;++j) A[i+offset][j+offset]+=A_thr[i][j];
  }

}

// vector communication policy: copy
template<typename T>
class CopyData{

public:

  typedef typename T::value_type IndexedType;

  static IndexedType gather(const T& v,int i){return v[i];}
  static void scatter(T& v,IndexedType item,int i){v[i]=item;}

};

// vector communication  policy: add
template<typename T>
class AddData{

public:

  typedef typename T::value_type IndexedType;

  static IndexedType gather(const T& v,int i){return v[i];}
  static void scatter(T& v,IndexedType item,int i){v[i]+=item;}

};

// matrix communication policy: copy
template<typename T>
class CopyDataMatrix{

public:

  typedef typename T::value_type IndexedType;

  static IndexedType gather(const T& m,int i){return m[i/m.rows()][i%m.rows()];}
  static void scatter(T& m,IndexedType item,int i){m[i/m.rows()][i%m.rows()]=item;}

};

// matrix communication  policy: add
template<typename T>
class AddDataMatrix{

public:

  typedef typename T::value_type IndexedType;

  static IndexedType gather(const T& m,int i){return m[i/m.rows()][i%m.rows()];}
  static void scatter(T& m,IndexedType item,int i){m[i/m.rows()][i%m.rows()]+=item;}

};

// parallel sync print of a value for all the processes
template<typename T,typename C>
void printAll(std::string&& str,T& value,C& comm){
  #ifdef DEBUG_FLAG
  #if DEBUG_FLAG
  for(size_t i=0;i!=comm.size();++i){
    if(comm.rank()==i) std::cout<<"[rank "<<comm.rank()<<"] "<<str<<value<<std::endl;
    comm.barrier();
  }
  #endif
  #endif
}

// parallel sync print of a value for only 1 process (default rank=0)
template<typename T,typename C>
void printOne(std::string&& str,T& value,C& comm,size_t rank=0){
  #ifdef DEBUG_FLAG
  #if DEBUG_FLAG
  if(comm.rank()==rank) std::cout<<str<<value<<std::endl;
  comm.barrier();
  #endif
  #endif
}

// parallel sync print of a vector for all the processes
template<typename T,typename C>
void printAll(std::string&& str,std::vector<T>& v,C& comm){
  #ifdef DEBUG_FLAG
  #if DEBUG_FLAG
  for(size_t i=0;i!=comm.size();++i){
    if(comm.rank()==i){
      std::cout<<"[rank "<<comm.rank()<<"] "<<str;
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
  #ifdef DEBUG_FLAG
  #if DEBUG_FLAG
  if(comm.rank()==rank){
    std::cout<<str;
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
  typedef double ctype;
  typedef Dune::FieldVector<ctype,WORLDDIM> CoordType;
  CoordType x0(x0Global+rank*lengthGrid);
  CoordType x1(x0+lengthGrid);

  // number of threads
  const size_t numThreads((NUM_THREADS<2?2:NUM_THREADS));

  // number of grid elements managed by each thread
  const size_t numGridElementsPerThread((GRID_ELEMENTS_PER_THREAD<1?1:GRID_ELEMENTS_PER_THREAD));

  // crate grid
  const size_t numNodes(numThreads*numGridElementsPerThread+1);
  const size_t numNodesGlobal((numNodes-1)*size+1);
  const ctype deltax(lengthGrid/(numNodes-1));
  typedef std::vector<CoordType> GridType;
  GridType grid(numNodes,x0);

  for(size_t i=1;i!=numNodes;++i) grid[i]+=(deltax*i);
  printAll("Local grid: ",grid,comm);
  printOne("","",comm);

  // define parallel local index and parallel index set
  enum flags{owner,overlap,border};
  typedef Dune::ParallelLocalIndex<flags> LocalIndexType;
  typedef Dune::ParallelIndexSet<size_t,LocalIndexType,numNodes> VectorParallelIndexType;
  typedef Dune::ParallelIndexSet<size_t,LocalIndexType,numNodes*numNodes> MatrixParallelIndexType;

  // create parallel index set for vector
  VectorParallelIndexType vectorPIS;
  const size_t vectorFirstGlobalIdx((numNodes-1)*rank);
  const size_t vectorLastGlobalIdx(vectorFirstGlobalIdx+numNodes-1);
  flags flg;

  vectorPIS.beginResize();
  if(rank==0) flg=border;
  else flg=overlap;
  vectorPIS.add(vectorFirstGlobalIdx,LocalIndexType(0,flg));
  for(size_t i=(vectorFirstGlobalIdx+1);i!=(vectorLastGlobalIdx-1);++i) vectorPIS.add(i,LocalIndexType(i-vectorFirstGlobalIdx,owner));
  if(rank==(size-1)) flg=border;
  else flg=overlap;
  vectorPIS.add(vectorLastGlobalIdx,LocalIndexType(vectorLastGlobalIdx-vectorFirstGlobalIdx,flg));
  vectorPIS.endResize();

  printAll("Vector parallel index set: ",vectorPIS,comm);
  printOne("","",comm);

  //create parallel index set for matrix
  MatrixParallelIndexType matrixPIS;
  const size_t matrixGlobalOffset((numNodes-1)*rank);

  matrixPIS.beginResize();
  for(size_t i=0;i!=numNodes;++i){
    for(size_t j=0;j!=numNodes;++j){
       flg=owner;
       if(i==0&&j==0){
         if(rank==0) flg=border;
         else flg=overlap;
       }
       if(i==(numNodes-1)&&j==(numNodes-1)){
         if(rank==(size-1)) flg=border;
         else flg=overlap;
       }
       matrixPIS.add((i+matrixGlobalOffset)*numNodesGlobal+(j+matrixGlobalOffset),LocalIndexType(i*numNodes+j,flg));
    }
  }
  matrixPIS.endResize();

  printAll("Matrix parallel index set: ",matrixPIS,comm);
  printOne("","",comm);

  // create remote index set for vector
  typedef Dune::RemoteIndices<VectorParallelIndexType> VectorRemoteIndicesType;
  VectorRemoteIndicesType vectorRIS(vectorPIS,vectorPIS,MPI_COMM_WORLD);
  vectorRIS.rebuild<true>();

  // create remote index set for matrix
  typedef Dune::RemoteIndices<MatrixParallelIndexType> MatrixRemoteIndicesType;
  MatrixRemoteIndicesType matrixRIS(matrixPIS,matrixPIS,MPI_COMM_WORLD);
  matrixRIS.rebuild<true>();

  // create interface for vector
  Dune::EnumItem<flags,overlap> overlapFlags;
  //Dune::EnumItem<flags,owner> ownerFlags;
  //Dune::EnumItem<flags,border> borderFlags;

  typedef Dune::Interface VectorInterfaceType;
  VectorInterfaceType vectorInterface(MPI_COMM_WORLD);
  vectorInterface.build(vectorRIS,overlapFlags,overlapFlags);

  // create interface for matrix
  typedef Dune::Interface MatrixInterfaceType;
  MatrixInterfaceType matrixInterface(MPI_COMM_WORLD);
  matrixInterface.build(matrixRIS,overlapFlags,overlapFlags);

  // set color (each row contains all the thread with the same color)
  size_t numColors(2); // 0 when tid is even, 1 when tid is odd
  std::vector<std::vector<size_t>> colors(numColors);
  for(size_t i=0;i!=2;++i){
    colors[i].resize((numThreads+1*(1-i))/2);
    for(size_t j=0;j!=colors[i].size();++j) colors[i][j]=j*2+1*i;
  }

  printOne("Threads with color 0: ",colors[0],comm);
  printOne("Threads with color 1: ",colors[1],comm);
  printOne("","",comm);

  // allocate stiffness matrix A, RHS vector b and solution vector x
  typedef Dune::DynamicMatrix<ctype> StiffnessMatrixType;
  StiffnessMatrixType A(numNodes,numNodes,0.0);

  typedef Dune::DynamicVector<ctype> VectorType;
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

  // impose boundary condition
  if(rank==0){
    A[0][0]=1;
    for(size_t i=1;i!=numNodes;++i) A[0][i]=0;
    b[0]=uExact(std::move(x0[0]));
  }

  if(rank==(size-1)){
    A[numNodes-1][numNodes-1]=1;
    for(size_t i=0;i!=(numNodes-1);++i) A[numNodes-1][i]=0;
    b[numNodes-1]=uExact(std::move(x1[0]));
  }

  printAll("A before communication:\n",A,comm);
  printOne("","",comm);

  printAll("b before communication: ",b,comm);
  printOne("","",comm);

  // communicate vector
  typedef Dune::BufferedCommunicator CommunicatorType;
  CommunicatorType bComm;

  bComm.build(b,b,vectorInterface);
  bComm.forward<AddData<VectorType>>(b,b);
  bComm.backward<CopyData<VectorType>>(b,b);

  // communicate matrix
  CommunicatorType AComm;

  AComm.build(A,A,matrixInterface);
  AComm.forward<AddDataMatrix<StiffnessMatrixType>>(A,A);
  AComm.backward<CopyDataMatrix<StiffnessMatrixType>>(A,A);

  printAll("A after communication:\n",A,comm);
  printOne("","",comm);

  printAll("b after communication: ",b,comm);
  printOne("","",comm);

  // finalize MPI
  MPI_Finalize();

  return 0;

}
