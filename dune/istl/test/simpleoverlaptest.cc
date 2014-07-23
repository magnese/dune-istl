#if HAVE_CONFIG_H
#include "config.h"
#endif

#include <iostream>
#include <vector>
#include <string>

#include <dune/common/fvector.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/enumset.hh>

#include <dune/common/parallel/mpicollectivecommunication.hh>
#include <dune/common/parallel/indexset.hh>
#include <dune/common/parallel/plocalindex.hh>
#include <dune/common/parallel/remoteindices.hh>
#include <dune/common/parallel/interface.hh>
#include <dune/common/parallel/communicator.hh>

#include <dune/istl/matrixmarket.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/schwarz.hh>
#include <dune/istl/owneroverlapcopy.hh>

#define BLOCK_DIM 1
#define DEBUG_FLAG 1

// some printing routines for debugging
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

template<typename M,typename C>
void printBMatrix(std::string&& str,M& matrix,C& comm){
  #ifdef DEBUG_FLAG
  #if DEBUG_FLAG
  typename M::block_type nullmatrix(0.0);
  for(size_t i=0;i!=comm.size();++i){
    if(comm.rank()==i){
      std::cout<<"[rank "<<comm.rank()<<"] "<<str;
      for(size_t row=0;row!=matrix.N();++row){
        std::cout<<"row "<<row<<":"<<std::endl;
        for(size_t col=0;col!=matrix.M();++col){
          if(matrix.exists(row,col)) std::cout<<matrix[row][col];
          else std::cout<<nullmatrix;
          std::cout<<std::endl;
        }
      }
    }
    comm.barrier();
  }
  #endif
  #endif
}

int main(int argc,char** argv){

  // init MPI
  MPI_Init(&argc,&argv);
  Dune::CollectiveCommunication<MPI_Comm> commColl(MPI_COMM_WORLD);

  // get size and rank
  const size_t size(commColl.size());
  const size_t rank(commColl.rank());

  // check if the code is run on 2 processes
  if(size!=2){
    printOne("Error: run the code again with 2 processes!","",commColl);
    MPI_Finalize();
    return 1;
  }

  // dimmension
  const size_t blockDim(BLOCK_DIM);
  const size_t dim(rank+2);

  // create vector and matrix
  typedef int FieldType;

  typedef Dune::FieldMatrix<FieldType,blockDim,blockDim> MatrixBlockType;
  typedef Dune::BCRSMatrix<MatrixBlockType> MatrixType;
  MatrixType A(dim,dim,MatrixType::random);

  typedef Dune::FieldVector<FieldType,blockDim> VectorBlockType;
  typedef Dune::BlockVector<VectorBlockType> VectorType;
  VectorType x(dim);
  VectorType y(dim);

  // fill vector x
  if(rank==0){
    x[0]=3;
  }
  else{
    x[0]=2;
    x[1]=1;
  }

  printAll("x before communication\n",x,commColl);
  printOne("","",commColl);

  // fill matrix A
  if(rank==0){
    A.setrowsize(0,2);
    A.setrowsize(1,0);
    A.endrowsizes();
    A.addindex(0,0);
    A.addindex(0,1);
    A.endindices();
    A[0][0]=1;
    A[0][1]=2;
  }
  else{
    A.setrowsize(0,2);
    A.setrowsize(1,1);
    A.setrowsize(2,2);
    A.endrowsizes();
    A.addindex(0,1);
    A.addindex(0,2);
    A.addindex(1,0);
    A.addindex(2,1);
    A.addindex(2,2);
    A.endindices();
    A[0][1]=3;
    A[0][2]=2;
    A[1][0]=1;
    A[2][1]=2;
    A[2][2]=1;
  }

  printBMatrix("A\n",A,commColl);
  printOne("","",commColl);

  // setup communication
  typedef Dune::OwnerOverlapCopyCommunication<size_t> OverlapCommunicationType;
  OverlapCommunicationType commOverlap(MPI_COMM_WORLD);

  typedef Dune::OwnerOverlapCopyAttributeSet AttributeSetType;
  typedef AttributeSetType::AttributeSet Flag;
  typedef Dune::ParallelLocalIndex<Flag> LocalIndexType;

  typedef OverlapCommunicationType::ParallelIndexSet ParallelIndexSetType;
  ParallelIndexSetType& indices(commOverlap.indexSet());

  indices.beginResize();
  if(rank==0){
    indices.add(0,LocalIndexType(0,AttributeSetType::owner,true));
    indices.add(2,LocalIndexType(1,AttributeSetType::copy,true));
  }
  else{
    indices.add(1,LocalIndexType(0,AttributeSetType::owner,true));
    indices.add(2,LocalIndexType(1,AttributeSetType::owner,true));
    indices.add(0,LocalIndexType(2,AttributeSetType::copy,true));
  }
  indices.endResize();

  printAll("Parallel index set\n",indices,commColl);
  printOne("","",commColl);

  // remote indices already built
  commOverlap.remoteIndices().rebuild<false>();

  // fill x with the values which need to be copied
  commOverlap.copyOwnerToAll(x,x);

  printAll("x after communication\n",x,commColl);
  printOne("","",commColl);

  // perform the product y=Ax
  Dune::OverlappingSchwarzOperator<MatrixType,VectorType,VectorType,OverlapCommunicationType> schwarzOperator(A,commOverlap);
  schwarzOperator.apply(x,y);

  printAll("y=Ax using Schwarz operator\n",y,commColl);
  printOne("","",commColl);

  // fill y with the values which need to be copied
  commOverlap.copyOwnerToAll(y,y);

  printAll("y after communication\n",y,commColl);
  printOne("","",commColl);

  // finalize MPI
  MPI_Finalize();
  return 0;

}
