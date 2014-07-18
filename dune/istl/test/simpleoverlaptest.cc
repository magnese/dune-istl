#if HAVE_CONFIG_H
#include "config.h"
#endif

// includes
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

// sizes
#define BLOCK_DIM 1
#define NUM_BLOCKS 3

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

  // dimmension
  const size_t blockDim(BLOCK_DIM);
  const size_t numBlocks(NUM_BLOCKS);

  // create vector and matrix
  typedef double FieldType;
  FieldType value(rank*numBlocks+1.0);

  typedef Dune::FieldMatrix<FieldType,blockDim,blockDim> MatrixBlockType;
  typedef Dune::BCRSMatrix<MatrixBlockType> MatrixType;
  MatrixType A(numBlocks,numBlocks,MatrixType::random);

  typedef Dune::FieldVector<FieldType,blockDim> VectorBlockType;
  typedef Dune::BlockVector<VectorBlockType> VectorType;
  VectorType x(numBlocks);

  // fill vector x
  for(size_t i=0;i!= x.size();++i){
    for(size_t j=0;j!=x[i].size();++j) x[i][j]=value;
    value+=1.0;
  }

  printAll("x\n",x,commColl);
  printOne("","",commColl);

  // fill matrix A
  for(size_t i=0;i!=numBlocks;++i) A.setrowsize(i,1); // each row has 1 block
  if(rank!=0) A.incrementrowsize(0);
  A.endrowsizes();
  for(size_t i=0;i!=numBlocks;++i) A.addindex(i,i); // the block is on the diagonal
  if(rank!=0) A.addindex(0,numBlocks-1);
  A.endindices();
  value=rank*numBlocks+1.0;
  for(size_t i=0;i!=numBlocks;++i){
    A[i][i]=value;
    value+=1.0;
  }
  if(rank!=0) A[0][numBlocks-1]=value;

  printBMatrix("A\n",A,commColl);
  printOne("","",commColl);

  // perform y=Ax localy
  VectorType y(numBlocks);

  A.mv(x,y);

  printAll("y=Ax\n",y,commColl);
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
  const size_t numBlocksGlobal((numBlocks-1)*size+1);
  const size_t idxOffset((numBlocks-1)*rank);
  for(size_t i=0;i!=numBlocks;++i){
    for(size_t j=0;j!=numBlocks;++j){
      if(A.exists(i,j)){
        Flag flag(AttributeSetType::owner);
        bool isPublic(false);
        size_t global((i+idxOffset)*numBlocksGlobal+(j+idxOffset));
        if(i==0&&j==0&&rank!=0){
          flag=AttributeSetType::copy;
          isPublic=true;
        }
        if(i==0&&j==(numBlocks-1)&&rank!=0){
          flag=AttributeSetType::overlap;
          isPublic=true;
        }
        indices.add(global,LocalIndexType(i*numBlocks+j,flag,isPublic));
      }
    }
  }
  indices.endResize();

  commOverlap.remoteIndices().rebuild<false>();
  commOverlap.copyOwnerToAll(x,x);

  printAll("x after communication\n",x,commColl);
  printOne("","",commColl);

  Dune::OverlappingSchwarzOperator<MatrixType,VectorType,VectorType,OverlapCommunicationType> schwarzOperator(A,commOverlap);
  schwarzOperator.apply(x,y);

  printAll("y=Ax with Schwarz operator\n",y,commColl);
  printOne("","",commColl);

  // finalize MPI
  MPI_Finalize();
  return 0;

}
