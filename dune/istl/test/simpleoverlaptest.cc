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

// parallel sync print for all the processes
template<typename T,typename C>
inline void printAll(const std::string& str,const T& value,const C& comm)
{
  for(int i=0;i!=comm.size();++i)
    printOne(str,value,comm,i);
}

// parallel sync print of a value for only 1 process (default rank=0)
template<typename T,typename C>
inline void printOne(const std::string& str,const T& value,const C& comm,const int& rank=0)
{
  #ifdef DEBUG_FLAG
  #if DEBUG_FLAG
  if(comm.rank()==rank)
    std::cout<<"[rank "<<rank<<"] "<<str<<value<<std::endl;
  comm.barrier();
  #endif
  #endif
}

// parallel sync print of a vector for only 1 process (default rank=0)
template<typename T,typename C>
void printOne(const std::string& str,const std::vector<T>& v,const C& comm,const int& rank=0)
{
  #ifdef DEBUG_FLAG
  #if DEBUG_FLAG
  if(comm.rank()==rank)
  {
    std::cout<<"[rank "<<rank<<"] "<<str;
    for(auto val:v)
      std::cout<<val<<" ";
    std::cout<<std::endl;
  }
  comm.barrier();
  #endif
  #endif
}

template<typename M,typename C>
void printBMatrix(const std::string& str,const M& matrix,const C& comm)
{
  #ifdef DEBUG_FLAG
  #if DEBUG_FLAG
  typename M::block_type nullblock(0.0);
  for(int i=0;i!=comm.size();++i)
  {
    if(comm.rank()==i)
    {
      std::cout<<"[rank "<<i<<"] "<<str;
      for(std::size_t row=0;row!=matrix.N();++row)
      {
        std::cout<<"row "<<row<<":"<<std::endl;
        for(std::size_t col=0;col!=matrix.M();++col)
        {
          if(matrix.exists(row,col))
            std::cout<<matrix[row][col];
          else
            std::cout<<nullblock;
          std::cout<<std::endl;
        }
      }
    }
    comm.barrier();
  }
  #endif
  #endif
}

int main(int argc,char** argv)
{
  // init MPI
  MPI_Init(&argc,&argv);
  Dune::CollectiveCommunication<MPI_Comm> commColl(MPI_COMM_WORLD);

  // get size and rank
  const int size(commColl.size());
  const int rank(commColl.rank());

  // check if the code is run on 2 processes
  if(size!=2)
    printOne("Error: run the code again with 2 processes!","",commColl);
  else
  {
    // dimmension
    const std::size_t blockDim(BLOCK_DIM);
    const std::size_t dim(rank+2);

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
    if(rank==0)
      x[0]=3;
    else
    {
      x[0]=2;
      x[1]=1;
    }
    printAll("x before communication\n",x,commColl);

    // fill matrix A
    if(rank==0)
    {
      A.setrowsize(0,2);
      A.setrowsize(1,0);
      A.endrowsizes();
      A.addindex(0,0);
      A.addindex(0,1);
      A.endindices();
      A[0][0]=1;
      A[0][1]=2;
    }
    else
    {
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

    // setup communication
    typedef Dune::OwnerOverlapCopyCommunication<size_t> OverlapCommunicationType;
    OverlapCommunicationType commOverlap(MPI_COMM_WORLD);

    typedef Dune::OwnerOverlapCopyAttributeSet AttributeSetType;
    typedef AttributeSetType::AttributeSet Flag;
    typedef Dune::ParallelLocalIndex<Flag> LocalIndexType;

    typedef OverlapCommunicationType::ParallelIndexSet ParallelIndexSetType;
    ParallelIndexSetType& indices(commOverlap.indexSet());

    indices.beginResize();
    if(rank==0)
    {
      indices.add(0,LocalIndexType(0,AttributeSetType::owner,true));
      indices.add(2,LocalIndexType(1,AttributeSetType::copy,true));
    }
    else
    {
      indices.add(1,LocalIndexType(0,AttributeSetType::owner,true));
      indices.add(2,LocalIndexType(1,AttributeSetType::owner,true));
      indices.add(0,LocalIndexType(2,AttributeSetType::copy,true));
    }
    indices.endResize();
    printAll("Parallel index set\n",indices,commColl);

    // remote indices already built
    commOverlap.remoteIndices().rebuild<false>();

    // fill x with the values which need to be copied
    commOverlap.copyOwnerToAll(x,x);
    printAll("x after communication\n",x,commColl);

    // perform the product y=Ax
    Dune::OverlappingSchwarzOperator<MatrixType,VectorType,VectorType,OverlapCommunicationType> schwarzOperator(A,commOverlap);
    schwarzOperator.apply(x,y);
    printAll("y=Ax using Schwarz operator\n",y,commColl);

    // fill y with the values which need to be copied
    commOverlap.copyOwnerToAll(y,y);
    printAll("y after communication\n",y,commColl);
  }

  // finalize MPI
  MPI_Finalize();

  return 0;
}
