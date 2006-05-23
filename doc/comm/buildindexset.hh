// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
// $Id$
#ifndef BUILDINDEXSET_HH
#define BUILDINDEXSET_HH


#include <dune/istl/indexset.hh>
#include <dune/istl/plocalindex.hh>
#include "mpi.h"

/**
 * @brief Flag for marking the indices.
 */
enum Flag {owner, overlap};

// The type of local index we use
typedef Dune::ParallelLocalIndex<Flag> LocalIndex;

/**
 * @brief Add indices to the example index set.
 * @param indexSet The index set to build.
 */
template<class TG, int N>
void build(Dune::ParallelIndexSet<TG,LocalIndex,N>& indexSet)
{
  //
  // The number of processes
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // The rank of our process
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Indicate that we add or remove indices.
  indexSet.beginResize();

  if(rank==0) {
    indexSet.add(0, LocalIndex(0,overlap,true));
    indexSet.add(2, LocalIndex(1,owner,true));
    indexSet.add(6, LocalIndex(2,owner,true));
    indexSet.add(3, LocalIndex(3,owner,true));
    indexSet.add(5, LocalIndex(4,owner,true));
  }

  if(rank==1) {
    indexSet.add(0, LocalIndex(0,owner,true));
    indexSet.add(1, LocalIndex(1,owner,true));
    indexSet.add(7, LocalIndex(2,owner,true));
    indexSet.add(5, LocalIndex(3,overlap,true));
    indexSet.add(4, LocalIndex(4,owner,true));
  }

  // Modification is over
  indexSet.endResize();
}
#endif