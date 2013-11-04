// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
// $Id$
#ifndef DUNE_GALERKIN_HH
#define DUNE_GALERKIN_HH

#include "aggregates.hh"
#include "pinfo.hh"
#include <dune/common/poolallocator.hh>
#include <dune/common/enumset.hh>
#include <set>
#include <limits>
#include <algorithm>

namespace Dune
{
  namespace Amg
  {
    /**
     * @addtogroup ISTL_PAAMG
     *
     * @{
     */
    /** @file
     * @author Markus Blatt
     * @brief Provides a class for building the galerkin product
     * based on a aggregation scheme.
     */

    class GalerkinProduct
    {
    public:
      /**
       * @brief Calculate the galerkin product.
       * @param fine The fine matrix.
       * @param aggregates The aggregate mapping.
       * @param coarse The coarse Matrix.
       * @param pinfo Parallel information about the fine level.
       * @param copy The attribute set identifying the copy nodes of the graph.
       */
      template<class MF, class V, class MC, class I, class O>
      void calculate(const MF& fine, const AggregatesMap<V>& aggregates, MC& coarse,
                     const I& pinfo, const O& copy)
      {
        typedef typename MF::ConstIterator RowIterator;
        RowIterator endRow = fine.end();

        for(RowIterator row = fine.begin(); row != endRow; ++row)
          if(aggregates[row.index()] != AggregatesMap<V>::ISOLATED)
          {
            assert(aggregates[row.index()]!=AggregatesMap<V>::UNAGGREGATED);
            typedef typename MF::ConstColIterator ColIterator;
            ColIterator endCol = row->end();

            for(ColIterator col = row->begin(); col != endCol; ++col)
              if(aggregates[col.index()] != AggregatesMap<V>::ISOLATED)
              {
                assert(aggregates[row.index()]!=AggregatesMap<V>::UNAGGREGATED);
                coarse[aggregates[row.index()]][aggregates[col.index()]]+=*col;
              }
          }

        // get the right diagonal matrix values on copy lines from owner processes
        typedef typename MF::block_type BlockType;
        std::vector<BlockType> rowsize(coarse.N(),BlockType(0));
        for (typename MF::size_type row = 0; row < coarse.N(); ++row)
          rowsize[row] = coarse[row][row];
        pinfo.copyOwnerToAll(rowsize,rowsize);
        for (typename MF::size_type row = 0; row < coarse.N(); ++row)
          coarse[row][row] = rowsize[row];
      }
    };
  } // namespace Amg
} // namespace Dune
#endif
