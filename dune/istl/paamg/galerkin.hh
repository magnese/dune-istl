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

    /** @brief A wrapper to treat with the BCRSMatrix build mode during Build Stage
     * @tparam M the matrix typedef
     * The used build mode of Dune::BCRSMatrix handles matrices different during
     * assembly and afterwards. As we want to do both with the same implementation
     * a wrapper is needed during the first assembly of the matrix hierarchy.
     */
    template<class M>
    class BuildModeWrapper
    {
    public:
      typedef typename M::ConstIterator ConstIterator;
      typedef typename M::ConstColIterator ConstColIterator;
      typedef typename M::block_type block_type;
      typedef typename M::size_type size_type;

      class row_object
      {
      public:
        row_object(M& m, size_type i) : _m(m), _i(i) {}

        block_type& operator[](size_type j)
        {
          return _m.entry(_i,j);
        }
      private:
        M& _m;
        size_type _i;
      };

      BuildModeWrapper(M& m) : _m(m) {}

      row_object operator[](size_type i) const
      {
        return row_object(_m,i);
      }

      size_type N() const
      {
        return _m.N();
      }

      ConstIterator begin() const
      {
        return _m.begin();
      }

      ConstIterator end() const
      {
        return _m.end();
      }

    private:
      M& _m;
    };

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
                     const I& pinfo, const O& copy);
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
        for (RowIterator row = coarse.begin(); row != coarse.end(); ++row)
          rowsize[row.index()]=coarse[row.index()][row.index()];
        pinfo.copyOwnerToAll(rowsize,rowsize);
        for (RowIterator row = coarse.begin(); row != coarse.end(); ++row)
          coarse[row.index()][row.index()] = rowsize[row.index()];
      }
    };
  } // namespace Amg
} // namespace Dune
#endif
