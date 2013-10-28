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
      template<class M, class V, class I, class O>
      void calculate(const M& fine, const AggregatesMap<V>& aggregates, M& coarse,
                     const I& pinfo, const O& copy);


        // don't set dirichlet boundaries for copy lines to make novlp case work,
        // the preconditioner yields slightly different results now.

        // Set the dirichlet border
        //DirichletBoundarySetter<P>::template set<M>(coarse, pinfo, copy);
      }
    };

    template<class T>
    struct DirichletBoundarySetter
    {
      template<class M, class O>
      static void set(M& coarse, const T& pinfo, const O& copy);
    };

    template<>
    struct DirichletBoundarySetter<SequentialInformation>
    {
      template<class M, class O>
      static void set(M& coarse, const SequentialInformation& pinfo, const O& copy);
    };

    template<class T>
    template<class M, class O>
    void DirichletBoundarySetter<T>::set(M& coarse, const T& pinfo, const O& copy)
    {
      typedef typename T::ParallelIndexSet::const_iterator ConstIterator;
      ConstIterator end = pinfo.indexSet().end();
      typedef typename M::block_type Block;
      Block identity=Block(0.0);
      for(typename Block::RowIterator b=identity.begin(); b !=  identity.end(); ++b)
        b->operator[](b.index())=1.0;

      for(ConstIterator index = pinfo.indexSet().begin();
          index != end; ++index) {
        if(copy.contains(index->local().attribute())) {
          typedef typename M::ColIterator ColIterator;
          typedef typename M::row_type Row;
          Row row = coarse[index->local()];
          ColIterator cend = row.find(index->local());
          ColIterator col  = row.begin();
          for(; col != cend; ++col)
            *col = 0;

          cend = row.end();

          assert(col != cend); // There should be a diagonal entry
          *col = identity;

          for(++col; col != cend; ++col)
            *col = 0;
        }
      }
    }

    template<class M, class O>
    void DirichletBoundarySetter<SequentialInformation>::set(M& coarse,
                                                             const SequentialInformation& pinfo,
                                                             const O& overlap)
    {}

  } // namespace Amg
} // namespace Dune
#endif
