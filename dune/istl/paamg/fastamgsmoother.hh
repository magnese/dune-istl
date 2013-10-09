// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_FASTAMGSMOOTHER_HH
#define DUNE_ISTL_FASTAMGSMOOTHER_HH

#include <cstddef>

namespace Dune
{
  namespace Amg
  {
    template<int l>
    struct GaussSeidelStepWithDefect;
     /** Specialization for Block Level 0 which performs the solve on the FieldType */
    template<>
    struct GaussSeidelStepWithDefect<0>
    {
      template<typename M, typename X, typename Y>
      static void apply(const M& A, X& x, Y& d, const Y& b, bool first, bool compDef)
      {
        A.solve(x,b);
      }
    };

    template<int l>
    struct GaussSeidelStepWithDefect
    {
      template<typename M, typename X, typename Y>
      static void apply(const M& A, X& x, Y& d, const Y& b, bool first=false, bool compDef=false)
      {
        typedef typename M::ConstRowIterator RowIterator;
        typedef typename M::ConstColIterator ColIterator;
        typedef typename Y::block_type YBlock;
        typedef typename X::block_type XBlock;

        typename Y::iterator dIter=d.begin();
        typename Y::const_iterator bIter=b.begin();
        typename X::iterator xIter=x.begin();

        for(RowIterator row=A.begin(), end=A.end(); row != end;
            ++row, ++dIter, ++xIter, ++bIter)
        {
          ColIterator col=(*row).begin();
          *dIter = *bIter;

          for (; col.index()<row.index(); ++col)
            (*col).mmv(x[col.index()],*dIter); // rhs -= sum_{j<i} a_ij * xnew_j

          assert(row.index()==col.index());
          ColIterator diag=col;

          // do upper triangle only if not the first iteration because x is 0 anyway
          if (!first)
          {
            ColIterator colEnd = row->end();
            //skip diagonal and iterate over the rest
            for (++col; col != colEnd; ++col)
              (*col).mmv(x[col.index()],*dIter);
          }

          //either go on the next blocklevel recursively or just solve with diagonal (TMP)
          GaussSeidelStepWithDefect<0>::apply(*diag,*xIter,*dIter,*bIter,first,compDef);

          if (compDef)
          {
            *dIter=0;   //as r=v TODO what?

            // Update residual for the symmetric case
            for(col=(*row).begin(); col.index()<row.index(); ++col)
              col->mmv(*xIter, d[col.index()]);     //d_j-=A_ij x_i
          }
        }
      }
    };

    struct GaussSeidelPresmoothDefect
    {
      template<typename M, typename X, typename Y>
      static void apply(const M& A, X& x, Y& d,
                        const Y& b, int num_iter)
      {
        // perform iterations. These have to know whether they are first and whether to compute a defect.
        // arguments are preferred over template paramters here to reduce compile time/program size
        // In contrast the needed if-clauses are performance irrelevant.
//         if (num_iter == 1)
          GaussSeidelStepWithDefect<M::blocklevel>::apply(A,x,d,b,true,true);
//         else
//         {
//           GaussSeidelStepWithDefect<M::blocklevel>::apply(A,x,d,b,true,false);
//           for (int i=0; i<num_iter-2; i++)
//             GaussSeidelStepWithDefect<M::blocklevel>::apply(A,x,d,b);
//           GaussSeidelStepWithDefect<M::blocklevel>::apply(A,x,d,b,false,true);
//         }
      }
    };

    template<std::size_t level>
    struct GaussSeidelPostsmoothDefect {

      template<typename M, typename X, typename Y>
      static void apply(const M& A, X& x, Y& d,
                        const Y& b)
      {
        typedef typename M::ConstRowIterator RowIterator;
        typedef typename M::ConstColIterator ColIterator;
        typedef typename Y::block_type YBlock;
        typedef typename X::block_type XBlock;

        typename Y::iterator dIter=d.beforeEnd();
        typename X::iterator xIter=x.beforeEnd();
        typename Y::const_iterator bIter=b.beforeEnd();

        for(RowIterator row=A.beforeEnd(), end=A.beforeBegin(); row != end;
            --row, --dIter, --xIter, --bIter)
        {
          ColIterator endCol=(*row).beforeBegin();
          ColIterator col=(*row).beforeEnd();
          *dIter = *bIter;

          for (; col.index()>row.index(); --col)
            (*col).mmv(x[col.index()],*dIter);     // rhs -= sum_{i>j} a_ij * xnew_j
          assert(row.index()==col.index());
          ColIterator diag=col;
          YBlock v=*dIter;
          // upper diagonal matrix
          for (--col; col!=endCol; --col)
            (*col).mmv(x[col.index()],v);     // v -= sum_{j<i} a_ij * xold_j

          // Not recursive yet. Just solve with the diagonal
          diag->solve(*xIter,v);

          *dIter-=v;

          // Update residual for the symmetric case
          // Skip residual computation as it is not needed.
          //for(col=(*row).begin();col.index()<row.index(); ++col)
          //col.mmv(*xIter, d[col.index()]); //d_j-=A_ij x_i
        }
      }
    };
  } // end namespace Amg
} // end namespace Dune
#endif
