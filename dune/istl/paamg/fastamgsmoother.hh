// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_FASTAMGSMOOTHER_HH
#define DUNE_ISTL_FASTAMGSMOOTHER_HH

#include <cstddef>

namespace Dune
{
  namespace Amg
  {
    template<int level>
    struct GaussSeidelStepWithDefect
    {
      template<typename M, typename X, typename Y>
      static void forward_apply(const M& A, X& x, Y& d, const Y& b, bool first=false, bool compDef=false)
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

          YBlock v = 0;
          // do upper triangle only if not the first iteration because x is 0 anyway
          if (!first)
          {
            ColIterator colEnd = row->end();
            //skip diagonal and iterate over the rest
            for (++col; col != colEnd; ++col)
            {
              (*col).mmv(x[col.index()],*dIter);
              (*col).umv(x[col.index()],v);
            }
          }

          //either go on the next blocklevel recursively or just solve with diagonal (TMP)
          //GaussSeidelStepWithDefect<level-1>::forward_apply(*diag,*xIter,*dIter,*bIter,first,compDef);
          diag->solve(*xIter,*dIter);

          if (compDef)
          {
            if (first)
              *dIter=0;
            else
              *dIter = v;


            // Update residual for the symmetric case
            for(col=(*row).begin(); col.index()<row.index(); ++col)
              col->mmv(*xIter, d[col.index()]);     //d_j-=A_ij x_i
          }
        }
        if (compDef)
        {
          Y newdef(b);
          typename Y::iterator dit = newdef.begin();
          typename X::iterator xit = x.begin();
          for (RowIterator row = A.begin(); row != A.end(); ++row, ++xit, ++dit)
            for(ColIterator col = row->begin(); col!=row->end(); ++col)
              col->mmv(x[col.index()], *dit);
              //col->mmv(*xit, d[col.index()]);
          for (int i=0; i<newdef.size(); i++)
           // std::cout << newdef[i] << " " << d[i] << std::endl;
            if (std::abs(newdef[i][0]-d[i][0])>1e-4)
              DUNE_THROW(Dune::Exception,"Falschen Defekt berechnet");
        }
      }

      template<typename M, typename X, typename Y>
      static void backward_apply(const M& A, X& x, Y& d,
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

          *dIter-=v; //TODO wieso genau diese Zeile (braucht man das residual denn nun?)

          // Update residual for the symmetric case
          // Skip residual computation as it is not needed.
//           for(col=(*row).begin();col.index()<row.index(); ++col)
//             (*col).mmv(*xIter, d[col.index()]); //d_j-=A_ij x_i
        }
      }
    };

    template<>
    struct GaussSeidelStepWithDefect<0>
    {
      template<typename M, typename X, typename Y>
      static void forward_apply(const M& A, X& x, Y& d, const Y& b, bool first, bool compDef)
      {
        //TODO to reproduce the optimized forward version, this needs to be d, but is this generally okay???
        A.solve(x,d);
      }

      template<typename M, typename X, typename Y>
      static void backward_apply(const M& A, X& x, Y& d, const Y& b, bool first, bool compDef)
      {
        //TODO to reproduce the optimized forward version, this needs to be d, but is this generally okay???
        A.solve(x,d);
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
        if (num_iter == 1)
          GaussSeidelStepWithDefect<M::blocklevel>::forward_apply(A,x,d,b,true,true);
        else
        {
          GaussSeidelStepWithDefect<M::blocklevel>::forward_apply(A,x,d,b,true,false);
          for (int i=0; i<num_iter-2; i++)
            GaussSeidelStepWithDefect<M::blocklevel>::forward_apply(A,x,d,b);
          GaussSeidelStepWithDefect<M::blocklevel>::forward_apply(A,x,d,b,false,true);
        }
      }
    };

    struct GaussSeidelPostsmoothDefect {

      template<typename M, typename X, typename Y>
      static void apply(const M& A, X& x, Y& d,
                        const Y& b, int num_iter)
      {
        for (int i=0; i<num_iter; i++)
          GaussSeidelStepWithDefect<M::blocklevel>::backward_apply(A,x,d,b);
      }
    };


    //! JACOBI SMOOTHING

    template<std::size_t level>
    struct JacobiStepWithDefect
    {
      template<typename M, typename X, typename Y, typename K>
      static void forward_apply(const M& A, X& x, Y& d, const Y& b, const K& w)
      {

      }

      template<typename M, typename X, typename Y, typename K>
      static void backward_apply(const M& A, X& x, Y& d, const Y& b, const K& w)
      {}
    };

    template<>
    struct JacobiStepWithDefect<0>
    {
      template<typename M, typename X, typename Y, typename K>
      static void forward_apply(const M& A, X& x, Y& d, const Y& b, const K& w)
      {}

      template<typename M, typename X, typename Y, typename K>
      static void backward_apply(const M& A, X& x, Y& d, const Y& b, const K& w)
      {}
    };

    struct JacobiPresmoothDefect
    {
      template<typename M, typename X, typename Y, typename K>
      static void apply(const M& A, X& x, Y& d, const Y& b, const K& w, int num_iter)
      {
        for (int i=0; i<num_iter; i++)
          JacobiStepWithDefect<M::blocklevel>::forward_apply(A,x,d,b,w);
      }
    };

    struct JacobiPostsmoothDefect
    {
      template<typename M, typename X, typename Y, typename K>
      static void apply(const M& A, X& x, Y& d, const Y& b, const K& w, int num_iter)
      {
        for (int i=0; i<num_iter; i++)
          JacobiStepWithDefect<M::blocklevel>::backward_apply(A,x,d,b,w);
      }
    };


  } // end namespace Amg
} // end namespace Dune
#endif
