// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_FASTAMGSMOOTHER_HH
#define DUNE_ISTL_FASTAMGSMOOTHER_HH

#include <cstddef>

#include "smoother.hh"

namespace Dune
{
  namespace Amg
  {
    template<typename S>
    struct SmootherCalculatesDefect
    {
      enum
      {
        /** true if given smoother has the interface and functionality
         * to directly calculate the defects
         */
        value = false
      };
    };

    template<class S, bool p>
    class SmootherWithDefectHelper
    {
      public:
      typedef typename S::matrix_type matrix_type;
      typedef typename S::matrix_type Matrix;
      typedef typename S::domain_type Domain;
      typedef typename S::range_type Range;

      typedef typename S::RecommendedCoarseSmoother RecommendedCoarseSmoother;

      SmootherWithDefectHelper(typename ConstructionTraits<S>::Arguments& args)
      {
        smoother = ConstructionTraits<S>::construct(args);
      }

      ~SmootherWithDefectHelper()
      {
        ConstructionTraits<S>::deconstruct(smoother);
      }

       void preApply(Domain& x, Range& d, const Range& b)
      {
        // apply the preconditioner
        smoother->preApply(x,d,b);
      }

      void postApply(Domain& x, Range& d, const Range& b)
      {
        smoother->postApply(x,d,b);
      }

      private:
      S* smoother;
    };

    template<class S>
    class SmootherWithDefectHelper<S,false>
    {
      public:
      typedef typename S::matrix_type matrix_type;
      typedef typename S::matrix_type Matrix;
      typedef typename S::domain_type Domain;
      typedef typename S::range_type Range;
      typedef S RecommendedCoarseSmoother;

      SmootherWithDefectHelper(typename ConstructionTraits<S>::Arguments& args)
        : A(args.getMatrix())
      {
        smoother = ConstructionTraits<S>::construct(args);
      }

      ~SmootherWithDefectHelper()
      {
        ConstructionTraits<S>::deconstruct(smoother);
      }

      void preApply(Domain& x, Range& d, const Range& b)
      {
        // apply the preconditioner
        SmootherApplier<S>::preSmooth(*smoother,x,b);

        //defect calculation
        d = b;
        typedef typename Matrix::ConstRowIterator RowIterator;
        typedef typename Matrix::ConstColIterator ColIterator;
        typename Range::const_iterator xIter = x.begin();
        for(RowIterator row=A.begin(), end=A.end(); row != end; ++row, ++xIter)
          for (ColIterator col = row->begin(), cEnd = row->end(); col != cEnd; ++col)
            col->mmv(*xIter,d[col.index()]);
      }

      void postApply(Domain& x, Range& d, const Range& b)
      {
        SmootherApplier<S>::postSmooth(*smoother,x,b);
      }

      private:
      const Matrix& A;
      S* smoother;
    };

    /** @brief helper class to use normal smoothers with fastamg
     * For smoothers that do defect calculation (those that have SmootherCalculatesDefect<S>::value==1)
     * this class just mimics the smoother. For other smoothers, this adds methods preApply() and
     * postApply() as expected by fastamg that will do the defect calculation.
     */
    template<class S>
    struct SmootherWithDefect : public SmootherWithDefectHelper<S,SmootherCalculatesDefect<S>::value >
    {};

    template<typename S>
    struct ConstructionTraits<SmootherWithDefect<S> >
    {
      typedef typename ConstructionTraits<S>::Arguments Arguments;

      static inline SmootherWithDefect<S>* construct(Arguments& args)
      {
        return static_cast<SmootherWithDefect<S>*>(new SmootherWithDefectHelper<S,SmootherCalculatesDefect<S>::value>(args));
      }

      static inline void deconstruct(SmootherWithDefect<S>* obj)
      {
        delete obj;
      }
    };

    template<typename S>
    struct SmootherTraits<SmootherWithDefect<S> >
    {
      typedef typename SmootherTraits<S>::Arguments Arguments;
    };

    template<int level>
    struct GaussSeidelStepWithDefect
    {
      template<typename M, typename X, typename Y>
      static void forward_apply(const M& A, X& x, Y& d, const Y& b, bool first=false, bool compDef=false)
      {
        typedef typename M::ConstRowIterator RowIterator;
        typedef typename M::ConstColIterator ColIterator;
        typedef typename Y::block_type YBlock;

        typename Y::iterator dIter=d.begin();
        typename Y::const_iterator bIter=b.begin();
        typename X::iterator xIter=x.begin();

        for(RowIterator row=A.begin(), end=A.end(); row != end;
            ++row, ++dIter, ++xIter, ++bIter)
        {
          *dIter = *bIter;

          // do lower triangular matrix
          ColIterator col=(*row).begin();
          for (; col.index()<row.index(); ++col)
            (*col).mmv(x[col.index()],*dIter); // d -= sum_{j<i} a_ij * xnew_j

          ColIterator diag=col;

          // do upper triangular matrix only if not the first iteration
          // because x would be 0 anyway, store result in v for latter use
          YBlock v(0.0);
          if (!first)
          {
            //skip diagonal and iterate over the rest
            ColIterator colEnd = row->end();
            for (++col; col != colEnd; ++col)
              (*col).umv(x[col.index()],v);
          }
          *dIter -= v;

          // TODO either go on the next blocklevel recursively or just solve with diagonal (TMP)
          //GaussSeidelStepWithDefect<level-1>::forward_apply(*diag,*xIter,*dIter,*bIter,first,compDef);
          diag->solve(*xIter,*dIter);

          // compute defect if necessary
          if (compDef)
          {
            *dIter = v;

            // Update residual for the symmetric case
            // (uses A_ji=A_ij for memory access efficiency)
            for(col=(*row).begin(); col.index()<row.index(); ++col)
              col->mmv(*xIter, d[col.index()]);     //d_j-=A_ij x_i
          }
        }
      }

      template<typename M, typename X, typename Y>
      static void backward_apply(const M& A, X& x, Y& d,
                        const Y& b)
      {
        typedef typename M::ConstRowIterator RowIterator;
        typedef typename M::ConstColIterator ColIterator;

        typename Y::iterator dIter=d.beforeEnd();
        typename X::iterator xIter=x.beforeEnd();
        typename Y::const_iterator bIter=b.beforeEnd();

        for(RowIterator row=A.beforeEnd(), end=A.beforeBegin(); row != end;
            --row, --dIter, --xIter, --bIter)
        {
          *dIter = *bIter;

          // do lower triangular matrix
          ColIterator endCol=(*row).beforeBegin();
          ColIterator col=(*row).beforeEnd();
          for (; col.index()>row.index(); --col)
            (*col).mmv(x[col.index()],*dIter);     // d -= sum_{i>j} a_ij * xnew_j

          ColIterator diag=col;

          // do upper triangular matrix manually skipping diagonal
          for (--col; col!=endCol; --col)
            (*col).mmv(x[col.index()],*dIter);     // d -= sum_{j<i} a_ij * xold_j

          // TODO Not recursive yet. Just solve with the diagonal
          diag->solve(*xIter,*dIter);
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

    template<typename M, typename X, typename Y>
    class GaussSeidelWithDefect
    {
      public:
      typedef M matrix_type;
      typedef M Matrix;
      typedef typename X::field_type field_type;
      typedef typename X::field_type Field;
      typedef X domain_type;
      typedef X Domain;
      typedef Y range_type;
      typedef Y Range;

      typedef typename Dune::SeqGS<M,X,Y> RecommendedCoarseSmoother;

      GaussSeidelWithDefect(const M& A_, int num_iter_) : A(A_),num_iter(num_iter_)
      {}

      enum {
        category = SolverCategory::sequential
      };

      void preApply(X& x, Y& d, const Y& b)
      {
        // perform iterations. These have to know whether they are first
        // and whether to compute a defect.
        if (num_iter == 1)
          GaussSeidelStepWithDefect<M::blocklevel>::forward_apply(A,x,d,b,true,true);
        else
        {
          GaussSeidelStepWithDefect<M::blocklevel>::forward_apply(A,x,d,b,true,false);
          for (int i=0; i<num_iter-2; i++)
            GaussSeidelStepWithDefect<M::blocklevel>::forward_apply(A,x,d,b, false,false);
          GaussSeidelStepWithDefect<M::blocklevel>::forward_apply(A,x,d,b,false,true);
        }
      }

      void postApply(X& x, Y& d,
                        const Y& b)
      {
        for (int i=0; i<num_iter; i++)
          GaussSeidelStepWithDefect<M::blocklevel>::backward_apply(A,x,d,b);
      }

      private:
      const M& A;
      int num_iter;
    };

    template<typename M, typename X, typename Y>
    struct ConstructionTraits<GaussSeidelWithDefect<M,X,Y> >
    {
      typedef DefaultConstructionArgs<GaussSeidelWithDefect<M,X,Y> > Arguments;

      static inline GaussSeidelWithDefect<M,X,Y>* construct(Arguments& args)
      {
        return new GaussSeidelWithDefect<M,X,Y>(args.getMatrix(),args.getArgs().iterations);
      }

      static void deconstruct(GaussSeidelWithDefect<M,X,Y>* obj)
      {
        delete obj;
      }
    };

    template<typename M,typename X,typename Y>
    struct SmootherTraits<GaussSeidelWithDefect<M,X,Y> >
    {
      typedef DefaultSmootherArgs<typename M::field_type> Arguments;
    };

    template<typename M,typename X,typename Y>
    struct SmootherCalculatesDefect<GaussSeidelWithDefect<M,X,Y> >
    {
      enum {
        value = true
      };
    };



    //! JACOBI SMOOTHING
//TODO adjust jacobi to the new style!
    template<std::size_t level>
    struct JacobiStepWithDefect
    {
      template<typename M, typename X, typename Y, typename K>
      static void forward_apply(const M& A, X& x, Y& d, const Y& b, const K& w, bool first, bool compDef)
      {
        typedef typename M::ConstRowIterator RowIterator;
        typedef typename M::ConstColIterator ColIterator;
        typedef typename Y::block_type YBlock;

        typename Y::const_iterator bIter=b.begin();
        typename X::iterator xIter=x.begin();

        if (compDef)
          d=b;

        if (first)
        {
          for(RowIterator row=A.begin(), end=A.end(); row != end;
            ++row, ++bIter, ++xIter)
          {
            (*row)[row.index()].solve(*xIter,*bIter);
            if (compDef)
            {
              ColIterator col = row->begin();
              ColIterator colEnd = row->end();
              for (; col != colEnd; ++col)
              {
                col->mmv(*xIter,d[col.index()]);
                *xIter *= w;
              }
            }
          }
        }
        else
        {
          // a jacobi-intrinsic problem: we need the entire old x until the end of the computation
          const X xold(x);

          for(RowIterator row=A.begin(), end=A.end(); row != end;
            ++row, ++xIter, ++bIter)
          {
            YBlock r = *bIter;

            ColIterator col = row->begin();
            ColIterator colEnd = row->end();
            for (; col.index() < row.index(); ++col)
              col->mmv(xold[col.index()],r);

            ColIterator diag = col;

            for (++col; col != colEnd; ++col)
              col->mmv(xold[col.index()],r);

            //TODO recursion
            diag->solve(*xIter,r);

            //damping
            *xIter *= 1-w;
            xIter->axpy(w,xold[row.index()]);

            if (compDef)
            {
              col = row->begin();
              for (; col != colEnd; ++col)
                col->mmv(*xIter,d[col.index()]);
            }
          }
          x *= w;
          x.axpy(K(1)-w,xold);
        }
      }

      template<typename M, typename X, typename Y, typename K>
      static void backward_apply(const M& A, X& x, Y& d, const Y& b, const K& w)
      {
        typedef typename M::ConstRowIterator RowIterator;
        typedef typename M::ConstColIterator ColIterator;
        typedef typename Y::block_type YBlock;

        typename X::iterator xIter=x.beforeEnd();
        typename Y::const_iterator bIter=b.beforeEnd();

        // a jacobi-intrinsic problem: we need the entire old x until the end of the computation
        const X xold(x);

        for(RowIterator row=A.beforeEnd(), end=A.beforeBegin(); row != end;
          --row, --xIter, --bIter)
        {
          YBlock r = *bIter;
          ColIterator endCol=(*row).beforeBegin();
          ColIterator col=(*row).beforeEnd();
          for (; col.index()>row.index(); --col)
            (*col).mmv(xold[col.index()],r);     // d -= sum_{i>j} a_ij * xnew_j

          ColIterator diag=col;

          for (--col; col!=endCol; --col)
            (*col).mmv(xold[col.index()],r);     // d -= sum_{j<i} a_ij * xold_j

          diag->solve(*xIter,r);
        }
        x *= w;
        x.axpy(K(1)-w,xold);
      }
    };

    template<typename M, typename X, typename Y>
    class JacobiWithDefect
    {
      public:
      typedef M matrix_type;
      typedef M Matrix;
      typedef typename M::field_type field_type;
      typedef typename M::field_type Field;
      typedef X domain_type;
      typedef X Domain;
      typedef Y range_type;
      typedef Y Range;

      typedef typename Dune::SeqJac<M,X,Y> RecommendedCoarseSmoother;

      JacobiWithDefect(const M& A_, int num_iter_, Field w_) : A(A_), num_iter(num_iter_), w(w_)
      {}

      enum {
        category = SolverCategory::sequential
      };


      void preApply(X& x, Y& d, const Y& b)
      {
        if (num_iter == 1)
          JacobiStepWithDefect<M::blocklevel>::forward_apply(A,x,d,b,w,true,true);
        else
        {
          JacobiStepWithDefect<M::blocklevel>::forward_apply(A,x,d,b,w,true,false);
          for (int i=0; i<num_iter-2; i++)
            JacobiStepWithDefect<M::blocklevel>::forward_apply(A,x,d,b,w,false,false);
          JacobiStepWithDefect<M::blocklevel>::forward_apply(A,x,d,b,w,false,true);
        }
      }

      void postApply(X& x, Y& d, const Y& b)
      {
        for (int i=0; i<num_iter; i++)
          JacobiStepWithDefect<M::blocklevel>::backward_apply(A,x,d,b,w);
      }

      private:
      const M& A;
      int num_iter;
      Field w;
    };

    template<typename M, typename X, typename Y>
    struct ConstructionTraits<JacobiWithDefect<M,X,Y> >
    {
      typedef DefaultConstructionArgs<JacobiWithDefect<M,X,Y> > Arguments;

      static inline JacobiWithDefect<M,X,Y>* construct(Arguments& args)
      {
        return new JacobiWithDefect<M,X,Y>(args.getMatrix(),args.getArgs().iterations,args.getArgs().relaxationFactor);
      }

      static void deconstruct(JacobiWithDefect<M,X,Y>* obj)
      {
        delete obj;
      }
    };

    template<typename M,typename X,typename Y>
    struct SmootherTraits<JacobiWithDefect<M,X,Y> >
    {
      typedef DefaultSmootherArgs<typename M::field_type> Arguments;
    };

    template<typename M,typename X,typename Y>
    struct SmootherCalculatesDefect<JacobiWithDefect<M,X,Y> >
    {
      enum {
        value = true
      };
    };

    template<typename M, typename X, typename Y>
    class ILUnWithDefect
    {
      public:
      typedef M matrix_type;
      typedef M Matrix;
      typedef typename X::field_type field_type;
      typedef typename X::field_type Field;
      typedef X domain_type;
      typedef X Domain;
      typedef Y range_type;
      typedef Y Range;

      typedef typename Dune::SeqILUn<M,X,Y> RecommendedCoarseSmoother;

      enum {
        category = SolverCategory::sequential
      };


      ILUnWithDefect(const M& A_, int n, Field w_) : A(A_), decomp(A.N(), A.M(),M::row_wise), w(w_)
      {
        if (n==0)
        {
          decomp = A;
          bilu0_decomposition(decomp);
        }
        else
          bilu_decomposition(A,n,decomp);
      }

      void preApply(X& x, Y& d, const Y& b)
      {
        // iterator types
        typedef typename M::ConstRowIterator rowiterator;
        typedef typename M::ConstColIterator coliterator;
        typedef typename Y::block_type Yblock;
        typedef typename X::block_type Xblock;

        // lower triangular solve
        rowiterator endi=A.end();
        for (rowiterator row=A.begin(); row!=endi; ++row)
        {
          Yblock rhs(b[row.index()]);
          for (coliterator col=row->begin(); col.index()<row.index(); ++col)
            col->mmv(x[col.index()],rhs);
          x[row.index()] = rhs;           // Lii = I
        }

        // upper triangular solve
        rowiterator rendi=A.beforeBegin();
        for (rowiterator row=A.beforeEnd(); row!=rendi; --row)
        {
          Xblock rhs(x[row.index()]);
          coliterator col;
          for (col=row->beforeEnd(); col.index()>row.index(); --col)
            col->mmv(x[col.index()],rhs);
          x[row.index()] = 0;
            col->umv(rhs,x[row.index()]);           // diagonal stores inverse!

          col = row->begin();
          coliterator colEnd = row->end();
          for (; col != colEnd; ++col)
            col->mmv(x[row.index()],d[col.index()]);
          //TODO introduce iterator over x
        }
      }

      void postApply(X& x, Y& d, const Y& b)
      {
        bilu_backsolve(decomp,x,b);
      }

      private:
      const M& A;
      M decomp;
      Field w;
    };

    template<typename M, typename X, typename Y>
    struct ConstructionTraits<ILUnWithDefect<M,X,Y> >
    {
      typedef ConstructionArgs<SeqILUn<M,X,Y> > Arguments;

      static inline ILUnWithDefect<M,X,Y>* construct(Arguments& args)
      {
        return new ILUnWithDefect<M,X,Y>(args.getMatrix(), args.getN(), args.getArgs().relaxationFactor);
      }

      static void deconstruct(ILUnWithDefect<M,X,Y>* obj)
      {
        delete obj;
      }
    };

    template<typename M, typename X, typename Y>
    struct SmootherTraits<ILUnWithDefect<M,X,Y> >
    {
      typedef DefaultSmootherArgs<typename M::field_type> Arguments;
    };

    template<typename M, typename X, typename Y>
    struct SmootherCalculatesDefect<ILUnWithDefect<M,X,Y> >
    {
      enum {
        value = true
      };
    };

  } // end namespace Amg
} // end namespace Dune
#endif
