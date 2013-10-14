// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_FASTAMGSMOOTHER_HH
#define DUNE_ISTL_FASTAMGSMOOTHER_HH

#include <cstddef>

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

    // implements "concept-like" behaviour depending on the property value p
    template<class S, bool p>
    struct SmootherWithDefectHelper : public S
    {
      template<typename... A>
      SmootherWithDefectHelper(A... args) : S(args...)
      {}
    };

    template<class S>
    struct SmootherWithDefectHelper<S,false> : public S
    {
      template<typename... A>
      SmootherWithDefectHelper(A... args) : S(args...)
      {}

      template<typename M, typename X, typename Y>
      void preApply(const M& A, X& x, Y& d, const Y& b)
      {
        // apply the preconditioner
        this->apply(x,b);

        //defect calculation
        d = b;
        typedef typename M::ConstRowIterator RowIterator;
        typedef typename M::ConstColIterator ColIterator;
        typename Y::const_iterator xIter = x.begin();
        for(RowIterator row=A.begin(), end=A.end(); row != end; ++row, ++xIter)
          for (ColIterator col = row->begin(), cEnd = row->end(); col != cEnd; ++col)
            col->mmv(*xIter,d[col.index()]);
      }

      template<typename M, typename X, typename Y>
      void postApply(const M& A, X& x, Y& d, const Y& b)
      {
        this->apply(x,b);
      }
    };

    /** @brief helper class to use normal smoothers with fastamg
     * For smoothers that do defect calculation (those that have SmootherCalculatesDefect<S>::value==1)
     * this class just mimics the smoother. For other smoothers, this adds methods preApply() and
     * postApply() as expected by fastamg that will do the defect calculation.
     */
    template<class S>
    struct SmootherWithDefect : public SmootherWithDefectHelper<S,SmootherCalculatesDefect<S>::value >
    {
      template<typename... A>
      SmootherWithDefect(A... args) : SmootherWithDefectHelper<S,SmootherCalculatesDefect<S>::value >(args...)
      {}
    };

    template<typename S>
    struct ConstructionTraits<SmootherWithDefect<S> >
    {
      typedef typename ConstructionTraits<S>::Arguments Arguments;

      static inline SmootherWithDefect<S>* construct(Arguments& args)
      {
        return static_cast<SmootherWithDefect<S>*>(ConstructionTraits<S>::construct(args));
      }

      static inline void deconstruct(S* prec)
      {
        ConstructionTraits<S>::deconstruct(prec);
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
          YBlock v = 0;
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
      GaussSeidelWithDefect(const M& A, int num_iter_) : A_(A),num_iter__(num_iter_)
      {}

      enum {
        category = SolverCategory::sequential
      };

      void preApply(const M& A, X& x, Y& d, const Y& b)
      {
        // perform iterations. These have to know whether they are first
        // and whether to compute a defect.
        int num_iter = 3;
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

      void postApply(const M& A, X& x, Y& d,
                        const Y& b)
      {
        int num_iter=3;
        for (int i=0; i<num_iter; i++)
          GaussSeidelStepWithDefect<M::blocklevel>::backward_apply(A,x,d,b);
      }

      private:
      const M& A_;
      int num_iter__;
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

    struct JacobiPresmoothDefect
    {
      template<typename M, typename X, typename Y, typename K>
      static void apply(const M& A, X& x, Y& d, const Y& b, const K& w, int num_iter)
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
//TODO this cannot work. we need an object for ilu smoothing
//     template<typename M>
//     class ILUSmoothDefect
//     {
//       public:
//
//
//       template<typename X, typename Y>
//       static void apply(const M& A, X& x, Y& d, const Y& b)
//       {
//         if (ilu_decomp.find(&A) == ilu_decomp.end())
//         {
//           ilu_decomp.insert(std::make_pair(&A, new M(A)));
//           bilu0_decomposition(*ilu_decomp[&A]);
//         }
//       }
//
//       private:
//       std::map<const M*,M*> ilu_decomp;
//     };


  } // end namespace Amg
} // end namespace Dune
#endif
