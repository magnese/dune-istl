// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_FASTAMGSMOOTHER_HH
#define DUNE_ISTL_FASTAMGSMOOTHER_HH

#include <dune/istl/paamg/smoother.hh>

namespace Dune
{
  namespace Amg
  {
    /** @file
     *  @brief Defines special smoothers that are able to calculate defects
     *  @author Dominic Kempf
     *
     * A new interface to be used with FastAMG is introduced. A compatibility
     * class SmootherWithDefect<S> allows to construct a smoother from a smoother
     * implementing the preconditioner interface. Special smoothers are implemented
     * based on observations how the defect can be calculated during smoothing in
     * the case of symmetric matrices. This results in time savings due to they
     * reduced memory access.
     */

    template<typename S>
    struct SmootherCalculatesDefect
    {
      enum
      {
        /** @brief whether smoother calculates defects
         *
         * true if given smoother has the interface and functionality
         * to directly calculate the defects
         */
        value = false
      };
    };

    struct GenericDefect
    {
      /** @brief calculate a defect from given x and b */
      template<typename M, typename X, typename Y>
      static void calculate(const M& A, const X& x, Y& d, const Y& b)
      {
        typedef typename M::ConstRowIterator RowIterator;
        typedef typename M::ConstColIterator ColIterator;
        typename Y::iterator dIter = d.begin();
        typename Y::const_iterator bIter = b.begin();
        for(RowIterator row=A.begin(), end=A.end(); row != end; ++row, ++dIter, ++bIter)
        {
          *dIter = *bIter;
          for (ColIterator col = row->begin(), cEnd = row->end(); col != cEnd; ++col)
            col->mmv(x[col.index()],*dIter); //d_i -= a-ij * x_j
        }
      }
    };

    //! apply the smoothers that calculate defects: calls are only forwarded
    template<class S, bool p>
    struct SmootherWithDefectHelper
    {
      typedef typename S::RecommendedCoarseSmoother RecommendedCoarseSmoother;

      static void preApply(S& smoother, const typename S::matrix_type&, typename S::domain_type& x, typename S::range_type& d, const typename S::range_type& b)
      {
        smoother.preApply(x,d,b);
      }

      static void postApply(S& smoother, const typename S::matrix_type& A, typename S::domain_type& x, typename S::range_type& d, const typename S::range_type& b)
      {
        smoother.postApply(x,d,b);
      }
    };

    //! apply smoothers that dont calculate defects: generic defect calculation is added
    template<class S>
    struct SmootherWithDefectHelper<S,false>
    {
      typedef S RecommendedCoarseSmoother;

      static void preApply(S& smoother, const typename S::matrix_type& A, typename S::domain_type& x, typename S::range_type& d, const typename S::range_type& b)
      {
        smoother.apply(x,b);
        GenericDefect::calculate(A,x,d,b);
      }

      static void postApply(S& smoother, const typename S::matrix_type&, typename S::domain_type& x, typename S::range_type& d, const typename S::range_type& b)
      {
        smoother.apply(x,b);
      }
    };

    //! ILU needs specialization to work with fastamg
    template<typename M, typename X, typename Y>
    struct SmootherWithDefectHelper<SeqILU0<M,X,Y>,false>
    {
      typedef SeqILU0<M,X,Y> S;
      typedef SeqILU0<M,X,Y> RecommendedCoarseSmoother;

      static void preApply(S& smoother, const typename S::matrix_type& A, typename S::domain_type& x, typename S::range_type& d, const typename S::range_type& b)
      {
        smoother.apply(x,b);
        GenericDefect::calculate(A,x,d,b);
      }

      static void postApply(S& smoother, const typename S::matrix_type& A, typename S::domain_type& x, typename S::range_type& d, const typename S::range_type& b)
      {
        // calculate the defect
        GenericDefect::calculate(A,x,d,b);

        typename S::domain_type v(x);
        SmootherApplier<SeqILU0<M,X,Y> >::postSmooth(*smoother,v,d);
        x += v;
      }
    };

    //! ILU needs specialization to work with fastamg
    template<typename M, typename X, typename Y>
    struct SmootherWithDefectHelper<SeqILUn<M,X,Y>,false>
    {
      typedef SeqILUn<M,X,Y> S;
      typedef SeqILUn<M,X,Y> RecommendedCoarseSmoother;

      static void preApply(S& smoother, const typename S::matrix_type& A, typename S::domain_type& x, typename S::range_type& d, const typename S::range_type& b)
      {
        smoother.apply(x,b);
        GenericDefect::calculate(A,x,d,b);
      }

      static void postApply(S& smoother, const typename S::matrix_type& A, typename S::domain_type& x, typename S::range_type& d, const typename S::range_type& b)
      {
        // calculate the defect
        GenericDefect::calculate(A,x,d,b);

        typename S::domain_type v(x);
        SmootherApplier<SeqILU0<M,X,Y> >::postSmooth(*smoother,v,d);
        x += v;
      }
    };

    /** @brief helper class to use normal smoothers with fastamg
     *  @tparam S the smoother to be wrapped
     * Wrapper class around the smoother, that fulfills the interface required from
     * FastAMG for a smoother. If S itself fulfills this interface, no functionality is
     * added. If not, the calls are forwarded to the preconditioner interface and generic
     * defect calculation is added.
     */
    template<class S>
    struct SmootherWithDefect
    {
      public:
      typedef typename S::matrix_type matrix_type;
      typedef typename S::matrix_type Matrix;
      typedef typename S::domain_type Domain;
      typedef typename S::range_type Range;

      typedef typename SmootherWithDefectHelper<S,SmootherCalculatesDefect<S>::value>
        ::RecommendedCoarseSmoother RecommendedCoarseSmoother;

      SmootherWithDefect(typename ConstructionTraits<S>::Arguments& args)
        : A(args.getMatrix()), smoother(ConstructionTraits<S>::construct(args))
      {}

      ~SmootherWithDefect()
      {
        ConstructionTraits<S>::deconstruct(smoother);
      }

      /** @brief apply the smoother in an AMG preSmoothing stage
       * @param x the left hand side
       * @param d the defect
       * @param b the right hand side
       */
      void preApply(Domain& x, Range& d, const Range& b)
      {
        // apply the preconditioner
        SmootherWithDefectHelper<S,SmootherCalculatesDefect<S>::value>::preApply(*smoother,A,x,d,b);
      }

      /** @brief apply the smoother in an AMG postSmoothing stage
       * @param x the left hand side
       * @param d the defect
       * @param b the right hand side
       */
      void postApply(Domain& x, Range& d, const Range& b)
      {
        // apply the preconditioner
        SmootherWithDefectHelper<S,SmootherCalculatesDefect<S>::value>::postApply(*smoother,A,x,d,b);
      }

      private:
      const Matrix& A;
      S* smoother;
    };

    //! traits specialization: a wrapped smoother is constructed by taking the arguments
    //! of the normal smoother and forwarding them in the constructor
    template<typename S>
    struct ConstructionTraits<SmootherWithDefect<S> >
    {
      typedef typename ConstructionTraits<S>::Arguments Arguments;

      static inline SmootherWithDefect<S>* construct(Arguments& args)
      {
        return new SmootherWithDefect<S>(args);
      }

      static inline void deconstruct(SmootherWithDefect<S>* obj)
      {
        delete obj;
      }
    };

    //! traits specialization: the wrapped smoother has the same traits as the smoother
    template<typename S>
    struct SmootherTraits<SmootherWithDefect<S> >
    {
      typedef typename SmootherTraits<S>::Arguments Arguments;
    };

    /** @brief a symmetric Gauss-Seidel smoother that does compute defects
     * @tparam M the matrix type
     * @tparam X the domain type
     * @tparam Y the range type
     * fulfills the interface required by FastAMG. Options may be passed via the
     * DefaultSmootherArgs factory concept.
     */
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

      //! the type of coarse grid smoother that fits this smoother in AMG
      typedef typename Dune::SeqSOR<M,X,Y> RecommendedCoarseSmoother;

      /** @brief construct a symetric Gauss-Seidel smoother that computes defects
       * @param A_ the matrix them smoother operates on
       * @param num_iter_ the number of iterations done by the smoother
       * These parameters may be provided via the ConstructionTraits class!
       */
      GaussSeidelWithDefect(const M& A_, int num_iter_) : A(A_),num_iter(num_iter_)
      {}

      enum {
        /** @brief The solver category */
        category = SolverCategory::sequential
      };

      /** @brief apply the smoother in an AMG preSmoothing stage
       * @param x the left hand side
       * @param d the defect
       * @param b the right hand side
       */
      void preApply(X& x, Y& d, const Y& b)
      {
        // perform iterations. These have to know whether they are first
        // and whether to compute a defect.
        if (num_iter == 1)
          forward_apply(A,x,d,b,true,true);
        else
        {
          forward_apply(A,x,d,b,true,false);
          for (int i=0; i<num_iter-2; i++)
            forward_apply(A,x,d,b, false,false);
          forward_apply(A,x,d,b,false,true);
        }
      }

      /** @brief apply the smoother in an AMG postSmoothing stage
       * @param x the left hand side
       * @param d the defect
       * @param b the right hand side
       */
      void postApply(X& x, Y& d,
                        const Y& b)
      {
        for (int i=0; i<num_iter; i++)
          backward_apply(A,x,d,b);
      }

      private:
      //! do one forward step
      void forward_apply(const M& A, X& x, Y& d, const Y& b, bool first, bool compDef)
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

      //! do one backward step
      void backward_apply(const M& A, X& x, Y& d, const Y& b)
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

          diag->solve(*xIter,*dIter);
        }
      }

      const M& A;
      int num_iter;
    };

    //! traits specialization: wrap the GaussSeidelWithDefect constructor
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

    //! traits specialization: GaussSeidelWithDefect takes no special arguments
    template<typename M,typename X,typename Y>
    struct SmootherTraits<GaussSeidelWithDefect<M,X,Y> >
    {
      typedef DefaultSmootherArgs<typename M::field_type> Arguments;
    };

    //! GaussSeidelWithDefect does compute defects!
    template<typename M,typename X,typename Y>
    struct SmootherCalculatesDefect<GaussSeidelWithDefect<M,X,Y> >
    {
      enum {
        value = true
      };
    };

    /** @brief a symmetric Jacobi smoother that does compute defects
     * @tparam M the matrix type
     * @tparam X the domain type
     * @tparam Y the range type
     * fulfills the interface required by FastAMG. Options may be passed via the
     * DefaultSmootherArgs factory concept.
     */
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

      //! the type of coarse grid smoother that fits this smoother in AMG
      typedef typename Dune::SeqJac<M,X,Y> RecommendedCoarseSmoother;

      /** @brief construct a symmetric Jacobi smoother that computes defects
       * @param A_ the matrix the smoother operates on
       * @param num_iter_ the number of iterations done by the smoother
       * @param w_ the relaxation factor
       * These parameters may be provided via the ConstructionTraits class!
       */
      JacobiWithDefect(const M& A_, int num_iter_, Field w_) : A(A_), num_iter(num_iter_), w(w_)
      {}

      enum {
        /** @brief The solver category */
        category = SolverCategory::sequential
      };

      /** @brief apply the smoother in an AMG preSmoothing stage
       * @param x the left hand side
       * @param d the defect
       * @param b the right hand side
       */
      void preApply(X& x, Y& d, const Y& b)
      {
        if (num_iter == 1)
          forward_apply(A,x,d,b,true,true);
        else
        {
          forward_apply(A,x,d,b,true,false);
          for (int i=0; i<num_iter-2; i++)
            forward_apply(A,x,d,b,false,false);
          forward_apply(A,x,d,b,false,true);
        }
      }

      /** @brief apply the smoother in an AMG postSmoothing stage
       * @param x the left hand side
       * @param d the defect
       * @param b the right hand side
       */
      void postApply(X& x, Y& d, const Y& b)
      {
        for (int i=0; i<num_iter; i++)
          backward_apply(A,x,d,b);
      }

      private:
      //! do one forward step
      void forward_apply(const M& A, X& x, Y& d, const Y& b, bool first, bool compDef)
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
          x.axpy(Field(1)-w,xold);
        }
      }

      //! do one backward step
      void backward_apply(const M& A, X& x, Y& d, const Y& b)
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
        x.axpy(Field(1)-w,xold);
      }

      const M& A;
      int num_iter;
      Field w;
    };

    //! traits specialization: wrap the JacobiWithDefect constructor
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

    //! traits specialization: JacobiWithDefect takes no special arguments
    template<typename M,typename X,typename Y>
    struct SmootherTraits<JacobiWithDefect<M,X,Y> >
    {
      typedef DefaultSmootherArgs<typename M::field_type> Arguments;
    };

    //! JacobiWithDefect does compute defects!
    template<typename M,typename X,typename Y>
    struct SmootherCalculatesDefect<JacobiWithDefect<M,X,Y> >
    {
      enum {
        value = true
      };
    };
  } // end namespace Amg
} // end namespace Dune
#endif
