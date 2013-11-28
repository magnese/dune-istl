// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
// $Id$
#ifndef DUNE_AMGHIERARCHY_HH
#define DUNE_AMGHIERARCHY_HH

#include <list>
#include <memory>
#include <limits>
#include <algorithm>
#include "aggregates.hh"
#include "graph.hh"
#include "galerkin.hh"
#include "renumberer.hh"
#include "graphcreator.hh"
#include <dune/common/stdstreams.hh>
#include <dune/common/timer.hh>
#include <dune/common/tuples.hh>
#include <dune/common/bigunsignedint.hh>
#include <dune/istl/bvector.hh>
#include <dune/common/parallel/indexset.hh>
#include <dune/istl/matrixutils.hh>
#include <dune/istl/matrixredistribute.hh>
#include <dune/istl/paamg/dependency.hh>
#include <dune/istl/paamg/graph.hh>
#include <dune/istl/paamg/indicescoarsener.hh>
#include <dune/istl/paamg/globalaggregates.hh>
#include <dune/istl/paamg/construction.hh>
#include <dune/istl/paamg/smoother.hh>
#include <dune/istl/paamg/transfer.hh>

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
     * @brief Provides a classes representing the hierarchies in AMG.
     */

    enum {
      /**
       * @brief Hard limit for the number of processes allowed.
       *
       * This is needed to prevent overflows when calculating
       * the coarsening rate. Currently set 72,000 which is
       * enough for JUGENE.
       */
      MAX_PROCESSES = 72000
    };

    /**
     * @brief A hierarchy of coantainers (e.g. matrices or vectors)
     *
     * Because sometimes a redistribution of the parallel data might be
     * advisable one can add redistributed version of the container at
     * each level.
     */
    template<typename T, typename A=std::allocator<T> >
    class Hierarchy
    {
    public:
      /**
       * @brief The type of the container we store.
       */
      typedef T MemberType;

      template<typename T1, typename T2>
      class LevelIterator;

    private:
      /**
       * @brief An element in the hierarchy.
       */
      struct Element
      {
        friend class LevelIterator<Hierarchy<T,A>, T>;
        friend class LevelIterator<const Hierarchy<T,A>, const T>;

        /** @brief The next coarser element in the list. */
        Element* coarser_;

        /** @brief The next finer element in the list. */
        Element* finer_;

        /** @brief Pointer to the element. */
        MemberType* element_;

        /** @brief The redistributed version of the element. */
        MemberType* redistributed_;
      };
    public:
      //       enum{
      //        /**
      //         * @brief If true only the method addCoarser will be usable
      //         * otherwise only the method addFiner will be usable.
      //         */
      //        coarsen = b
      //          };

      /**
       * @brief The allocator to use for the list elements.
       */
      typedef typename A::template rebind<Element>::other Allocator;

      typedef typename ConstructionTraits<T>::Arguments Arguments;

      /**
       * @brief Construct a new hierarchy.
       * @param first The first element in the hierarchy.
       */
      Hierarchy(MemberType& first);

      /**
       * @brief Construct a new hierarchy.
       * @param first Pointer to the first element in the hierarchy.
       * @warning Hierarchy will be responsible for the memory
       * management of the pointer.
       */
      Hierarchy(MemberType* first);

      /**
       * @brief Construct a new empty hierarchy.
       */
      Hierarchy();

      /**
       * @brief Copy constructor.
       */
      Hierarchy(const Hierarchy& other);
      /**
       * @brief Add an element on a coarser level.
       * @param args The arguments needed for the construction.
       */
      void addCoarser(Arguments& args);

      void addRedistributedOnCoarsest(Arguments& args);

      /**
       * @brief Add an element on a finer level.
       * @param args The arguments needed for the construction.
       */
      void addFiner(Arguments& args);

      /**
       * @brief Iterator over the levels in the hierarchy.
       *
       * operator++() moves to the next coarser level in the hierarchy.
       * while operator--() moves to the next finer level in the hierarchy.
       */
      template<class C, class T1>
      class LevelIterator
        : public BidirectionalIteratorFacade<LevelIterator<C,T1>,T1,T1&>
      {
        friend class LevelIterator<typename remove_const<C>::type,
            typename remove_const<T1>::type >;
        friend class LevelIterator<const typename remove_const<C>::type,
            const typename remove_const<T1>::type >;

      public:
        /** @brief Constructor. */
        LevelIterator()
          : element_(0)
        {}

        LevelIterator(Element* element)
          : element_(element)
        {}

        /** @brief Copy constructor. */
        LevelIterator(const LevelIterator<typename remove_const<C>::type,
                          typename remove_const<T1>::type>& other)
          : element_(other.element_)
        {}

        /** @brief Copy constructor. */
        LevelIterator(const LevelIterator<const typename remove_const<C>::type,
                          const typename remove_const<T1>::type>& other)
          : element_(other.element_)
        {}

        /**
         * @brief Equality check.
         */
        bool equals(const LevelIterator<typename remove_const<C>::type,
                        typename remove_const<T1>::type>& other) const
        {
          return element_ == other.element_;
        }

        /**
         * @brief Equality check.
         */
        bool equals(const LevelIterator<const typename remove_const<C>::type,
                        const typename remove_const<T1>::type>& other) const
        {
          return element_ == other.element_;
        }

        /** @brief Dereference the iterator. */
        T1& dereference() const
        {
          return *(element_->element_);
        }

        /** @brief Move to the next coarser level */
        void increment()
        {
          element_ = element_->coarser_;
        }

        /** @brief Move to the next fine level */
        void decrement()
        {
          element_ = element_->finer_;
        }

        /**
         * @brief Check whether there was a redistribution at the current level.
         * @return True if there is a redistributed version of the conatainer at the current level.
         */
        bool isRedistributed() const
        {
          return element_->redistributed_;
        }

        /**
         * @brief Get the redistributed container.
         * @return The redistributed container.
         */
        T1& getRedistributed() const
        {
          assert(element_->redistributed_);
          return *element_->redistributed_;
        }
        void addRedistributed(T1* t)
        {
          element_->redistributed_ = t;
        }

        void deleteRedistributed()
        {
          element_->redistributed_ = nullptr;
        }

      private:
        Element* element_;
      };

      /** @brief Type of the mutable iterator. */
      typedef LevelIterator<Hierarchy<T,A>,T> Iterator;

      /** @brief Type of the const iterator. */
      typedef LevelIterator<const Hierarchy<T,A>, const T> ConstIterator;

      /**
       * @brief Get an iterator positioned at the finest level.
       * @return An iterator positioned at the finest level.
       */
      Iterator finest();

      /**
       * @brief Get an iterator positioned at the coarsest level.
       * @return An iterator positioned at the coarsest level.
       */
      Iterator coarsest();


      /**
       * @brief Get an iterator positioned at the finest level.
       * @return An iterator positioned at the finest level.
       */
      ConstIterator finest() const;

      /**
       * @brief Get an iterator positioned at the coarsest level.
       * @return An iterator positioned at the coarsest level.
       */
      ConstIterator coarsest() const;

      /**
       * @brief Get the number of levels in the hierarchy.
       * @return The number of levels.
       */
      std::size_t levels() const;

      /** @brief Destructor. */
      ~Hierarchy();

    private:
      /** @brief The finest element in the hierarchy. */
      Element* finest_;
      /** @brief The coarsest element in the hierarchy. */
      Element* coarsest_;
      /** @brief Whether the first element was not allocated by us. */
      Element* nonAllocated_;
      /** @brief The allocator for the list elements. */
      Allocator allocator_;
      /** @brief The number of levels in the hierarchy. */
      int levels_;
    };

    /**
     * @brief The hierarchies build by the coarsening process.
     *
     * Namely a hierarchy of matrices, index sets, remote indices,
     * interfaces and communicators.
     */
    template<class M, class PI, class A=std::allocator<M> >
    class MatrixHierarchy
    {
    public:
      /** @brief The type of the matrix operator. */
      typedef M MatrixOperator;

      /** @brief The type of the matrix. */
      typedef typename MatrixOperator::matrix_type Matrix;

      /** @brief The type of the index set. */
      typedef PI ParallelInformation;

      /** @brief The allocator to use. */
      typedef A Allocator;

      /** @brief The type of the aggregates map we use. */
      typedef Dune::Amg::AggregatesMap<typename MatrixGraph<Matrix>::VertexDescriptor> AggregatesMap;

      /** @brief The type of the parallel matrix hierarchy. */
      typedef Dune::Amg::Hierarchy<MatrixOperator,Allocator> ParallelMatrixHierarchy;

      /** @brief The type of the parallel informarion hierarchy. */
      typedef Dune::Amg::Hierarchy<ParallelInformation,Allocator> ParallelInformationHierarchy;

      /** @brief Allocator for pointers. */
      typedef typename Allocator::template rebind<AggregatesMap*>::other AAllocator;

      /** @brief The type of the aggregates maps list. */
      typedef std::list<AggregatesMap*,AAllocator> AggregatesMapList;

      /** @brief The type of the redistribute information. */
      typedef RedistributeInformation<ParallelInformation> RedistributeInfoType;

      /** @brief Allocator for RedistributeInfoType. */
      typedef typename Allocator::template rebind<RedistributeInfoType>::other RILAllocator;

      /** @brief The type of the list of redistribute information. */
      typedef std::list<RedistributeInfoType,RILAllocator> RedistributeInfoList;

      /**
       * @brief Constructor
       * @param fineMatrix The matrix to coarsen.
       * @param pinfo The information about the parallel data decomposition at the first level.
       */
      MatrixHierarchy(const MatrixOperator& fineMatrix,
                      const ParallelInformation& pinfo=ParallelInformation());


      ~MatrixHierarchy();

      /**
       * @brief Build the matrix hierarchy using aggregation.
       *
       * @brief criterion The criterion describing the aggregation process.
       */
      template<typename O, typename T>
      int build(const T& criterion);

      /**
       * @brief Recalculate the galerkin products.
       *
       * If the data of the fine matrix changes but not its sparsity pattern
       * this will recalculate all coarser levels without starting the expensive
       * aggregation process all over again.
       */
      template<class F>
      void recalculateGalerkin(const F& copyFlags);

      /**
       * @brief Coarsen the vector hierarchy according to the matrix hierarchy.
       * @param hierarchy The vector hierarchy to coarsen.
       */
      template<class V, class TA>
      void coarsenVector(Hierarchy<BlockVector<V,TA> >& hierarchy) const;

      /**
       * @brief Coarsen the smoother hierarchy according to the matrix hierarchy.
       * @param smoothers The smoother hierarchy to coarsen.
       * @param args The arguments for the construction of the coarse level smoothers.
       */
      template<class S, class TA>
      void coarsenSmoother(Hierarchy<S,TA>& smoothers,
                           const typename SmootherTraits<S>::Arguments& args) const;

      /**
       * @brief Get the number of levels in the hierarchy.
       * @return The number of levels.
       */
      std::size_t levels() const;

      /**
       * @brief Get the max number of levels in the hierarchy of processors.
       * @return The maximum number of levels.
       */
      std::size_t maxlevels() const;

      bool hasCoarsest() const;

      /**
       * @brief Whether the hierarchy was built.
       * @return true if the MatrixHierarchy::build method was called.
       */
      bool isBuilt() const;

      /**
       * @brief Get the matrix hierarchy.
       * @return The matrix hierarchy.
       */
      const ParallelMatrixHierarchy& matrices() const;

      /**
       * @brief Get the hierarchy of the parallel data distribution information.
       * @return The hierarchy of the parallel data distribution information.
       */
      const ParallelInformationHierarchy& parallelInformation() const;

      /**
       * @brief Get the hierarchy of the mappings of the nodes onto aggregates.
       * @return The hierarchy of the mappings of the nodes onto aggregates.
       */
      const AggregatesMapList& aggregatesMaps() const;

      /**
       * @brief Get the hierachy of the information about redistributions,
       * @return The hierarchy of the information about redistributions of the
       * data to fewer processes.
       */
      const RedistributeInfoList& redistributeInformation() const;


      typename MatrixOperator::field_type getProlongationDampingFactor() const
      {
        return prolongDamp_;
      }

      /**
       * @brief Get the mapping of fine level unknowns to coarse level
       * aggregates.
       *
       * For each fine level unknown i the correcponding data[i] is the
       * aggregate it belongs to on the coarsest level.
       *
       * @param[out] data The mapping of fine level unknowns to coarse level
       * aggregates.
       */
      void getCoarsestAggregatesOnFinest(std::vector<std::size_t>& data) const;

    private:
      typedef typename ConstructionTraits<MatrixOperator>::Arguments MatrixArgs;
      typedef typename ConstructionTraits<ParallelInformation>::Arguments CommunicationArgs;
      /** @brief The list of aggregates maps. */
      AggregatesMapList aggregatesMaps_;
      /** @brief The list of redistributes. */
      RedistributeInfoList redistributes_;
      /** @brief The hierarchy of parallel matrices. */
      ParallelMatrixHierarchy matrices_;
      /** @brief The hierarchy of the parallel information. */
      ParallelInformationHierarchy parallelInformation_;

      /** @brief Whether the hierarchy was built. */
      bool built_;

      /** @brief The maximum number of level across all processors.*/
      int maxlevels_;

      typename MatrixOperator::field_type prolongDamp_;

      /**
       * @brief functor to print matrix statistics.
       */
      template<class Matrix, bool print>
      struct MatrixStats
      {

        /**
         * @brief Print matrix statistics.
         */
        static void stats(const Matrix& matrix)
        {}
      };

      template<class Matrix>
      struct MatrixStats<Matrix,true>
      {
        struct calc
        {
          typedef typename Matrix::size_type size_type;
          typedef typename Matrix::row_type matrix_row;

          calc()
          {
            min=std::numeric_limits<size_type>::max();
            max=0;
            sum=0;
          }

          void operator()(const matrix_row& row)
          {
            min=std::min(min, row.size());
            max=std::max(max, row.size());
            sum += row.size();
          }

          size_type min;
          size_type max;
          size_type sum;
        };
        /**
         * @brief Print matrix statistics.
         */
        static void stats(const Matrix& matrix)
        {
          calc c= for_each(matrix.begin(), matrix.end(), calc());
          dinfo<<"Matrix row: min="<<c.min<<" max="<<c.max
               <<" average="<<static_cast<double>(c.sum)/matrix.N()
               <<std::endl;
        }
      };
    };

    /**
     * @brief The criterion describing the stop criteria for the coarsening process.
     */
    template<class T>
    class CoarsenCriterion : public T
    {
    public:
      /**
       * @brief The criterion for tagging connections as strong and nodes as isolated.
       * This might be e.g. SymmetricDependency or UnSymmetricCriterion.
       */
      typedef T AggregationCriterion;

      /**
       * @brief Constructor
       * @param maxLevel The maximum number of levels allowed in the matrix hierarchy (default: 100).
       * @param coarsenTarget If the number of nodes in the matrix is below this threshold the
       * coarsening will stop (default: 1000).
       * @param minCoarsenRate If the coarsening rate falls below this threshold the
       * coarsening will stop (default: 1.2)
       * @param prolongDamp The damping factor to apply to the prolongated update (default: 1.6)
       * @param accumulate Whether to accumulate the data onto fewer processors on coarser levels.
       */
      CoarsenCriterion(int maxLevel=100, int coarsenTarget=1000, double minCoarsenRate=1.2,
                       double prolongDamp=1.6, AccumulationMode accumulate=successiveAccu)
        : AggregationCriterion(Dune::Amg::Parameters(maxLevel, coarsenTarget, minCoarsenRate, prolongDamp, accumulate))
      {}

      CoarsenCriterion(const Dune::Amg::Parameters& parms)
        : AggregationCriterion(parms)
      {}

    };

    template<typename M, typename C1>
    bool repartitionAndDistributeMatrix(const M& origMatrix, M& newMatrix,
                                        SequentialInformation& origSequentialInformationomm,
                                        SequentialInformation*& newComm,
                                        RedistributeInformation<SequentialInformation>& ri,
                                        int nparts, C1& criterion)
    {
      DUNE_THROW(NotImplemented, "Redistribution does not make sense in sequential code!");
    }


    template<typename M, typename C, typename C1>
    bool repartitionAndDistributeMatrix(const M& origMatrix, M& newMatrix, C& origComm, C*& newComm,
                                        RedistributeInformation<C>& ri,
                                        int nparts, C1& criterion)
    {
      Timer time;
#ifdef AMG_REPART_ON_COMM_GRAPH
      // Done not repartition the matrix graph, but a graph of the communication scheme.
      bool existentOnRedist=Dune::commGraphRepartition(origMatrix, origComm, nparts, newComm,
                                                       ri.getInterface(),
                                                       criterion.debugLevel()>1);

#else
      typedef Dune::Amg::MatrixGraph<const M> MatrixGraph;
      typedef Dune::Amg::PropertiesGraph<MatrixGraph,
          VertexProperties,
          EdgeProperties,
          IdentityMap,
          IdentityMap> PropertiesGraph;
      MatrixGraph graph(origMatrix);
      PropertiesGraph pgraph(graph);
      buildDependency(pgraph, origMatrix, criterion, false);

#ifdef DEBUG_REPART
      if(origComm.communicator().rank()==0)
        std::cout<<"Original matrix"<<std::endl;
      origComm.communicator().barrier();
      printGlobalSparseMatrix(origMatrix, origComm, std::cout);
#endif
      bool existentOnRedist=Dune::graphRepartition(pgraph, origComm, nparts,
                                                   newComm, ri.getInterface(),
                                                   criterion.debugLevel()>1);
#endif // if else AMG_REPART

      if(origComm.communicator().rank()==0  && criterion.debugLevel()>1)
        std::cout<<"Repartitioning took "<<time.elapsed()<<" seconds."<<std::endl;

      ri.setSetup();

#ifdef DEBUG_REPART
      ri.checkInterface(origComm.indexSet(), newComm->indexSet(), origComm.communicator());
#endif

      redistributeMatrix(const_cast<M&>(origMatrix), newMatrix, origComm, *newComm, ri);

#ifdef DEBUG_REPART
      if(origComm.communicator().rank()==0)
        std::cout<<"Original matrix"<<std::endl;
      origComm.communicator().barrier();
      if(newComm->communicator().size()>0)
        printGlobalSparseMatrix(newMatrix, *newComm, std::cout);
      origComm.communicator().barrier();
#endif

      if(origComm.communicator().rank()==0  && criterion.debugLevel()>1)
        std::cout<<"Redistributing matrix took "<<time.elapsed()<<" seconds."<<std::endl;
      return existentOnRedist;

    }

    template<typename M>
    bool repartitionAndDistributeMatrix(M& origMatrix, M& newMatrix,
                                        SequentialInformation& origComm,
                                        SequentialInformation& newComm,
                                        RedistributeInformation<SequentialInformation>& ri)
    {
      return true;
    }

    template<class M, class IS, class A>
    MatrixHierarchy<M,IS,A>::MatrixHierarchy(const MatrixOperator& fineOperator,
                                             const ParallelInformation& pinfo)
      : matrices_(const_cast<MatrixOperator&>(fineOperator)),
        parallelInformation_(const_cast<ParallelInformation&>(pinfo))
    {
      dune_static_assert((static_cast<int>(MatrixOperator::category) ==
                          static_cast<int>(SolverCategory::sequential) ||
                          static_cast<int>(MatrixOperator::category) ==
                          static_cast<int>(SolverCategory::overlapping) ||
                          static_cast<int>(MatrixOperator::category) ==
                          static_cast<int>(SolverCategory::nonoverlapping)),
                         "MatrixOperator must be of category sequential or overlapping or nonoverlapping");
      if (static_cast<int>(MatrixOperator::category) != static_cast<int>(pinfo.getSolverCategory()))
        DUNE_THROW(ISTLError, "MatrixOperator and ParallelInformation must belong to the same category!");

    }

    template<class M, class IS, class A>
    template<typename O, typename T>
    int MatrixHierarchy<M,IS,A>::build(const T& criterion)
    {
      prolongDamp_ = criterion.getProlongationDampingFactor();
      typedef O OverlapFlags;
      typedef typename ParallelMatrixHierarchy::Iterator MatIterator;
      typedef typename ParallelInformationHierarchy::Iterator PInfoIterator;

      static const int noints=(Dune::Amg::MAX_PROCESSES/4096>0) ? (Dune::Amg::MAX_PROCESSES/4096) : 1;

      typedef bigunsignedint<sizeof(int)*8*noints> BIGINT;
      GalerkinProduct productBuilder;
      MatIterator mlevel = matrices_.finest();
      MatrixStats<typename M::matrix_type,MINIMAL_DEBUG_LEVEL<=INFO_DEBUG_LEVEL>::stats(mlevel->getmat());

      PInfoIterator infoLevel = parallelInformation_.finest();
      BIGINT finenonzeros=countNonZeros(mlevel->getmat());
      finenonzeros = infoLevel->communicator().sum(finenonzeros);
      BIGINT allnonzeros = finenonzeros;


      int level = 0;
      int rank = 0;

      BIGINT unknowns = mlevel->getmat().N();

      unknowns = infoLevel->communicator().sum(unknowns);
      double dunknowns=unknowns.todouble();
      infoLevel->buildGlobalLookup(mlevel->getmat().N());
      redistributes_.push_back(RedistributeInfoType());

      // initialize a bcrs compression statistic object to allow recurisve heuristics on parameters
      CompressionStatistics<typename M::matrix_type::size_type> compress_stats;
      // This will be used as the averag number of nonzeros per row for the firt coarse level matrix.
      // We use ceil to be on the safer side.
      compress_stats.avg = std::ceil(static_cast<double>(mlevel->getmat().nonzeroes())/static_cast<double>(mlevel->getmat().N()));
      compress_stats.overflow_total = 0;
      compress_stats.mem_ratio = 0.0;

      // Marker to check whether aggregation failed or not.
      // A non-zero entry indicates failure.
      int globalsuccess=0;

      for(; level < criterion.maxLevel(); ++level, ++mlevel) {
        assert(matrices_.levels()==redistributes_.size());
        rank = infoLevel->communicator().rank();
        if(rank==0 && criterion.debugLevel()>1)
          std::cout<<"Level "<<level<<" has "<<dunknowns<<" unknowns, "<<dunknowns/infoLevel->communicator().size()
                   <<" unknowns per proc (procs="<<infoLevel->communicator().size()<<")"<<std::endl;

        MatrixOperator* matrix=&(*mlevel);
        ParallelInformation* info =&(*infoLevel);

        if((
#if HAVE_PARMETIS
             criterion.accumulate()==successiveAccu
#else
             false
#endif
             || (criterion.accumulate()==atOnceAccu
                 && dunknowns < 30*infoLevel->communicator().size()))
           && infoLevel->communicator().size()>1 &&
           dunknowns/infoLevel->communicator().size() <= criterion.coarsenTarget())
        {
          // accumulate to fewer processors
          Matrix* redistMat= new Matrix();
          ParallelInformation* redistComm=0;
          std::size_t nodomains = (std::size_t)std::ceil(dunknowns/(criterion.minAggregateSize()
                                                                    *criterion.coarsenTarget()));
          if( nodomains<=criterion.minAggregateSize()/2 ||
              dunknowns <= criterion.coarsenTarget() )
            nodomains=1;

          bool existentOnNextLevel =
            repartitionAndDistributeMatrix(mlevel->getmat(), *redistMat, *infoLevel,
                                           redistComm, redistributes_.back(), nodomains,
                                           criterion);
          BIGINT unknowns = redistMat->N();
          unknowns = infoLevel->communicator().sum(unknowns);
          dunknowns= unknowns.todouble();
          if(redistComm->communicator().rank()==0 && criterion.debugLevel()>1)
            std::cout<<"Level "<<level<<" (redistributed) has "<<dunknowns<<" unknowns, "<<dunknowns/redistComm->communicator().size()
                     <<" unknowns per proc (procs="<<redistComm->communicator().size()<<")"<<std::endl;
          MatrixArgs args(*redistMat, *redistComm);
          mlevel.addRedistributed(ConstructionTraits<MatrixOperator>::construct(args));
          assert(mlevel.isRedistributed());
          infoLevel.addRedistributed(redistComm);
          infoLevel->freeGlobalLookup();

          if(!existentOnNextLevel)
            // We do not hold any data on the redistributed partitioning
            break;

          // Work on the redistributed Matrix from now on
          matrix = &(mlevel.getRedistributed());
          info = &(infoLevel.getRedistributed());
          info->buildGlobalLookup(matrix->getmat().N());
        }

        rank = info->communicator().rank();
        if(dunknowns <= criterion.coarsenTarget())
          // No further coarsening needed
          break;

        typedef PropertiesGraphCreator<MatrixOperator> GraphCreator;
        typedef typename GraphCreator::PropertiesGraph PropertiesGraph;
        typedef typename GraphCreator::GraphTuple GraphTuple;

        typedef typename PropertiesGraph::VertexDescriptor Vertex;

        std::vector<bool> excluded(matrix->getmat().N(), false);

        GraphTuple graphs = GraphCreator::create(*matrix, excluded, *info, OverlapFlags());

        AggregatesMap* aggregatesMap=new AggregatesMap(get<1>(graphs)->maxVertex()+1);

        aggregatesMaps_.push_back(aggregatesMap);

        Timer watch;
        watch.reset();
        int noAggregates, isoAggregates, oneAggregates, skippedAggregates;

        tie(noAggregates, isoAggregates, oneAggregates, skippedAggregates) =
          aggregatesMap->buildAggregates(matrix->getmat(), *(get<1>(graphs)), criterion, level==0);

        if(rank==0 && criterion.debugLevel()>2)
          std::cout<<" Have built "<<noAggregates<<" aggregates totally ("<<isoAggregates<<" isolated aggregates, "<<
          oneAggregates<<" aggregates of one vertex,  and skipped "<<
          skippedAggregates<<" aggregates)."<<std::endl;
#ifdef TEST_AGGLO
        {
          // calculate size of local matrix in the distributed direction
          int start, end, overlapStart, overlapEnd;
          int procs=info->communicator().rank();
          int n = UNKNOWNS/procs; // number of unknowns per process
          int bigger = UNKNOWNS%procs; // number of process with n+1 unknows

          // Compute owner region
          if(rank<bigger) {
            start = rank*(n+1);
            end   = (rank+1)*(n+1);
          }else{
            start = bigger + rank * n;
            end   = bigger + (rank + 1) * n;
          }

          // Compute overlap region
          if(start>0)
            overlapStart = start - 1;
          else
            overlapStart = start;

          if(end<UNKNOWNS)
            overlapEnd = end + 1;
          else
            overlapEnd = end;

          assert((UNKNOWNS)*(overlapEnd-overlapStart)==aggregatesMap->noVertices());
          for(int j=0; j< UNKNOWNS; ++j)
            for(int i=0; i < UNKNOWNS; ++i)
            {
              if(i>=overlapStart && i<overlapEnd)
              {
                int no = (j/2)*((UNKNOWNS)/2)+i/2;
                (*aggregatesMap)[j*(overlapEnd-overlapStart)+i-overlapStart]=no;
              }
            }
        }
#endif
        if(criterion.debugLevel()>1 && info->communicator().rank()==0)
          std::cout<<"aggregating finished."<<std::endl;

        BIGINT gnoAggregates=noAggregates;
        gnoAggregates = info->communicator().sum(gnoAggregates);
        double dgnoAggregates = gnoAggregates.todouble();
#ifdef TEST_AGGLO
        BIGINT gnoAggregates=((UNKNOWNS)/2)*((UNKNOWNS)/2);
#endif

        if(criterion.debugLevel()>2 && rank==0)
          std::cout << "Building "<<dgnoAggregates<<" aggregates took "<<watch.elapsed()<<" seconds."<<std::endl;

        if(dgnoAggregates==0 || dunknowns/dgnoAggregates<criterion.minCoarsenRate())
        {
          if(rank==0)
          {
            if(dgnoAggregates>0)
              std::cerr << "Stopped coarsening because of rate breakdown "<<dunknowns<<"/"<<dgnoAggregates
                        <<"="<<dunknowns/dgnoAggregates<<"<"
                        <<criterion.minCoarsenRate()<<std::endl;
            else
              std::cerr<< "Could not build any aggregates. Probably no connected nodes."<<std::endl;
          }
          aggregatesMap->free();
          delete aggregatesMap;
          aggregatesMaps_.pop_back();

          if(criterion.accumulate() && mlevel.isRedistributed() && info->communicator().size()>1) {
            // coarse level matrix was already redistributed, but to more than 1 process
            // Therefore need to delete the redistribution. Further down it will
            // then be redistributed to 1 process
            delete &(mlevel.getRedistributed().getmat());
            mlevel.deleteRedistributed();
            delete &(infoLevel.getRedistributed());
            infoLevel.deleteRedistributed();
            redistributes_.back().resetSetup();
          }

          break;
        }
        unknowns =  noAggregates;
        dunknowns = dgnoAggregates;

        CommunicationArgs commargs(info->communicator(),info->getSolverCategory());
        parallelInformation_.addCoarser(commargs);

        ++infoLevel; // parallel information on coarse level

        typename PropertyMapTypeSelector<VertexVisitedTag,PropertiesGraph>::Type visitedMap =
          get(VertexVisitedTag(), *(get<1>(graphs)));

        watch.reset();
        int aggregates = IndicesCoarsener<ParallelInformation,OverlapFlags>
                         ::coarsen(*info,
                                   *(get<1>(graphs)),
                                   visitedMap,
                                   *aggregatesMap,
                                   *infoLevel,
                                   noAggregates);
        GraphCreator::free(graphs);

        if(criterion.debugLevel()>2) {
          if(rank==0)
            std::cout<<"Coarsening of index sets took "<<watch.elapsed()<<" seconds."<<std::endl;
        }

        watch.reset();

        infoLevel->buildGlobalLookup(aggregates);
        AggregatesPublisher<Vertex,OverlapFlags,ParallelInformation>::publish(*aggregatesMap,
                                                                              *info,
                                                                              infoLevel->globalLookup());


        if(criterion.debugLevel()>2) {
          if(rank==0)
            std::cout<<"Communicating global aggregate numbers took "<<watch.elapsed()<<" seconds."<<std::endl;
        }

        watch.reset();
        std::vector<bool>& visited=excluded;

        typedef std::vector<bool>::iterator Iterator;
        typedef IteratorPropertyMap<Iterator, IdentityMap> VisitedMap2;
        Iterator end = visited.end();
        for(Iterator iter= visited.begin(); iter != end; ++iter)
          *iter=false;

        VisitedMap2 visitedMap2(visited.begin(), Dune::IdentityMap());

        info->freeGlobalLookup();

        delete get<0>(graphs);

        typename M::matrix_type::size_type avg = std::ceil(compress_stats.avg);

        typename MatrixOperator::matrix_type* coarseMatrix;
        double overflow=criterion.getOverflowFraction();
        int success;

        for(std::size_t tries=0; tries<3; ++tries)
        {
          try
          {
            coarseMatrix= new typename MatrixOperator::matrix_type(aggregates,aggregates, avg , overflow ,MatrixOperator::matrix_type::implicit);

            ImplicitMatrixBuilder<typename M::matrix_type> wrapped(*coarseMatrix);
            productBuilder.calculate(matrix->getmat(), *aggregatesMap, wrapped, *infoLevel, OverlapFlags());
            compress_stats = coarseMatrix->compress();
            if(criterion.debugLevel()>2) {
              if(rank==0)
                std::cout<<"Calculation entries of Galerkin product took "<<watch.elapsed()<<" seconds."<<std::endl;
            }
            success=0;
            break;
          }
          catch(ImplicitModeOverflowExhausted e)
          {
            std::cerr<<e.what()<<std::endl;
            overflow*=2.0;
            std::cerr<<"Increasing overflow for matrix setup to "<<overflow<<std::endl;
            delete coarseMatrix;
            success=1;
          }
        }
        globalsuccess=infoLevel->communicator().sum(success);
        if(globalsuccess)
        {
          // Building the matrix failed on at least one process
          if(!success){
            delete coarseMatrix;
          }
          break;
        }

        BIGINT nonzeros = countNonZeros(*coarseMatrix);
        allnonzeros = allnonzeros + infoLevel->communicator().sum(nonzeros);
        MatrixArgs args(*coarseMatrix, *infoLevel);

        matrices_.addCoarser(args);
        redistributes_.push_back(RedistributeInfoType());
      } // end level loop


      infoLevel->freeGlobalLookup();
      // Check success over all processes, i.e. the ones on the final level.
      globalsuccess = parallelInformation_.finest()->communicator().sum(globalsuccess);
      if(globalsuccess)
        return 1;

      built_=true;
      AggregatesMap* aggregatesMap=new AggregatesMap(0);
      aggregatesMaps_.push_back(aggregatesMap);

      if(criterion.debugLevel()>0) {
        if(level==criterion.maxLevel()) {
          BIGINT unknowns = mlevel->getmat().N();
          unknowns = infoLevel->communicator().sum(unknowns);
          double dunknowns = unknowns.todouble();
          if(rank==0 && criterion.debugLevel()>1) {
            std::cout<<"Level "<<level<<" has "<<dunknowns<<" unknowns, "<<dunknowns/infoLevel->communicator().size()
                     <<" unknowns per proc (procs="<<infoLevel->communicator().size()<<")"<<std::endl;
          }
        }
      }

      if(criterion.accumulate() && !redistributes_.back().isSetup() &&
         infoLevel->communicator().size()>1) {
#if HAVE_MPI && !HAVE_PARMETIS
        if(criterion.accumulate()==successiveAccu &&
           infoLevel->communicator().rank()==0)
          std::cerr<<"Successive accumulation of data on coarse levels only works with ParMETIS installed."
                   <<"  Fell back to accumulation to one domain on coarsest level"<<std::endl;
#endif

        // accumulate to fewer processors
        Matrix* redistMat= new Matrix();
        ParallelInformation* redistComm=0;
        int nodomains = 1;

        repartitionAndDistributeMatrix(mlevel->getmat(), *redistMat, *infoLevel,
                                       redistComm, redistributes_.back(), nodomains,criterion);
        MatrixArgs args(*redistMat, *redistComm);
        BIGINT unknowns = redistMat->N();
        unknowns = infoLevel->communicator().sum(unknowns);

        if(redistComm->communicator().rank()==0 && criterion.debugLevel()>1) {
          double dunknowns= unknowns.todouble();
          std::cout<<"Level "<<level<<" redistributed has "<<dunknowns<<" unknowns, "<<dunknowns/redistComm->communicator().size()
                   <<" unknowns per proc (procs="<<redistComm->communicator().size()<<")"<<std::endl;
        }
        mlevel.addRedistributed(ConstructionTraits<MatrixOperator>::construct(args));
        infoLevel.addRedistributed(redistComm);
        infoLevel->freeGlobalLookup();
      }

      int levels = matrices_.levels();
      maxlevels_ = parallelInformation_.finest()->communicator().max(levels);
      assert(matrices_.levels()==redistributes_.size());
      if(hasCoarsest() && rank==0 && criterion.debugLevel()>1)
        std::cout<<"operator complexity: "<<allnonzeros.todouble()/finenonzeros.todouble()<<std::endl;
      return 0;
    }

    template<class M, class IS, class A>
    const typename MatrixHierarchy<M,IS,A>::ParallelMatrixHierarchy&
    MatrixHierarchy<M,IS,A>::matrices() const
    {
      return matrices_;
    }

    template<class M, class IS, class A>
    const typename MatrixHierarchy<M,IS,A>::ParallelInformationHierarchy&
    MatrixHierarchy<M,IS,A>::parallelInformation() const
    {
      return parallelInformation_;
    }

    template<class M, class IS, class A>
    void MatrixHierarchy<M,IS,A>::getCoarsestAggregatesOnFinest(std::vector<std::size_t>& data) const
    {
      int levels=aggregatesMaps().size();
      int maxlevels=parallelInformation_.finest()->communicator().max(levels);
      std::size_t size=(*(aggregatesMaps().begin()))->noVertices();
      // We need an auxiliary vector for the consecutive prolongation.
      std::vector<std::size_t> tmp;
      std::vector<std::size_t> *coarse, *fine;

      // make sure the allocated space suffices.
      tmp.reserve(size);
      data.reserve(size);

      // Correctly assign coarse and fine for the first prolongation such that
      // we end up in data in the end.
      if(levels%2==0) {
        coarse=&tmp;
        fine=&data;
      }else{
        coarse=&data;
        fine=&tmp;
      }

      // Number the unknowns on the coarsest level consecutively for each process.
      if(levels==maxlevels) {
        const AggregatesMap& map = *(*(++aggregatesMaps().rbegin()));
        std::size_t m=0;

        for(typename AggregatesMap::const_iterator iter = map.begin(); iter != map.end(); ++iter)
          if(*iter< AggregatesMap::ISOLATED)
            m=std::max(*iter,m);

        coarse->resize(m+1);
        std::size_t i=0;
        srand((unsigned)std::clock());
        std::set<size_t> used;
        for(typename std::vector<std::size_t>::iterator iter=coarse->begin(); iter != coarse->end();
            ++iter, ++i)
        {
          std::pair<std::set<std::size_t>::iterator,bool> ibpair
            = used.insert(static_cast<std::size_t>((((double)rand())/(RAND_MAX+1.0)))*coarse->size());

          while(!ibpair.second)
            ibpair = used.insert(static_cast<std::size_t>((((double)rand())/(RAND_MAX+1.0))*coarse->size()));
          *iter=*(ibpair.first);
        }
      }

      typename ParallelInformationHierarchy::Iterator pinfo = parallelInformation().coarsest();
      --pinfo;

      // Now consecutively project the numbers to the finest level.
      for(typename AggregatesMapList::const_reverse_iterator aggregates=++aggregatesMaps().rbegin();
          aggregates != aggregatesMaps().rend(); ++aggregates,--levels) {

        fine->resize((*aggregates)->noVertices());
        fine->assign(fine->size(), 0);
        Transfer<typename AggregatesMap::AggregateDescriptor, std::vector<std::size_t>, ParallelInformation>
        ::prolongateVector(*(*aggregates), *coarse, *fine, static_cast<std::size_t>(1), *pinfo);
        --pinfo;
        std::swap(coarse, fine);
      }

      // Assertion to check that we really projected to data on the last step.
      assert(coarse==&data);
    }

    template<class M, class IS, class A>
    const typename MatrixHierarchy<M,IS,A>::AggregatesMapList&
    MatrixHierarchy<M,IS,A>::aggregatesMaps() const
    {
      return aggregatesMaps_;
    }
    template<class M, class IS, class A>
    const typename MatrixHierarchy<M,IS,A>::RedistributeInfoList&
    MatrixHierarchy<M,IS,A>::redistributeInformation() const
    {
      return redistributes_;
    }

    template<class M, class IS, class A>
    MatrixHierarchy<M,IS,A>::~MatrixHierarchy()
    {
      typedef typename AggregatesMapList::reverse_iterator AggregatesMapIterator;
      typedef typename ParallelMatrixHierarchy::Iterator Iterator;
      typedef typename ParallelInformationHierarchy::Iterator InfoIterator;

      AggregatesMapIterator amap = aggregatesMaps_.rbegin();
      InfoIterator info = parallelInformation_.coarsest();
      for(Iterator level=matrices_.coarsest(), finest=matrices_.finest(); level != finest;  --level, --info, ++amap) {
        (*amap)->free();
        delete *amap;
        delete &level->getmat();
        if(level.isRedistributed())
          delete &(level.getRedistributed().getmat());
      }
      delete *amap;
    }

    template<class M, class IS, class A>
    template<class V, class TA>
    void MatrixHierarchy<M,IS,A>::coarsenVector(Hierarchy<BlockVector<V,TA> >& hierarchy) const
    {
      assert(hierarchy.levels()==1);
      typedef typename ParallelMatrixHierarchy::ConstIterator Iterator;
      typedef typename RedistributeInfoList::const_iterator RIter;
      RIter redist = redistributes_.begin();

      Iterator matrix = matrices_.finest(), coarsest = matrices_.coarsest();
      int level=0;
      if(redist->isSetup())
        hierarchy.addRedistributedOnCoarsest(matrix.getRedistributed().getmat().N());
      Dune::dvverb<<"Level "<<level<<" has "<<matrices_.finest()->getmat().N()<<" unknowns!"<<std::endl;

      while(matrix != coarsest) {
        ++matrix; ++level; ++redist;
        Dune::dvverb<<"Level "<<level<<" has "<<matrix->getmat().N()<<" unknowns!"<<std::endl;

        hierarchy.addCoarser(matrix->getmat().N());
        if(redist->isSetup())
          hierarchy.addRedistributedOnCoarsest(matrix.getRedistributed().getmat().N());

      }

    }

    template<class M, class IS, class A>
    template<class S, class TA>
    void MatrixHierarchy<M,IS,A>::coarsenSmoother(Hierarchy<S,TA>& smoothers,
                                                  const typename SmootherTraits<S>::Arguments& sargs) const
    {
      assert(smoothers.levels()==0);
      typedef typename ParallelMatrixHierarchy::ConstIterator MatrixIterator;
      typedef typename ParallelInformationHierarchy::ConstIterator PinfoIterator;
      typedef typename AggregatesMapList::const_iterator AggregatesIterator;

      typename ConstructionTraits<S>::Arguments cargs;
      cargs.setArgs(sargs);
      PinfoIterator pinfo = parallelInformation_.finest();
      AggregatesIterator aggregates = aggregatesMaps_.begin();
      int level=0;
      for(MatrixIterator matrix = matrices_.finest(), coarsest = matrices_.coarsest();
          matrix != coarsest; ++matrix, ++pinfo, ++aggregates, ++level) {
        cargs.setMatrix(matrix->getmat(), **aggregates);
        cargs.setComm(*pinfo);
        smoothers.addCoarser(cargs);
      }
      if(maxlevels()>levels()) {
        // This is not the globally coarsest level and therefore smoothing is needed
        cargs.setMatrix(matrices_.coarsest()->getmat(), **aggregates);
        cargs.setComm(*pinfo);
        smoothers.addCoarser(cargs);
        ++level;
      }
    }

    template<class M, class IS, class A>
    template<class F>
    void MatrixHierarchy<M,IS,A>::recalculateGalerkin(const F& copyFlags)
    {
      typedef typename AggregatesMapList::iterator AggregatesMapIterator;
      typedef typename ParallelMatrixHierarchy::Iterator Iterator;
      typedef typename ParallelInformationHierarchy::Iterator InfoIterator;

      AggregatesMapIterator amap = aggregatesMaps_.begin();
      GalerkinProduct productBuilder;
      InfoIterator info = parallelInformation_.finest();
      typename RedistributeInfoList::iterator riIter = redistributes_.begin();
      Iterator level = matrices_.finest(), coarsest=matrices_.coarsest();
      if(level.isRedistributed()) {
        info->buildGlobalLookup(info->indexSet().size());
        redistributeMatrixEntries(const_cast<Matrix&>(level->getmat()),
                                  const_cast<Matrix&>(level.getRedistributed().getmat()),
                                  *info,info.getRedistributed(), *riIter);
        info->freeGlobalLookup();
      }

      for(; level!=coarsest; ++amap) {
        const Matrix& fine = (level.isRedistributed() ? level.getRedistributed() : *level).getmat();
        ++level;
        ++info;
        ++riIter;
        const_cast<Matrix&>(level->getmat()) = static_cast<typename M::field_type>(0);
        productBuilder.calculate(fine, *(*amap), const_cast<Matrix&>(level->getmat()), *info, copyFlags);
        if(level.isRedistributed()) {
          info->buildGlobalLookup(info->indexSet().size());
          redistributeMatrixEntries(const_cast<Matrix&>(level->getmat()),
                                    const_cast<Matrix&>(level.getRedistributed().getmat()), *info,
                                    info.getRedistributed(), *riIter);
          info->freeGlobalLookup();
        }
      }
    }

    template<class M, class IS, class A>
    std::size_t MatrixHierarchy<M,IS,A>::levels() const
    {
      return matrices_.levels();
    }

    template<class M, class IS, class A>
    std::size_t MatrixHierarchy<M,IS,A>::maxlevels() const
    {
      return maxlevels_;
    }

    template<class M, class IS, class A>
    bool MatrixHierarchy<M,IS,A>::hasCoarsest() const
    {
      return levels()==maxlevels() &&
             (!matrices_.coarsest().isRedistributed() ||matrices_.coarsest()->getmat().N()>0);
    }

    template<class M, class IS, class A>
    bool MatrixHierarchy<M,IS,A>::isBuilt() const
    {
      return built_;
    }

    template<class T, class A>
    Hierarchy<T,A>::Hierarchy()
      : finest_(0), coarsest_(0), nonAllocated_(0), allocator_(), levels_(0)
    {}

    template<class T, class A>
    Hierarchy<T,A>::Hierarchy(MemberType& first)
      : allocator_()
    {
      finest_ = allocator_.allocate(1,0);
      finest_->element_ = &first;
      finest_->redistributed_ = nullptr;
      nonAllocated_ = finest_;
      coarsest_ = finest_;
      coarsest_->coarser_ = coarsest_->finer_ = nullptr;
      levels_ = 1;
    }

    template<class T, class A>
    Hierarchy<T,A>::Hierarchy(MemberType* first)
      : allocator_()
    {
      finest_ = allocator_.allocate(1,0);
      finest_->element_ = first;
      finest_->redistributed_ = nullptr;
      nonAllocated_ = nullptr;
      coarsest_ = finest_;
      coarsest_->coarser_ = coarsest_->finer_ = nullptr;
      levels_ = 1;
    }
    template<class T, class A>
    Hierarchy<T,A>::~Hierarchy()
    {
      while(coarsest_) {
        Element* current = coarsest_;
        coarsest_ = coarsest_->finer_;
        if(current != nonAllocated_) {
          if(current->redistributed_)
            ConstructionTraits<T>::deconstruct(current->redistributed_);
          ConstructionTraits<T>::deconstruct(current->element_);
        }
        allocator_.deallocate(current, 1);
        current=nullptr;
        //coarsest_->coarser_ = nullptr;
      }
    }

    template<class T, class A>
    Hierarchy<T,A>::Hierarchy(const Hierarchy& other)
    : nonAllocated_(), allocator_(other.allocator_),
      levels_(other.levels_)
    {
      if(!other.finest_)
      {
        finest_=coarsest_=nonAllocated_=nullptr;
        return;
      }
      finest_=allocator_.allocate(1,0);
      Element* finer_         = nullptr;
      Element* current_      = finest_;
      Element* otherCurrent_ = other.finest_;

      while(otherCurrent_)
      {
        T* t=new T(*(otherCurrent_->element_));
        current_->element_=t;
        current_->finer_=finer_;
        if(otherCurrent_->redistributed_)
          current_->redistributed_ = new T(*otherCurrent_->redistributed_);
        else
          current_->redistributed_= nullptr;
        finer_=current_;
        if(otherCurrent_->coarser_)
        {
          current_->coarser_=allocator_.allocate(1,0);
          current_=current_->coarser_;
        }else
          current_->coarser_=nullptr;
        otherCurrent_=otherCurrent_->coarser_;
      }
      coarsest_=current_;
    }

    template<class T, class A>
    std::size_t Hierarchy<T,A>::levels() const
    {
      return levels_;
    }

    template<class T, class A>
    void Hierarchy<T,A>::addRedistributedOnCoarsest(Arguments& args)
    {
      coarsest_->redistributed_ = ConstructionTraits<MemberType>::construct(args);
    }

    template<class T, class A>
    void Hierarchy<T,A>::addCoarser(Arguments& args)
    {
      if(!coarsest_) {
        assert(!finest_);
        coarsest_ = allocator_.allocate(1,0);
        coarsest_->element_ = ConstructionTraits<MemberType>::construct(args);
        finest_ = coarsest_;
        coarsest_->finer_ = nullptr;
      }else{
        coarsest_->coarser_ = allocator_.allocate(1,0);
        coarsest_->coarser_->finer_ = coarsest_;
        coarsest_ = coarsest_->coarser_;
        coarsest_->element_ = ConstructionTraits<MemberType>::construct(args);
      }
      coarsest_->redistributed_ = nullptr;
      coarsest_->coarser_=nullptr;
      ++levels_;
    }


    template<class T, class A>
    void Hierarchy<T,A>::addFiner(Arguments& args)
    {
      if(!finest_) {
        assert(!coarsest_);
        finest_ = allocator_.allocate(1,0);
        finest_->element = ConstructionTraits<T>::construct(args);
        coarsest_ = finest_;
        coarsest_->coarser_ = coarsest_->finer_ = nullptr;
      }else{
        finest_->finer_ = allocator_.allocate(1,0);
        finest_->finer_->coarser_ = finest_;
        finest_ = finest_->finer_;
        finest_->finer = nullptr;
        finest_->element = ConstructionTraits<T>::construct(args);
      }
      ++levels_;
    }

    template<class T, class A>
    typename Hierarchy<T,A>::Iterator Hierarchy<T,A>::finest()
    {
      return Iterator(finest_);
    }

    template<class T, class A>
    typename Hierarchy<T,A>::Iterator Hierarchy<T,A>::coarsest()
    {
      return Iterator(coarsest_);
    }

    template<class T, class A>
    typename Hierarchy<T,A>::ConstIterator Hierarchy<T,A>::finest() const
    {
      return ConstIterator(finest_);
    }

    template<class T, class A>
    typename Hierarchy<T,A>::ConstIterator Hierarchy<T,A>::coarsest() const
    {
      return ConstIterator(coarsest_);
    }
    /** @} */
  } // namespace Amg
} // namespace Dune

#endif
