// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"

#include "anisotropic.hh"
#include <dune/common/timer.hh>
#include <dune/common/parallel/indexset.hh>
#include <dune/common/parallel/collectivecommunication.hh>
#include <dune/istl/overlappingschwarz.hh>
#include <dune/istl/paamg/fastamg.hh>
#include <dune/istl/paamg/pinfo.hh>
#include <dune/istl/solvers.hh>
#include <cstdlib>
#include <ctime>

template<class M, class V>
void randomize(const M& mat, V& b)
{
  V x=b;

  srand((unsigned)std::clock());

  typedef typename V::iterator iterator;
  for(iterator i=x.begin(); i != x.end(); ++i)
    *i=(rand() / (RAND_MAX + 1.0));

  mat.mv(static_cast<const V&>(x), b);
}

template <int BS>
void testAMG(int N, int coarsenTarget, int ml)
{

  std::cout<<"N="<<N<<" coarsenTarget="<<coarsenTarget<<" maxlevel="<<ml<<std::endl;


  typedef Dune::ParallelIndexSet<int,LocalIndex,512> ParallelIndexSet;

  ParallelIndexSet indices;
  typedef Dune::FieldMatrix<double,BS,BS> MatrixBlock;
  typedef Dune::BCRSMatrix<MatrixBlock> BCRSMat;
  typedef Dune::FieldVector<double,BS> VectorBlock;
  typedef Dune::BlockVector<VectorBlock> Vector;
  typedef Dune::MatrixAdapter<BCRSMat,Vector,Vector> Operator;
  typedef Dune::CollectiveCommunication<void*> Comm;
  int n;

  Comm c;
  BCRSMat mat = setupAnisotropic2d<BS,double>(N, indices, c, &n, 1);

  Vector b(mat.N()), x(mat.M());

  b=0;
  x=100;

  setBoundary(x, b, N);

  x=0;
  randomize(mat, b);

  if(N<6) {
    Dune::printmatrix(std::cout, mat, "A", "row");
    Dune::printvector(std::cout, x, "x", "row");
    Dune::printvector(std::cout, b, "b", "row");
  }

  Dune::Timer watch;

  watch.reset();
  Operator fop(mat);

  typedef Dune::Amg::AggregationCriterion<Dune::Amg::SymmetricMatrixDependency<BCRSMat,Dune::Amg::FirstDiagonal> > CriterionBase;
  typedef Dune::Amg::CoarsenCriterion<CriterionBase> Criterion;

  Criterion criterion(15,coarsenTarget);
  criterion.setDefaultValuesIsotropic(2);
  criterion.setAlpha(.67);
  criterion.setBeta(1.0e-4);
  criterion.setMaxLevel(ml);
  criterion.setSkipIsolated(false);

  Dune::SeqScalarProduct<Vector> sp;

//   // smoothers reusulting in a symmetric AMG work like a charm and are 10% faster even when not optimized
//   typedef Dune::SeqSSOR<typename Operator::matrix_type, Vector, Vector> Smoother;
//   typedef Dune::SeqSOR<typename Operator::matrix_type, Vector, Vector> Smoother;
//   typedef Dune::SeqOverlappingSchwarz<BCRSMat,Vector,Dune::SymmetricMultiplicativeSchwarzMode> Smoother;
//   typedef Dune::SeqGS<typename Operator::matrix_type, Vector, Vector> Smoother;
//   // the optimized ones also work (set relaxationFactor to 0.5 or so for Jacobi)
//   typedef Dune::Amg::GaussSeidelWithDefect<typename Operator::matrix_type, Vector, Vector> Smoother;
//   typedef Dune::Amg::JacobiWithDefect<typename Operator::matrix_type, Vector, Vector> Smoother;
//
//   // non-symmetric ones just dont work (in neither case)
//   typedef Dune::SeqILU0<typename Operator::matrix_type, Vector, Vector> Smoother;
//   typedef Dune::SeqILUn<BCRSMat,Vector,Vector> Smoother;
//   typedef Dune::Amg::ILUnWithDefect<typename Operator::matrix_type, Vector, Vector> Smoother;

  typedef Dune::Amg::FastAMG<Operator,Vector, Smoother> AMG;
  Dune::Amg::Parameters parms;

  typedef typename Dune::Amg::SmootherTraits<Smoother>::Arguments SmootherArgs;
  SmootherArgs smootherArgs;
  smootherArgs.iterations = 1;
  smootherArgs.relaxationFactor = 1.0;

  AMG amg(fop, criterion, parms,smootherArgs);

  double buildtime = watch.elapsed();

  std::cout<<"Building hierarchy took "<<buildtime<<" seconds"<<std::endl;

  Dune::GeneralizedPCGSolver<Vector> amgCG(fop,amg,1e-6,80,2);
//   Dune::LoopSolver<Vector> amgCG(fop, amg, 1e-4, 10000, 2);
  watch.reset();
  Dune::InverseOperatorResult r;
  amgCG.apply(x,b,r);

  double solvetime = watch.elapsed();

  std::cout<<"AMG solving took "<<solvetime<<" seconds"<<std::endl;

  std::cout<<"AMG building took "<<(buildtime/r.elapsed*r.iterations)<<" iterations"<<std::endl;
  std::cout<<"AMG building together with solving took "<<buildtime+solvetime<<std::endl;

  /*
     watch.reset();
     cg.apply(x,b,r);

     std::cout<<"CG solving took "<<watch.elapsed()<<" seconds"<<std::endl;
   */
}


int main(int argc, char** argv)
{

  int N=1000;
  int coarsenTarget=1200;
  int ml=10;

  if(argc>1)
    N = atoi(argv[1]);

  if(argc>2)
    coarsenTarget = atoi(argv[2]);

  if(argc>3)
    ml = atoi(argv[3]);

  testAMG<1>(N, coarsenTarget, ml);
  //testAMG<4>(N, coarsenTarget, ml);

}
