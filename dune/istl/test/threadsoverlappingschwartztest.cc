// 1D FEM scheme for equation -u''=f with Dirichlet boundary conditions to test threads

#define WORLDDIM 1
#define GRIDDIM 1

#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>

#include <thread>
//#include <mutex>

#include <dune/common/fvector.hh>
#include <dune/common/dynmatrix.hh>
#include <dune/common/dynvector.hh>

// problem definition
// exact solution x^2+1
#define COOR_X0 0.0
#define COOR_X1 1.0
inline double f(double&& x){return 2.0;}
double ux0(1.0);
double ux1(2.0);

// threads control
#define NUM_THREADS 4
#define GRID_ELEMENTS_PER_THREAD 2
#define DEBUG_THREADS_TEST 1

// basis functions
inline double phi0(double&& x){return 1.0-x;}
inline double phi1(double&& x){return x;}
inline double derphi0(double&& x){return -1.0;}
inline double derphi1(double&& x){return 1.0;}

// flag
enum flags{shared,NOTshared};

// assemble function
template<typename AType,typename bType,typename IType,typename GType,typename FType>
void assemble(unsigned int tid,AType& A_loc,bType& b_loc,IType& idx,GType& grid,unsigned int numGridElements,FType& phi,FType& derphi){

    std::vector<std::vector<double>> A_entity(2,std::vector<double>(2,0.0));
    std::vector<double> b_entity(2,0.0);

    size_t startElem(tid*numGridElements);
    size_t endElem((tid+1)*numGridElements);
    for(size_t elem=startElem;elem!=endElem;++elem){

        // assemble A_enity and b_entity
        for(size_t i=0;i!=2;++i){

            b_entity[i]=0.0;
            // using trapezoid rule
            b_entity[i]+=0.5*(phi[i](0.0)*f(0.0));
            b_entity[i]+=0.5*(phi[i](1.0)*f(1.0));
            b_entity[i]*=(grid[elem+1]-grid[elem]);

            for(size_t j=0;j!=2;++j){
                A_entity[i][j]=0.0;
                // using trapezoid rule
                A_entity[i][j]+=0.5*(derphi[i](0.0)*derphi[j](0.0));
                A_entity[i][j]+=0.5*(derphi[i](1.0)*derphi[j](1.0));
                A_entity[i][j]/=(grid[elem+1]-grid[elem]);
            }

        }

        // add entity contributions to local A_loc and local b_loc
        for(size_t i=0;i!=2;++i){
            b_loc[elem-startElem+i]+=b_entity[i];
            for(size_t j=0;j!=2;++j){
                A_loc[elem-startElem+i][elem-startElem+j]+=A_entity[i][j];
            }
        }

    }

}

// push function
template<typename AType,typename bType>
void push(unsigned int tid,AType& A_loc,bType& b_loc,AType& A,bType& b){

    size_t size(b_loc.size());
    size_t offset((size-1)*tid);
    for(size_t i=0;i!=size;++i){
        b[i+offset]+=b_loc[i];
        for(size_t j=0;j!=size;++j) A[i+offset][j+offset]+=A_loc[i][j];
    }

}

// print vector function
template<typename VType>
void printVector(VType& vector){
    size_t size(vector.size());
    for(size_t i=0;i!=size;++i) std::cout<<vector[i]<<" ";
    std::cout<<std::endl<<std::endl;
}

// print matrix function
template<typename MType>
void printMatrix(MType& matrix){
    size_t size(matrix.size());
    for(size_t i=0;i!=size;++i){
        for(size_t j=0;j!=size;++j) std::cout<<matrix[i][j]<<" ";
        std::cout<<std::endl;
    }
    std::cout<<std::endl;
}

int main(void){

    // initial description
    std::cout<<"FEM scheme over "<<GRIDDIM<<"D grid in a "<<WORLDDIM<<"D world to solve equation -u''=f with Dirichlet boundary conditions u("<<COOR_X0<<")="<<ux0<<" and u("<<COOR_X1<<")="<<ux1<<"."<<std::endl<<std::endl;

    // geometry definition
    typedef Dune::FieldVector<double,WORLDDIM> CoordType;
    CoordType x0(COOR_X0);
    CoordType x1(COOR_X1);

    if(x0[0]>x1[0]){
        std::swap(x0[0],x1[0]);
        std::swap(ux0,ux1);
    }

    std::cout<<"Coordinate of the starting point of the grid : "<<x0<<std::endl;
    std::cout<<"Coordinate of the ending point of the grid : "<<x1<<std::cout<<std::endl;

    // number of threads
    unsigned int numThreads((NUM_THREADS<2?2:NUM_THREADS));
    std::cout<<"Number of threads to use : "<<numThreads<<std::endl;

    // number of grid elements managed by each thread
    unsigned int numGridElementsPerThread((GRID_ELEMENTS_PER_THREAD<1?1:GRID_ELEMENTS_PER_THREAD));
    std::cout<<"Number of grid elements for each thread : "<<numGridElementsPerThread<<std::endl<<std::endl;

    // grid creation
    unsigned int numNodes(numThreads*numGridElementsPerThread+1);
    CoordType x1x0Vector(x1-x0);
    double deltax(x1x0Vector.two_norm()/(numNodes-1));
    typedef std::vector<CoordType> GridType;
    GridType grid(numNodes,x0);

    for(size_t i=1;i!=numNodes;++i) grid[i]+=(deltax*i);

    #ifdef DEBUG_THREADS_TEST
    #if DEBUG_THREADS_TEST
    std::cout<<"DEBUG: grid coordinates"<<std::endl;
    for(std::vector<CoordType>::iterator it=grid.begin();it!=grid.end();++it) std::cout<<*it<<" ";
    std::cout<<std::endl<<std::endl;
    #endif
    #endif

    // set type of index
    typedef std::vector<flags> ThreadsIndexType;
    ThreadsIndexType tit(numNodes,NOTshared);

    for(size_t i=0;i!=(numThreads-1);++i) tit[(i+1)*numGridElementsPerThread]=shared;

    #ifdef DEBUG_THREADS_TEST
    #if DEBUG_THREADS_TEST
    std::cout<<"DEBUG: shared-NOTshared nodes"<<std::endl;
    for(ThreadsIndexType::iterator it=tit.begin();it!=(tit.end()-1);++it){
        if(*it==NOTshared) std::cout<<"N--";
        if(*it==shared) std::cout<<"S--";
    }
    if(*(tit.rbegin())==NOTshared) std::cout<<"N"<<std::endl<<std::endl;
    if(*(tit.rbegin())==shared) std::cout<<"S"<<std::endl<<std::endl;
    #endif
    #endif

    // set color (each row contains all the thread with the same color)
    unsigned int numColors(2); // 0 when tid is even, 1 when tid is odd
    std::vector<std::vector<unsigned int>> colors(numColors);
    for(size_t i=0;i!=2;++i){
        colors[i].resize((numThreads+1*(1-i))/2);
        for(size_t j=0;j!=colors[i].size();++j) colors[i][j]=j*2+1*i;
    }

    #ifdef DEBUG_THREADS_TEST
    #if DEBUG_THREADS_TEST
    std::cout<<"DEBUG: colors"<<std::endl;
    for(std::vector<std::vector<unsigned int>>::iterator rowIt=colors.begin();rowIt!=colors.end();++rowIt){
        std::cout<<"Threads which have color "<<rowIt-colors.begin()<<" : ";
        for(std::vector<unsigned int>::iterator it=rowIt->begin();it!=rowIt->end();++it) std::cout<<*it<<" ";
        std::cout<<std::endl;
    }
    std::cout<<std::endl;
    #endif
    #endif

    // allocate stiffness matrix A, RHS vector b and solution vector x
    typedef Dune::DynamicMatrix<double> StiffnessMatrixType;
    StiffnessMatrixType A(numNodes,numNodes,0.0);

    typedef Dune::DynamicVector<double> VectorType;
    VectorType b(numNodes,0.0);
    VectorType x(numNodes,0.0);

    // allocate local stiffness matriices A_loc, local RHS vectors b_loc and local solution vectors x_loc; here local is referred to thread scope
    std::vector<StiffnessMatrixType> A_loc(numThreads,StiffnessMatrixType(numGridElementsPerThread+1,numGridElementsPerThread+1,0.0));
    std::vector<VectorType> b_loc(numThreads,VectorType(numGridElementsPerThread+1,0.0));
    std::vector<VectorType> x_loc(numThreads,VectorType(numGridElementsPerThread+1,0.0));

    // basis functions
    typedef std::function<double(double&&)> FunctionType;

    std::vector<FunctionType> phi(2);
    phi[0]=phi0; // phi[0]=[](double& x)->double{return 1.0-x;};
    phi[1]=phi1; // phi[1]=[](double& x)->double{return x;};

    std::vector<FunctionType> derphi(2);
    derphi[0]=derphi0; // derphi[0]=[](double& x)->double{return -1.0;};
    derphi[1]=derphi1; // derphi[1]=[](double& x)->double{return 1;};

    // launch a group of threads to assemble the local stiffness matrices and the local RHS vectors
    std::vector<std::thread> t(numThreads);

    for(size_t i=0;i!=numThreads;++i) t[i]=std::thread(assemble<StiffnessMatrixType,VectorType,ThreadsIndexType,GridType,std::vector<FunctionType>>,i,std::ref(A_loc[i]),std::ref(b_loc[i]),std::ref(tit),std::ref(grid),numGridElementsPerThread,std::ref(phi),std::ref(derphi));
    for(size_t i=0;i!=numThreads;++i) t[i].join();

    // launch a group of threads to update the global stiffness matrix and the global RHS with the valus stored in the local ones;
    for(size_t i=0;i!=numColors;++i){
        for(size_t j=0;j!=colors[i].size();++j) t[j]=std::thread(push<StiffnessMatrixType,VectorType>,colors[i][j],std::ref(A_loc[i]),std::ref(b_loc[i]),std::ref(A),std::ref(b));
        for(size_t j=0;j!=colors[i].size();++j) t[j].join();
    }

    // impose boundary condition
    A[0][0]=1;
    for(size_t i=1;i!=numNodes;++i) A[0][i]=0;
    b[0]=ux0;

    A[numNodes-1][numNodes-1]=1;
    for(size_t i=0;i!=(numNodes-1);++i) A[numNodes-1][i]=0;
    b[numNodes-1]=ux1;

    #ifdef DEBUG_THREADS_TEST
    #if DEBUG_THREADS_TEST
    std::cout<<"DEBUG: global stiffness matrix"<<std::endl;
    printMatrix(A);
    std::cout<<"DEBUG: global RHS"<<std::endl;
    printVector(b);
    #endif
    #endif

    // solve the problem
    A.solve(x,b);

    #ifdef DEBUG_THREADS_TEST
    #if DEBUG_THREADS_TEST
    std::cout<<"DEBUG: global solution"<<std::endl;
    printVector(x);
    #endif
    #endif

    return 0;

}
