#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/eigen_types.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
// #include <g2o/core

// #include <Eigen/Core>
#include <chrono>
#include <cmath>
#include <eigen3/Eigen/Core>
#include <iostream>
#include <opencv2/core/core.hpp>

using namespace std;
//곡선 모델의 정점, 템플릿 매개변수: 가변 차원 및 데이터 유형 최적화
class CurveFittingVertex : public g2o::BaseVertex<3, Eigen::Vector3d> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    //초기화
    virtual void setToOriginImpl() {
        _estimate << 0, 0, 0;
    }
    virtual void oplusImpl(const double* update) {
        _estimate += Eigen::Vector3d(update);
    }
    virtual bool read(istream& in) {}
    virtual bool write(ostream& out) const {}
};

class CurveFittingEdge : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CurveFittingEdge(double x) : BaseUnaryEdge(), _x(x) {}
    void computeError() {
        const CurveFittingVertex* v = static_cast<const CurveFittingVertex*>(_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();
        _error(0, 0) = _measurement - std::exp(abc(0, 0) * _x * _x + abc(1, 0) * _x + abc(2, 0));
    }
    virtual bool read(istream& in) {}
    virtual bool write(ostream& out) const {}

   public:
    double _x;
};

int main(int argc, char const* argv[]) {
    double a = 1.0, b = 2.0, c = 1.0;
    int N = 100;
    double w_sigma = 1.0;
    cv::RNG rng;
    double abc[3] = {0, 0, 0};

    vector<double> x_data, y_data;
    cout << "generating data" << endl;
    for (int i; i < N; i++) {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(a * x * x + b * x + c) + rng.gaussian(w_sigma));
        cout << x_data[i] << ", " << y_data[i] << endl;
    }
    // 그래프 최적화 빌드, 먼저 g2o 설정

    // 각 오차항 최적화 변수의 차원은 3, 오차값의 차원은 1
    // typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1> > Block;
    // 선형 방정식 솔버
    std::unique_ptr<g2o::BlockSolverX::LinearSolverType> linearSolver(new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>());
    //행렬 블록 솔버
    std::unique_ptr<g2o::BlockSolverX> solver_ptr(new g2o::BlockSolverX(std::move(linearSolver)));
    // 경사하강법, GN, LM, DogLeg 중에서 선택
    // g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    // g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr );
    // g2o::OptimizationAlgorithmDogleg* solver = new g2o::OptimizationAlgorithmDogleg( solver_ptr );
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));

    g2o::SparseOptimizer optimizer;  // 그래프 모델
    optimizer.setAlgorithm(solver);  //솔버 설정
    optimizer.setVerbose(true);      // 디버그 출력 켜기

    //그래프에 정점 추가
    CurveFittingVertex* v = new CurveFittingVertex();
    v->setEstimate(Eigen::Vector3d(0, 0, 0));
    v->setId(0);
    optimizer.addVertex(v);
    //그래프에 간선 추가
    for (int i = 0; i < N; i++) {
        CurveFittingEdge* edge = new CurveFittingEdge(x_data[i]);
        edge->setId(i);
        edge->setVertex(0, v);            //연결된 정점 설정
        edge->setMeasurement(y_data[i]);  //관측값
        // 정보 행렬: 공분산 행렬의 역행렬
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1 / (w_sigma * w_sigma));
        optimizer.addEdge(edge);
    }
    //최적화 수행
    cout << "start optimization" << endl;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.optimize(100);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve time cost : " << time_used.count() << "second" << endl;
    //출력 최적화 값
    Eigen::Vector3d abc_estimate = v->estimate();
    cout << "estimated model : " << abc_estimate.transpose() << endl;

    return 0;
}
