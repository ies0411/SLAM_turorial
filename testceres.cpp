#include <ceres/ceres.h>

#include <chrono>
#include <iostream>
#include <opencv2/core/core.hpp>

#include "glog/logging.h"

struct CURVE_FITTING_COST {
    CURVE_FITTING_COST(double x, double y) : _x(x), _y(y) {}
    const double _x, _y;
    template <typename T>
    bool operator()(const T* const abc, T* residual) const {
        residual[0] = T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]);  // y-exp(ax^2+bx+c)
        return true;
    }
};
struct CostFunctor {
    template <typename T>
    bool operator()(const T* const x, T* residual) const {
        residual[0] = 10.0 - x[0];
        return true;
    }
};
struct CostFunctor2 {
    template <typename T>
    bool operator()(const T* const x, T* residual) const {
        // 첫 번째 cost function:  0.5 (10 - x)^2
        residual[0] = 10.0 - x[0];
        // 두 번째 cost function:  0.5 (x)^2
        residual[1] = x[0];
        return true;
    }
};

int main(int argc, char const* argv[]) {
    google::InitGoogleLogging(argv[0]);
#if 0
    double x = -100.7893123;
    const double initial_x = x;
    /***case 1***/


    // 1.problem
    ceres::Problem problem1;

    // 2.costfunction
    // autodiffcostfinction - jacobian
    ceres::CostFunction* cost_function =
        new ceres::AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);
    problem1.AddResidualBlock(cost_function, NULL, &x);

    // 3. solve
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem1, &summary);
    std::cout << summary.BriefReport() << "\n";
    std::cout << "x : " << initial_x
              << " -> " << x << "\n";

    /***case 2***/
    x = 0.5;
    const double initial_x2 = x;
    ceres::Problem problem2;
    ceres::CostFunction* cost_function2 =
        new ceres::AutoDiffCostFunction<CostFunctor2, 2, 1>(new CostFunctor2);
    problem2.AddResidualBlock(cost_function2, NULL, &x);

    ceres::Solver::Options options2;
    options2.function_tolerance = 1e-10;
    options2.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary2;
    ceres::Solve(options2, &problem2, &summary2);

    std::cout << summary2.BriefReport() << "\n";
    std::cout << "x : " << initial_x2
              << " -> " << x << "\n";
#endif
/****case****/
#if 1
    double a = 1.0, b = 2.0, c = 1.0;
    int N = 200;
    double w_sigma = 1.0;
    //난수생성
    cv::RNG rng;
    double abc[3] = {0, 0, 0};
    std::vector<double> x_data, y_data;
    std::cout << "generating data" << std::endl;

    for (int i = 0; i < N; i++) {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(std::exp(a * x * x + b * x + c) + rng.gaussian(w_sigma));
    }

    //최소제곱문제
    ceres::Problem problem;
    for (int i = 0; i < N; i++) {
        problem.AddResidualBlock(new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>(new CURVE_FITTING_COST(x_data[i], y_data[i])), nullptr, abc);
    }
    // sover
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;  //증분 방정식을 푸는 방법
    options.minimizer_progress_to_stdout = true;   //출력 ok

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);  //최적화시작

    //출력결과
    std::cout << summary.BriefReport() << std::endl;
    for (auto& p : abc) std::cout << p << " ";
#endif
    return 0;
}
