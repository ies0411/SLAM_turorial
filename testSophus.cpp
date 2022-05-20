#include <cmath>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <iostream>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

using namespace std;

int main(int argc, char const *argv[]) {
    // Z축을 따라 90도 회전된 회전 행렬
    Eigen::Matrix3d R = Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 0, 1)).toRotationMatrix();

    Sophus::SO3<double> SO3_R(R);  /// Sophus::SO(3)는 회전 행렬에서 직접 구성할 수 있습니다.
    // Sophus::SO3<double> SO3_v(0, 0, M_PI / 2);  // 회전 벡터에서 생성할 수도 있습니다.
    Eigen::Quaterniond q(R);  // 또는 쿼터니언
    Sophus::SO3<double> SO3_q(q);
    // 위의 표현은 모두 동일합니다.
    // SO(3) 출력 시 so(3) 형식으로 출력
    cout << "SO(3) from matrix: \n"
         << SO3_R.log() << endl;
    // cout << "SO(3) from vector: " << SO3_v.log() << endl;
    cout << "SO(3) from quaternion :\n"
         << SO3_q.log() << endl;
    //로그매핑
    Eigen::Vector3d so3 = SO3_R.log();
    cout << "so3 = " << so3.transpose() << endl;

    // hat은 대칭 행렬에 대한 벡터입니다.
    cout << "so3 hat=\n"
         << Sophus::SO3<double>::hat(so3) << endl;
    // 반대로 ve는 벡터에 대해 비대칭입니다.
    cout << "so3 hat vee= \n"
         << Sophus::SO3<double>::vee(Sophus::SO3<double>::hat(so3)).transpose() << endl;

    //
    Eigen::Vector3d update_so3(1e-4, 0, 0);
    Sophus::SO3<double> SO3_updated = Sophus::SO3<double>::exp(update_so3) * SO3_R;
    cout << "updated = " << SO3_updated.log() << endl;

    Eigen::Vector3d t(1, 0, 0);
    Sophus::SE3<double> SE3_Rt(R, t);
    Sophus::SE3<double> SE3_qt(q, t);

    cout << "se3\n"
         << SE3_Rt.log() << endl;
    cout << "se3 ver2\n"
         << SE3_qt.log() << endl;
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    Vector6d se3 = SE3_Rt.log();
    cout << "se3 = " << se3.transpose() << endl;

    cout << "se3 hat = " << endl
         << Sophus::SE3<double>::hat(se3) << endl;
    cout << "se3 hat vee = " << Sophus::SE3<double>::vee(Sophus::SE3<double>::hat(se3)).transpose() << endl;

    //섭동모델
    Vector6d update_se3;
    update_se3.setZero();
    update_se3(0, 0) = 1e-4d;  //업데이트 양이 이 정도라고 가정해 봅시다.
    Sophus::SE3<double> SE3_updated = Sophus::SE3<double>::exp(update_se3) * SE3_Rt;
    cout << "SE3 updated = " << endl
         << SE3_updated.matrix() << endl;
    return 0;
}
