#include <ctime>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>  // header file
#include <eigen3/Eigen/Geometry>
#include <iostream>
#define MATRIX_SIZE 100
using namespace std;
int main(int argc, char const *argv[]) {
    /***Eigen tutorial*****/
    Eigen::Matrix<float, 2, 3> matrix_23;
    Eigen::Vector3d v_3d;
    Eigen::Matrix<float, 3, 1> vd_3d;
    Eigen::Matrix3d matrix_33 = Eigen::Matrix3d::Zero();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix_dynamic;
    Eigen::MatrixXd matrix_x;
    matrix_23 << 1, 2, 3, 4, 5, 6;
    std::cout << matrix_23 << std::endl;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++)
            std::cout << matrix_23(i, j) << "\t";
        std::cout << std::endl;
    }
    v_3d << 3, 2, 1;
    vd_3d << 4, 5, 6;
    Eigen::Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d;
    std::cout << ";" << result << std::endl;
    // wrong :  Eigen::Matrix<double, 2, 3> result_wrong_dimension = matrix_23.cast<double>() * v_3d;

    Eigen::Matrix<float, 2, 1> result2 = matrix_23 * vd_3d;
    std::cout << result2 << std::endl;

    Eigen::Matrix2d matrix_22;
    // matrix_33 = Eigen::Matrix2i::Random();
    matrix_22 << 3.0, 2.0, 3.0, 4.0;
    cout << matrix_22 << endl
         << endl;

    cout << matrix_22.transpose() << endl;
    cout << matrix_22.sum() << endl;
    cout << matrix_22.trace() << endl;
    cout << 10 * matrix_22 << endl;
    cout << matrix_22.inverse() << endl;
    cout << matrix_22.determinant() << endl;

    Eigen::EigenSolver<Eigen::Matrix<double, 2, 2> > s(matrix_22);

    std::cout << "eigenvalues:" << std::endl;
    std::cout << s.eigenvalues() << std::endl;
    std::cout << "eigenvectors=" << std::endl;
    std::cout << s.eigenvectors() << std::endl;

    // EigenSsolver사용해서 eigenvalue구하는게 더 정확함

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigen_solver;
    eigen_solver.compute(matrix_22);
    cout << "Eigen values = \n"
         << eigen_solver.eigenvalues() << endl;
    cout << "Eigen vectors = \n"
         << eigen_solver.eigenvectors() << endl;
    //// QR decomposition
    Eigen::Matrix<double, MATRIX_SIZE, MATRIX_SIZE> matrix_NN;
    matrix_NN = Eigen::MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
    Eigen::Matrix<double, MATRIX_SIZE, 1> v_Nd;
    v_Nd = Eigen::MatrixXd::Random(MATRIX_SIZE, 1);

    clock_t time_stt = clock();

    Eigen::Matrix<double, MATRIX_SIZE, 1> x = matrix_NN.inverse() * v_Nd;
    cout << "time use in normal inverse is " << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms" << endl;

    time_stt = clock();
    x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
    cout << "time use in Qr decomposition is " << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms" << endl;
    /**좌표변환***/

    // Eigen/Geometry 모듈은 다양한 회전 및 변환 표현을 제공합니다.
    // 3D 회전 행렬은 Matrix3d ​​또는 Matrix3f를 직접 사용합니다.
    Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity();
    // Z축을 따라 45도 회전
    //  회전 벡터는 AngleAxis를 사용합니다. 기본 레이어는 직접 Matrix가 아니지만 연산을 행렬로 사용할 수 있습니다(연산자가 오버로드되기 때문에).
    Eigen::AngleAxisd rotation_vector(M_PI / 2, Eigen::Vector3d(0, 0, 1));
    cout.precision(3);
    cout << "rotation matrix =\n"
         << rotation_vector.matrix() << endl;  //用matrix()转换成矩阵
    rotation_matrix = rotation_vector.toRotationMatrix();

    // AngleAxis로 좌표변환 가능 , 또는 회전 행렬로

    Eigen::Vector3d v(1, 0, 0);
    Eigen::Vector3d v_rotated = rotation_vector * v;
    cout << "(1,0,0) after rotation = " << v_rotated.transpose() << endl;
    v_rotated = rotation_matrix * v;
    cout << "(1,0,0) after rotation = " << v_rotated.transpose() << endl;

    // 오일러 각도: 회전 행렬을 오일러 각도로 직접 변환할 수 있습니다.

    Eigen::Vector3d euler_angles = rotation_matrix.eulerAngles(2, 1, 0) * 180. / M_PI;  // ZYX 순서, 즉 롤 피치 요 순서
    cout << "yaw pitch roll = " << euler_angles.transpose() << endl;
    euler_angles = rotation_matrix.eulerAngles(1, 2, 0) * 180. / M_PI;  // ZYX 순서, 즉 롤 피치 요 순서
    cout << "pitch yaw roll = " << euler_angles.transpose() << endl;

    // Eigen::Isometry를 사용한 유클리드 변환 행렬 - homogeneous matrix
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();  // 3d라고 하지만 본질적으로 4*4 행렬입니다.
    T.rotate(rotation_vector);                            // rotation_vector에 따라 회전
    T.pretranslate(Eigen::Vector3d(1, 3, 4));             // 변환 벡터를 (1,3,4)로 설정
    cout << "Transform matrix = \n"
         << T.matrix() << endl;

    // 변환 행렬을 사용한 좌표 변환
    Eigen::Vector3d v_transformed = T * v;  // R*v+t
    cout << "v tranformed = " << v_transformed.transpose() << endl;

    // 아핀 및 사영 변환의 경우 Eigen::Affine3d 및 Eigen::Projective3d 사용, 생략
    // 쿼터니언
    // AngleAxis를 쿼터니언에 직접 할당하거나 그 반대로 할당할 수 있습니다.
    Eigen::Quaterniond q = Eigen::Quaterniond(rotation_vector);
    cout << "quaternion = \n"
         << q.coeffs() << endl;  //계수의 순서는 (x, y, z, w)이고 w는 실수부이고 처음 세 개는 허수부입니다.

    q = Eigen::Quaterniond(rotation_matrix);  // 회전 행렬을 할당할 수도 있습니다.
    cout << "quaternion = \n"
         << q.coeffs() << endl;
    // 쿼터니언을 사용하여 벡터를 회전하고 오버로드된 곱셈을 사용합니다.
    v_rotated = q * v;  // 수학적으로 qvq^{-1}입니다.
    cout << "(1,0,0) after rotation = " << v_rotated.transpose() << endl;
    return 0;
}
