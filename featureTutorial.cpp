// #include <g2o/core/base_unary_edge.h>
// // #include <g2o/core/base_binary_edge.h>
// #include <g2o/core/base_vertex.h>
// #include <g2o/core/block_solver.h>
// #include <g2o/core/optimization_algorithm_levenberg.h>
// #include <g2o/solvers/csparse/linear_solver_csparse.h>
// #include <g2o/types/sba/types_six_dof_expmap.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <chrono>
#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

// feature
void findFeatureMatches(const cv::Mat& img_1, const cv::Mat& img_2,
                        std::vector<cv::KeyPoint>& keypoints_1, std::vector<cv::KeyPoint>& keypoints_2,
                        std::vector<cv::DMatch>& matches) {
    cv::Mat descriptors_1, descriptors_2;

    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    std::vector<cv::DMatch> match;
    matcher->match(descriptors_1, descriptors_2, match);

    double min_dist = 10000, max_dist = 0;
    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = match[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    for (int i = 0; i < descriptors_1.rows; i++) {
        if (match[i].distance <= std::max(2 * min_dist, 30.0)) {
            matches.push_back(match[i]);
        }
    }
}

void pose_estimation_2d2d(std::vector<cv::KeyPoint> keypoint_1, std::vector<cv::KeyPoint> keypoint_2, std::vector<cv::DMatch> matches,
                          cv::Mat& R, cv::Mat& t) {
    //카메라 내부 매개변수 intrinsic parameter
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    //일치하는 점을 vector<Point2f> 형식으로 변환합니다.
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;

    for (int i = 0; i < matches.size(); i++) {
        // queryIdx : 매칭된 img1의 index
        // trainIdx : 매칭된 img2의 index
        // pt : pixel
        points1.push_back(keypoint_1[matches[i].queryIdx].pt);
        points2.push_back(keypoint_2[matches[i].trainIdx].pt);
    }
    // F matrix
    cv::Mat fundamental_matrix;
    fundamental_matrix = cv::findFundamentalMat(points1, points2, CV_FM_8POINT);
    std::cout << "F = \n"
              << fundamental_matrix << std::endl;
    // E matrix
    cv::Point2d principal_point(325.1, 249.7);
    double focal_length = 521;
    cv::Mat essential_matrix;
    essential_matrix = cv::findEssentialMat(points1, points2, focal_length, principal_point);

    cv::recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
    std::cout << "R :" << std::endl
              << R << std::endl;
    std::cout << "t :" << std::endl
              << t << std::endl;
}

// nomal camera plane 좌표
cv::Point2d pixel2cam(const cv::Point2d& p, const cv::Mat& K) {
    return cv::Point2d(
        (p.x - K.at<double>(0.2)) / K.at<double>(0, 0),
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

void triangulation(const std::vector<cv::KeyPoint>& keypoint_1, const std::vector<cv::KeyPoint>& keypoint_2,
                   const std::vector<cv::DMatch>& matches, const cv::Mat& R, const cv::Mat& t, std::vector<cv::Point3d>& points) {
    // 3d 좌표
    cv::Mat T1 = (cv::Mat_<float>(3, 4) << 1, 0, 0, 0,
                  0, 1, 0, 0,
                  0, 0, 1, 0);
    cv::Mat T2 = (cv::Mat_<float>(3, 4) << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
                  R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
                  R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0));

    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    // 2d좌표
    std::vector<cv::Point2f> pts_1, pts_2;
    for (auto& m : matches) {
        //픽셀좌표를 카메라 좌표로 전환
        pts_1.push_back(pixel2cam(keypoint_1[m.queryIdx].pt, K));
        pts_2.push_back(pixel2cam(keypoint_2[m.trainIdx].pt, K));
    }
    cv::Mat pts_4d;
    cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);

    //
    for (int i = 0; i < pts_4d.cols; i++) {
        cv::Mat x = pts_4d.col(i);
        x /= x.at<float>(3, 0);  //정규화
        cv::Point3d p(x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2, 0));
        points.push_back(p);
    }
}

int main(int argc, char const* argv[]) {
    std::string img1_path = "../data/1.png";
    std::string img2_path = "../data/2.png";
    // read
    cv::Mat img_1 = cv::imread(img1_path, cv::IMREAD_COLOR);
    cv::Mat img_2 = cv::imread(img2_path, cv::IMREAD_COLOR);
    // initialized
    std::vector<cv::KeyPoint> keypoint_1, keypoint_2;
    std::vector<cv::DMatch> matches;
    cv::Mat R, t;
    std::vector<cv::Point3d> points;
    findFeatureMatches(img_1, img_2, keypoint_1, keypoint_2, matches);
    std::cout << "matches size = " << matches.size() << std::endl;

    pose_estimation_2d2d(keypoint_1, keypoint_2, matches, R, t);

    triangulation(keypoint_1, keypoint_2, matches, R, t, points);
    //삼각점과 특징점 간의 재투영 관계 확인
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    for (int i = 0; i < matches.size(); i++) {
        cv::Point2d pt1_cam = pixel2cam(keypoint_1[matches[i].queryIdx].pt, K);
        cv::Point2d pt1_cam_3d(points[i].x / points[i].z, points[i].y / points[i].z);

        std::cout << "first camera frame :" << pt1_cam << std::endl;
        std::cout << "projected from 3D :" << pt1_cam_3d << ", d=" << points[i].z << std::endl;

        cv::Point2f pt2_cam = pixel2cam(keypoint_2[matches[i].trainIdx].pt, K);
        cv::Mat pt2_trans = R * (cv::Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) + t;
        pt2_trans /= pt2_trans.at<double>(2, 0);
        std::cout << "second camera frame :" << pt2_cam << std::endl;
        std::cout << "reprojected from second frame: " << pt2_trans.t() << std::endl;
    }

    /**3D -> 2D**/
    std::string img_depth_1 = "../data/1_depth.png";
    cv::Mat d1 = cv::imread(img_depth_1, CV_LOAD_IMAGE_UNCHANGED);

    std::vector<cv::Point3f> pts_3d;
    std::vector<cv::Point2f> pts_2d;
    for (auto& m : matches) {
        ushort d = d1.ptr<unsigned short>(int(keypoint_1[m.queryIdx].pt.y))[int(keypoint_1[m.queryIdx].pt.x)];
        if (d == 0) continue;
        float dd = d / 5000.0;
        cv::Point2d p1 = pixel2cam(keypoint_1[m.queryIdx].pt, K);
        pts_3d.push_back(cv::Point3f(p1.x * dd, p1.y * dd, dd));
        pts_2d.push_back(keypoint_2[m.trainIdx].pt);
    }
    cv::Mat r;
    // OpenCV의 PnP 솔루션을 호출하고 EPNP, DLS 및 기타 방법을 선택하십시오.
    cv::solvePnP(pts_3d, pts_2d, K, cv::Mat(), r, t, false);
    // r은 Rodrigues 공식을 사용하여 행렬로 변환된 회전 벡터의 형태입니다.
    cv::Rodrigues(r, R);
    std::cout << "R=" << std::endl
              << R << std::endl;
    std::cout << "t=" << std::endl
              << t << std::endl;
    std::cout << "====BA====" << std::endl;
    return 0;
}
