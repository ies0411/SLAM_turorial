#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <boost/format.hpp>
#include <chrono>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
int main(int argc, char const* argv[]) {
    // load
    std::string path = "../ubuntu.png";
    cv::Mat image = cv::imread(path);
    cout << "height:" << image.rows << "\nwidth:" << image.cols << endl;
    // cv::imshow("image", image);
    // cv::waitKey(0);
    cout << "channel : " << image.channels() << endl;

    for (int y = 0; y < image.rows; y++) {
        unsigned char* row_ptr = image.ptr<unsigned char>(y);
        for (int x = 0; x < image.cols; x++) {
            unsigned char* data_ptr = &row_ptr[x * image.channels()];
            for (int c = 0; c != image.channels(); c++) {
                unsigned char data = data_ptr[c];
            }
        }
    }
    // clone
    cv::Mat image_another = image;
    image_another(cv::Rect(0, 0, 100, 100)).setTo(0);
    // cv::imshow("image", image);
    // cv::waitKey(0);
    cv::Mat image_clone = image.clone();
    image_clone(cv::Rect(0, 0, 100, 100)).setTo(255);
    // cv::imshow("image", image);
    // cv::imshow("image_clone", image_clone);
    // cv::waitKey(0);

    /*****map****/

    std::ifstream fin("../pose.txt");
    std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses;
    // aligned_allocator, 16byte이상 allocate할때 사용
    vector<cv::Mat> colorImgs, depthImgs;
    for (int i = 0; i < 5; i++) {
        boost::format fmt("../%s/%d.%s");
        colorImgs.push_back(cv::imread((fmt % "color" % (i + 1) % "png").str()));
        depthImgs.push_back(cv::imread((fmt % "depth" % (i + 1) % "pgm").str()));
        std::array<double, 7> data;  //   double data[7] = {0};
        for (auto& d : data) fin >> d;
        Eigen::Quaterniond q(data[6], data[3], data[4], data[5]);
        Eigen::Isometry3d T(q);
        T.pretranslate(Eigen::Vector3d(data[0], data[1], data[2]));
        poses.push_back(T);
    }

    // 포인트 클라우드 카메라 내부 매개변수
    double cx = 325.5;
    double cy = 253.5;
    double fx = 518.0;
    double fy = 519.0;
    double depthScale = 1000.0;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (int i = 0; i < 5; i++) {
        cv::Mat color = colorImgs[i];
        cv::Mat depth = depthImgs[i];
        Eigen::Isometry3d T = poses[i];
        for (int v = 0; v < color.rows; v++) {
            for (int u = 0; u < color.cols; u++) {
                unsigned int d = depth.ptr<unsigned short>(v)[u];
                if (d == 0) continue;
                Eigen::Vector3d point;
                // 2D -> 3D
                point[2] = double(d) / depthScale;
                point[0] = (u - cx) * point[2] / fx;
                point[1] = (v - cy) * point[2] / fy;

                // transform to world frame
                Eigen::Vector3d pointWorld = T * point;

                pcl::PointXYZRGB p;
                p.x = pointWorld[0];
                p.y = pointWorld[1];
                p.z = pointWorld[2];

                p.b = color.data[v * color.step + u * color.channels()];
                p.g = color.data[v * color.step + u * color.channels() + 1];
                p.r = color.data[v * color.step + u * color.channels() + 2];
                pointCloud->points.push_back(p);
            }
        }
    }
    pointCloud->is_dense = false;
    cout << "size : " << pointCloud->size() << endl;
    pcl::io::savePCDFileBinary("./map_rev.pcd", *pointCloud);
    return 0;
}
