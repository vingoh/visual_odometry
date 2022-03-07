#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <sstream>
#include <thread>

#include <dirent.h>
#include "opencv2/opencv.hpp"
//#include <opencv2/core/eigen.hpp>
#include <sophus/se3.hpp>

using namespace std;
using namespace cv;
using namespace Eigen;

using Timestamp_d = double;
double compute_var(Mat img1, Mat img2, Mat depth);

struct intrinsic {
  /// focal length x
  double fx;

  /// focal length y
  double fy;

  /// optical center x
  double cx;

  /// optical center y
  double cy;

  /// factor for the depth information
  int factor;
};

const intrinsic intr = {525.0, 525.0, 319.5, 239.5, 5000};

int main() {
  Mat img1 = imread(
      "/Users/hpj123/project/data/rgbd_dataset_freiburg1_desk/rgb/"
      "1305031453.359684.png",
      0);
  Mat img2 = imread(
      "../../data/rgbd_dataset_freiburg1_desk/rgb/1305031453.391690.png");
  Mat depth = imread(
      "../../data/rgbd_dataset_freiburg1_desk/depth/1305031453.374112.png");
  // imshow("", depth);
  // waitKey(0);
  string dataset_path = "../../data/rgbd_dataset_freiburg1_xyz";
  std::map<int, vector<string>> timestamp_depth_rgb;

  imshow("", img1);
  waitKey(0);

  normalize(img1, img1, 1, 0, NORM_MINMAX, CV_32F);
  normalize(img2, img2, 1, 0, NORM_MINMAX, CV_32F);

  imshow("", img1);
  waitKey(0);

  const string match_txt = dataset_path + "/match.txt";
  ifstream matches(match_txt);
  int id = 0;
  while (matches) {
    string line;
    getline(matches, line);
    if (line.size() < 20) continue;
    string timestamp = line.substr(0, line.find("depth"));
    string depth_name = line.substr(line.find("depth"),
                                    line.find('g') - line.find("depth") + 1);
    string rgb_name = line.substr(line.find("rgb"), line.size());
    // cout << timestamp << "," << depth_name << "," << rgb_name << endl;
    vector<string> match = {timestamp, dataset_path + '/' + depth_name,
                            dataset_path + '/' + rgb_name};
    timestamp_depth_rgb[id] = match;
    id++;
  }
  // compute_var(img1, img2, depth);
}

double compute_var(Mat img1, Mat img2, Mat depth) {
  int height = img1.rows;
  int width = img1.cols;
  double residual_sum = 0;
  double var_old = 999;
  double var_new = 1;
  Sophus::SE3<double> T_c2_c1(MatrixXd::Identity(3, 3), Vector3d(0, 0, 0));
  vector<double> residuals;

  // compute residual of each pixel
  for (int u = 0; u < height; u++) {
    for (int v = 0; v < width; v++) {
      Eigen::Matrix<double, 3, 1> p_c1;
      p_c1(2) = depth.at<uint16_t>(u, v) / intr.factor;
      p_c1(0) = (u + intr.cx) * p_c1(2) / intr.fx;
      p_c1(1) = (v + intr.cy) * p_c1(2) / intr.fy;

      Matrix<double, 3, 1> p_c2 = T_c2_c1 * p_c1;
      int u_next = int(intr.fx * p_c2(0) / p_c2(2) - intr.cx + 0.5);
      int v_next = int(intr.fy * p_c2(1) / p_c2(2) - intr.cy + 0.5);

      if (u_next >= 0 && u_next < height && v_next >= 0 && v_next < width) {
        double residual = img2.at<float>(u_next, v_next) - img1.at<float>(u, v);
        residuals.push_back(residual * residual);
      }
    }
  }
  while (abs(var_new - var_old) > 0.001) {
    var_old = var_new;
    for (int i = 0; i < (int)residuals.size(); i++) {
      double r = residuals[i];
      residual_sum += r * r * 6 / (5 + pow(r, 2) / var_old);
    }
    var_new = residual_sum / residuals.size();
    cout << "old:" << var_old << " new:" << var_new << endl;
  }
  return var_new;
}
