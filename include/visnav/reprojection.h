/**
BSD 3-Clause License

Copyright (c) 2018, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <memory>

#include <Eigen/Dense>
#include <sophus/se3.hpp>

#include <visnav/common_types.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <ceres/cubic_interpolation.h>

using namespace std;
using namespace cv;
using namespace Eigen;

namespace visnav {

template <class T>
class AbstractCamera;

struct ReprojectionCostFunctor {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ReprojectionCostFunctor(const Eigen::Vector2d& p_2d,
                          const Eigen::Vector3d& p_3d,
                          const std::string& cam_model)
      : p_2d(p_2d), p_3d(p_3d), cam_model(cam_model) {}

  template <class T>
  bool operator()(T const* const sT_w_i, T const* const sT_i_c,
                  T const* const sIntr, T* sResiduals) const {
    // Eigen::Map 的作用是将一个已有的C数组映射为一个Sophus的矩阵，类似于引用
    Eigen::Map<Sophus::SE3<T> const> const T_w_i(sT_w_i);
    Eigen::Map<Sophus::SE3<T> const> const T_i_c(sT_i_c);
    Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(sResiduals);
    const std::shared_ptr<AbstractCamera<T>> cam =
        AbstractCamera<T>::from_data(cam_model, sIntr);

    // TODO SHEET 2: implement the rest of the functor

    residuals = p_2d - cam->project((T_w_i * T_i_c).inverse() * p_3d);
    return true;
  }

  Eigen::Vector2d p_2d;
  Eigen::Vector3d p_3d;
  std::string cam_model;
};

struct BundleAdjustmentReprojectionCostFunctor {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  BundleAdjustmentReprojectionCostFunctor(const Eigen::Vector2d& p_2d,
                                          const std::string& cam_model)
      : p_2d(p_2d), cam_model(cam_model) {}

  template <class T>
  bool operator()(T const* const sT_w_c, T const* const sp_3d_w,
                  T const* const sIntr, T* sResiduals) const {
    // map inputs
    Eigen::Map<Sophus::SE3<T> const> const T_w_c(sT_w_c);
    Eigen::Map<Eigen::Matrix<T, 3, 1> const> const p_3d_w(sp_3d_w);
    Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(sResiduals);
    const std::shared_ptr<AbstractCamera<T>> cam =
        AbstractCamera<T>::from_data(cam_model, sIntr);

    // TODO SHEET 4: Compute reprojection error
    residuals = p_2d - cam->project(T_w_c.inverse() * p_3d_w);

    return true;
  }

  Eigen::Vector2d p_2d;
  std::string cam_model;
};

struct PhotoConsistencyCostFunctor {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  PhotoConsistencyCostFunctor(const Mat& gray_current, const intrinsic& intr,
                              const int& u, const int& v,
                              const double array[640 * 480],
                              const Matrix<double, 3, 1> p_c1, const double var)
      : gray_current(gray_current),
        intr(intr),
        u(u),
        v(v),
        p_c1(p_c1),
        var(var) {
    gray_grid.reset(new ceres::Grid2D<double>(array, 0, gray_current.rows, 0,
                                              gray_current.cols));
    interpolator.reset(
        new ceres::BiCubicInterpolator<ceres::Grid2D<double>>(*gray_grid));
  }

  template <class T>
  bool operator()(T const* const sT_c1_w, T const* const sT_c2_w,
                  T* sResiduals) const {
    Eigen::Map<Sophus::SE3<T> const> const T_c1_w(sT_c1_w);
    Eigen::Map<Sophus::SE3<T> const> const T_c2_w(sT_c2_w);
    Eigen::Map<Eigen::Matrix<T, 1, 1>> residuals(sResiduals);

    // compute photo-consistency error
    Matrix<T, 3, 1> p_c2 = T_c2_w * T_c1_w.inverse() * p_c1;
    T row_next, col_next, pixel_gray_val_next;
    row_next = intr.fx * (p_c2[0] / p_c2[2]) - intr.cx;
    col_next = intr.fy * (p_c2(1) / p_c2(2)) - intr.cy;

    // Get the gray value of the transformed coordinates after interpolation in
    // the next frame
    interpolator->Evaluate(row_next, col_next, &pixel_gray_val_next);
    T r = pixel_gray_val_next - T(gray_current.at<float>(u, v));
    residuals(0) = r * r * (T)6 / ((T)5 + r * r / (T)var);
    // residuals(0) = pixel_gray_val_next - T(gray_current.at<float>(u, v));
    return true;
  }

  Mat gray_current;
  intrinsic intr;
  int u;
  int v;
  double array[640 * 480];
  Matrix<double, 3, 1> p_c1;
  double var;
  std::unique_ptr<ceres::Grid2D<double>> gray_grid;
  std::unique_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double>>>
      interpolator;
};

struct newPhotoConsistencyCostFunctor {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  /*
  PhotoConsistencyCostFunctor(const Mat &gray_current, const intrinsic &intr,
                              const int &u, const int &v,
                              const double array[640 * 480],
                              const Matrix<double, 3, 1> p_c1, const double var)
                              */
  newPhotoConsistencyCostFunctor(const Mat& gray_current, const intrinsic& intr,
                                 const int& u, const int& v,
                                 const double array[640 * 480],
                                 const Matrix<double, 3, 1> p_c1)
      // : gray_current(gray_current), intr(intr), u(u), v(v), p_c1(p_c1),
      // var(var)
      : gray_current(gray_current), intr(intr), u(u), v(v), p_c1(p_c1) {
    gray_grid.reset(new ceres::Grid2D<double>(array, 0, gray_current.rows, 0,
                                              gray_current.cols));
    interpolator.reset(
        new ceres::BiCubicInterpolator<ceres::Grid2D<double>>(*gray_grid));
  }

  template <class T>
  bool operator()(T const* const sT_c1_w, T const* const sT_c2_w,
                  T* sResiduals) const {
    Eigen::Map<Sophus::SE3<T> const> const T_c1_w(sT_c1_w);
    Eigen::Map<Sophus::SE3<T> const> const T_c2_w(sT_c2_w);
    Eigen::Map<Eigen::Matrix<T, 1, 1>> residuals(sResiduals);

    // compute photo-consistency error
    Matrix<T, 3, 1> p_c2 = T_c2_w * T_c1_w.inverse() * p_c1;
    T row_next, col_next, pixel_gray_val_next;
    row_next = intr.fx * (p_c2[0] / p_c2[2]) + intr.cx;
    col_next = intr.fy * (p_c2[1] / p_c2[2]) + intr.cy;

    // Get the gray value of the transformed coordinates after interpolation in
    // the next frame
    interpolator->Evaluate(row_next, col_next, &pixel_gray_val_next);
    T r = pixel_gray_val_next - T(gray_current.at<float>(u, v));
    residuals(0) = r;

    return true;
  }

  Mat gray_current;
  intrinsic intr;
  int u;
  int v;
  double array[640 * 480];
  Matrix<double, 3, 1> p_c1;
  double var;
  std::unique_ptr<ceres::Grid2D<double>> gray_grid;
  std::unique_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double>>>
      interpolator;
};

}  // namespace visnav
