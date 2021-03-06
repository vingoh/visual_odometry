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

#include <bitset>
#include <set>

#include <Eigen/Dense>
#include <sophus/se3.hpp>

#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>

#include <visnav/camera_models.h>
#include <visnav/common_types.h>

using namespace opengv;

namespace visnav {

void computeEssential(const Sophus::SE3d& T_0_1, Eigen::Matrix3d& E) {
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

  // TODO SHEET 3: compute essential matrix
  UNUSED(E);
  UNUSED(t_0_1);
  UNUSED(R_0_1);
  Eigen::Matrix3d t_cap = Sophus::SO3<double>::hat(t_0_1.normalized());
  E = t_cap * R_0_1;
}

void findInliersEssential(const KeypointsData& kd1, const KeypointsData& kd2,
                          const std::shared_ptr<AbstractCamera<double>>& cam1,
                          const std::shared_ptr<AbstractCamera<double>>& cam2,
                          const Eigen::Matrix3d& E,
                          double epipolar_error_threshold, MatchData& md) {
  md.inliers.clear();
  for (size_t j = 0; j < md.matches.size(); j++) {
    const Eigen::Vector2d p0_2d = kd1.corners[md.matches[j].first];
    const Eigen::Vector2d p1_2d = kd2.corners[md.matches[j].second];

    // TODO SHEET 3: determine inliers and store in md.inliers
    UNUSED(cam1);
    UNUSED(cam2);
    UNUSED(E);
    UNUSED(epipolar_error_threshold);
    UNUSED(p0_2d);
    UNUSED(p1_2d);

    double err =
        cam1->unproject(p0_2d).transpose() * E * cam2->unproject(p1_2d);
    if (abs(err) < epipolar_error_threshold)
      md.inliers.push_back(
          make_pair(md.matches[j].first, md.matches[j].second));
  }
}

void findInliersRansac(const KeypointsData& kd1, const KeypointsData& kd2,
                       const std::shared_ptr<AbstractCamera<double>>& cam1,
                       const std::shared_ptr<AbstractCamera<double>>& cam2,
                       const double ransac_thresh, const int ransac_min_inliers,
                       MatchData& md) {
  md.inliers.clear();
  md.T_i_j = Sophus::SE3d();

  // TODO SHEET 3: Run RANSAC with using opengv's CentralRelativePose and store
  // the final inlier indices in md.inliers and the final relative pose in
  // md.T_i_j (normalize translation). If the number of inliers is smaller than
  // ransac_min_inliers, leave md.inliers empty. Note that if the initial RANSAC
  // was successful, you should do non-linear refinement of the model parameters
  // using all inliers, and then re-estimate the inlier set with the refined
  // model parameters.
  UNUSED(kd1);
  UNUSED(kd2);
  UNUSED(cam1);
  UNUSED(cam2);
  UNUSED(ransac_thresh);
  UNUSED(ransac_min_inliers);

  bearingVectors_t bVectors1, bVectors2;
  for (size_t i = 0; i < md.matches.size(); i++) {
    bearingVector_t bvec1 = cam1->unproject(kd1.corners[md.matches[i].first]);
    bearingVector_t bvec2 = cam2->unproject(kd2.corners[md.matches[i].second]);
    bVectors1.push_back(bvec1);
    bVectors2.push_back(bvec2);
  }

  relative_pose::CentralRelativeAdapter adapter(bVectors1, bVectors2);

  // create a RANSAC object
  sac::Ransac<sac_problems::relative_pose::CentralRelativePoseSacProblem>
      ransac;
  // create a CentralRelativePoseSacProblem
  // (set algorithm to STEWENIUS, NISTER, SEVENPT, or EIGHTPT)
  std::shared_ptr<sac_problems::relative_pose::CentralRelativePoseSacProblem>
      relposeproblem_ptr(
          new sac_problems::relative_pose::CentralRelativePoseSacProblem(
              adapter, sac_problems::relative_pose::
                           CentralRelativePoseSacProblem::NISTER));

  // run ransac
  ransac.sac_model_ = relposeproblem_ptr;
  ransac.threshold_ = ransac_thresh;
  ransac.max_iterations_ = 50;
  ransac.computeModel();

  transformation_t new_transformation =
      relative_pose::optimize_nonlinear(adapter, ransac.inliers_);

  // optimize transformation using inliers

  Eigen::Matrix3d R = new_transformation.block(0, 0, 3, 3);
  Eigen::Matrix<double, 3, 1> t = new_transformation.block(0, 3, 3, 1);
  Sophus::SE3d T(R, t.normalized());
  md.T_i_j = T;

  // compute inliers again using new T, trying to replace adapter
  vector<int> inliers_indices;
  ransac.sac_model_->selectWithinDistance(new_transformation, ransac_thresh,
                                          inliers_indices);

  if (int(inliers_indices.size()) > ransac_min_inliers) {
    for (size_t i = 0; i < inliers_indices.size(); i++)
      md.inliers.push_back(md.matches[inliers_indices[i]]);
  }
}
}  // namespace visnav
