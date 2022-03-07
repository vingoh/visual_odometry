#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <sstream>
#include <thread>

#include <sophus/se3.hpp>

#include <tbb/concurrent_unordered_map.h>

#include <pangolin/display/image_view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/image/image.h>
#include <pangolin/image/image_io.h>
#include <pangolin/image/typed_image.h>
#include <pangolin/pangolin.h>

#include <CLI/CLI.hpp>

#include <visnav/calibration.h>
#include <visnav/common_types.h>
#include <visnav/keypoints.h>
#include <visnav/map_utils.h>
#include <visnav/matching_utils.h>
#include <visnav/vo_utils.h>

#include <visnav/gui_helper.h>
#include <visnav/tracks.h>

#include <visnav/serialization.h>

#include <boost/filesystem.hpp>
#include <dirent.h>
#include <fstream>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <stdio.h>
#include <sys/stat.h>
using namespace visnav;
using namespace std;
using namespace cv;
using namespace Eigen;

///////////////////////////////////////////////////////////////////////////////
/// Declarations
///////////////////////////////////////////////////////////////////////////////
void load_data(const std::string& path);
void draw_image_overlay(pangolin::View& v, size_t view_id);
void change_display_to_image(const FrameCamId& fcid);
void convert_to_gray(const string& dataset_path);
double compute_residual(const Sophus::SE3<double> T_c2_c1);
bool next_step();
void optimize();
void optimize_single(Sophus::SE3<double>& T_c2_c1);
void draw_scene();
// void store_timestamp();
void record_pose(Sophus::SE3<double> current_pose);
void initialize_world_cord_and_transformation();
double compute_var(Mat img1, Mat img2, Mat depth);
void window_optimize(Sophus::SE3<double>& T_w_c1, Sophus::SE3<double>& T_w_c2,
                     Sophus::SE3<double>& T_w_c3, Sophus::SE3<double>& T_w_c4);
void sized_window_optimize(int offset = 0);

///////////////////////////////////////////////////////////////////////////////
/// Constants
///////////////////////////////////////////////////////////////////////////////

constexpr int UI_WIDTH = 200;
constexpr int NUM_CAMS = 1;
const intrinsic intr = {525.0, 525.0, 319.5, 239.5, 5000};

///////////////////////////////////////////////////////////////////////////////
/// Variables
///////////////////////////////////////////////////////////////////////////////

/// map with structure (frame id, (timestamp, depth, rgb))
std::map<int, vector<string>> timestamp_depth_rgb;

/// current frame id
size_t current_frame = 300;

/// number of frames in the optimization window
size_t window_size = 4;

/// transforamtion matrix frome current frame to next frame
Sophus::SE3<double> T_c2_c1(MatrixXd::Identity(3, 3), Vector3d(0, 0, 0));

/// initial pose
// Sophus::SE3<double> T_c1_w;

/// camera pose of the frame after current frame
Sophus::SE3<double> current_pose;

///
Matrix<double, 4, 1> world_cord;  //?

/// 3D points of the current frame
// std::vector<Matrix<double, 3, 1>> current_point;

/// bool flag recording wether last optimization converge
bool convergence = true;

/// the frame till which all optimizations are converged
Mat key_frame;

/// pose of the key frame
Sophus::SE3<double> key_pose(Matrix3d::Identity(), Vector3d(0, 0, 0));

/// id of key frame
size_t key_id = current_frame;

/// map storing the relative poses of each sector of trajectory
// map<size_t, map<size_t, Sophus::SE3<double>>> pose_map;

/// map storing the keyposes of each sector of trajectory
// map<size_t, Sophus::SE3<double>> keypose_map;

/// vector stores all relative poses regarding to keyframe
vector<Sophus::SE3<double>> pose(600);

/// all absolute poses
vector<Sophus::SE3<double>> abs_pose(600);

/// flag recording the last optimized frame, avoiding a frame is optimized
/// multiple times, eventhough the trans is large
size_t last_optimized_frame;

size_t last_failed_frame = 0;

std::string dataset_path = "../data/rgbd_dataset_freiburg1_desk";
string timestamp_txt = dataset_path + "/timestamp.txt";
string pose_txt = dataset_path + "/pose_" + ".txt";
ofstream pose_file;
vector<string> time_storage;
bool first_pose = true;

///////////////////////////////////////////////////////////////////////////////
/// GUI parameters
///////////////////////////////////////////////////////////////////////////////

pangolin::Var<bool> ui_show_hidden("ui.show_extra_options", false, true);

///////////////////////////////////////////////////////////////////////////////
/// GUI buttons
///////////////////////////////////////////////////////////////////////////////

// if you enable this, next_step is called repeatedly until completion
pangolin::Var<bool> continue_next("ui.continue_next", false, true);

using Button = pangolin::Var<std::function<void(void)>>;

Button next_step_btn("ui.next_step", &next_step);

//////////////////////////////////////////////
/// Image display options

pangolin::Var<int> show_frame1("ui.show_frame1", 0, 0, 1500);
pangolin::Var<int> show_cam1("ui.show_cam1", 0, 0, NUM_CAMS - 1);
pangolin::Var<int> show_frame2("ui.show_frame2", 0, 0, 1500);
pangolin::Var<int> show_cam2("ui.show_cam2", 1, 0, NUM_CAMS - 1);
pangolin::Var<bool> lock_frames("ui.lock_frames", true, true);

pangolin::Var<bool> show_cameras3d("hidden.show_cameras", true, true);
pangolin::Var<bool> show_points3d("hidden.show_points", true, true);
pangolin::Var<bool> show_old_points3d("hidden.show_old_points3d", true, true);
///////////////////////////////////////////////////////////////////////////////
/// Main
///////////////////////////////////////////////////////////////////////////////
bool dirExists(const std::string& path) {
  struct stat info;
  if (stat(path.c_str(), &info) == 0 && info.st_mode & S_IFDIR) {
    return true;
  }
  return false;
}

int main(int argc, char** argv) {
  bool show_gui = true;

  time_t t = time(0);
  char tmp[32] = {};
  strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S", localtime(&t));
  string pose_txt = dataset_path + "/pose_" + tmp + ".txt";
  pose_file.open(pose_txt);
  initialize_world_cord_and_transformation();

  CLI::App app{"RGB-D slam"};

  app.add_option("--show-gui", show_gui, "Show GUI");
  app.add_option("--dataset-path", dataset_path,
                 "Dataset path. Default: " + dataset_path);

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  // load gray images
  load_data(dataset_path);
  // store the timestamp
  // store_timestamp();

  if (show_gui) {
    cout << "in gui " << endl;
    pangolin::CreateWindowAndBind("Main", 1800, 1000);

    glEnable(GL_DEPTH_TEST);

    // main parent display for images and 3d viewer
    pangolin::View& main_view =
        pangolin::Display("main")
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0)
            .SetLayout(pangolin::LayoutEqualVertical);

    // parent display for images
    pangolin::View& img_view_display =
        pangolin::Display("images").SetLayout(pangolin::LayoutEqual);
    main_view.AddDisplay(img_view_display);

    // main ui panel
    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0,
                                          pangolin::Attach::Pix(UI_WIDTH));

    // extra options panel
    pangolin::View& hidden_panel = pangolin::CreatePanel("hidden").SetBounds(
        0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH),
        pangolin::Attach::Pix(2 * UI_WIDTH));
    ui_show_hidden.Meta().gui_changed = true;

    // 2D image views
    std::vector<std::shared_ptr<pangolin::ImageView>> img_view;
    // here 2 for the image and depth
    while (img_view.size() < 2) {
      std::shared_ptr<pangolin::ImageView> iv(new pangolin::ImageView);

      size_t idx = img_view.size();
      img_view.push_back(iv);

      img_view_display.AddDisplay(*iv);
      iv->extern_draw_function =
          std::bind(&draw_image_overlay, std::placeholders::_1, idx);
    }

    // 3D visualization (initial camera view optimized to see full map)
    pangolin::OpenGlRenderState camera(
        pangolin::ProjectionMatrix(640, 480, 400, 400, 320, 240, 0.001, 10000),
        pangolin::ModelViewLookAt(-3.4, -3.7, -8.3, 2.1, 0.6, 0.2,
                                  pangolin::AxisNegY));

    pangolin::View& display3D =
        pangolin::Display("scene")
            .SetAspect(-640 / 480.0)
            .SetHandler(new pangolin::Handler3D(camera));
    main_view.AddDisplay(display3D);
    //

    while (!pangolin::ShouldQuit()) {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      if (ui_show_hidden.GuiChanged()) {
        hidden_panel.Show(ui_show_hidden);
        const int panel_width = ui_show_hidden ? 2 * UI_WIDTH : UI_WIDTH;
        main_view.SetBounds(0.0, 1.0, pangolin::Attach::Pix(panel_width), 1.0);
      }

      // display3D.Activate(camera);
      // glClearColor(0.95f, 0.95f, 0.95f, 1.0f);  // light gray background

      // draw_scene();

      img_view_display.Activate();
      if (lock_frames) {
        // in case of locking frames, chaning one should change the other
        if (show_frame1.GuiChanged()) {
          change_display_to_image(FrameCamId(show_frame1, 0));
          change_display_to_image(FrameCamId(show_frame1, 1));
        } else if (show_frame2.GuiChanged()) {
          change_display_to_image(FrameCamId(show_frame2, 0));
          change_display_to_image(FrameCamId(show_frame2, 1));
        }
      }
      if (show_frame1.GuiChanged() || show_cam1.GuiChanged()) {
        auto frame_id = static_cast<FrameId>(show_frame1);
        if (timestamp_depth_rgb.find(frame_id) != timestamp_depth_rgb.end()) {
          pangolin::TypedImage img =
              pangolin::LoadImage(timestamp_depth_rgb[frame_id][2]);

          img_view[0]->SetImage(img);
        } else {
          img_view[0]->Clear();
        }
      }
      if (show_frame2.GuiChanged() || show_cam2.GuiChanged()) {
        auto frame_id = static_cast<FrameId>(show_frame2);
        if (timestamp_depth_rgb.find(frame_id) != timestamp_depth_rgb.end()) {
          pangolin::GlPixFormat fmt;
          fmt.glformat = GL_LUMINANCE;
          fmt.gltype = GL_UNSIGNED_BYTE;
          fmt.scalable_internal_format = GL_LUMINANCE8;
          pangolin::TypedImage img =
              pangolin::LoadImage(timestamp_depth_rgb[frame_id][1]);

          img_view[1]->SetImage(img);
        } else {
          img_view[1]->Clear();
        }
      }
      pangolin::FinishFrame();

      if (continue_next) {
        // stop if there is nothing left to do
        continue_next = next_step();
      } else {
        // if the gui is just idling, make sure we don't burn too much CPU
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
    }
  }
}

bool next_step() {
  // stop when the current frame is the second last frame
  if (current_frame >= timestamp_depth_rgb.size() - 1) {
    cout << "next step ended!" << endl;
    return false;
  }

  cout << "current frame: " << current_frame << endl;
  cout << "key id: " << key_id << endl;
  sized_window_optimize();
  // optimize();

  // reinitialize every x frames
  /*
  if (current_frame % 999 == 0 && current_frame != 0) {
    cout << "reinitializing" << endl;
    key_pose = pose[current_frame];
    Sophus::SE3<double> identity(Matrix3d::Identity(), Vector3d(0, 0, 0));
    for (size_t i = 0; i < window_size; i++) {
      pose[current_frame + i] = identity;
    }
    first_pose = true;

    sized_window_optimize();

    // cout << "relative pose:" << pose[current_frame].matrix3x4() << endl;
    // cout << pose[current_frame + 1].matrix3x4() << endl;
    // cout << "absolute pose;" << (pose[current_frame] *
  key_pose).matrix3x4()
    //      << endl;
    // cout << (pose[current_frame + 1] * key_pose).matrix3x4() << endl;
    // cout << "" << endl;
  }*/

  double max_trans = 0.04, max_rot = 0.04;

  Sophus::SE3<double> pose_current = pose[current_frame] * key_pose;
  Sophus::SE3<double> pose_last;
  if (!first_pose)
    pose_last = abs_pose[current_frame - 1];
  else
    pose_last = pose_current;

  Quaterniond q_current, q_last;
  q_current = pose_current.so3().matrix();
  q_last = pose_last.so3().matrix();
  double wx = abs(q_current.x() - q_last.x());
  double wy = abs(q_current.y() - q_last.y());
  double wz = abs(q_current.z() - q_last.z());

  double dx = abs((pose_current.matrix() - pose_last.matrix())(0, 3));
  double dy = abs((pose_current.matrix() - pose_last.matrix())(1, 3));
  double dz = abs((pose_current.matrix() - pose_last.matrix())(2, 3));

  if (wx > max_rot || wy > max_rot || wz > max_rot || dx > max_trans ||
      dy > max_trans || dz > max_trans) {
    cout << "last failed frame:" << last_failed_frame << endl;
    if (current_frame == last_failed_frame) {  // if frame still not qualified
                                               // after reinitialization
      cout << "give up optimization for frame " << current_frame << endl;
      pose[current_frame] = abs_pose[current_frame - 1] *
                            abs_pose[current_frame - 2].inverse() *
                            pose[current_frame - 1];

      abs_pose[current_frame] = pose[current_frame] * key_pose;
      record_pose(abs_pose[current_frame]);
      current_frame++;  // cf + 1
    } else {
      last_failed_frame = current_frame;
    }
    cout << "translation or rotation too large, reinitializing" << endl;
    cout << pose_current.translation().transpose() << endl;

    key_pose = abs_pose[current_frame - 1];  // cf or cf-1
    key_id = current_frame - 1;              // cf or cf-1
    cout << "new key pose taken at frame " << current_frame - 1 << endl;

    // re-initialize
    current_frame--;  // cf or cf-1
    Sophus::SE3<double> identity(Matrix3d::Identity(), Vector3d(0, 0, 0));
    for (size_t i = 0; i < window_size; i++) {
      pose[current_frame + i] = identity;
    }
    first_pose = true;  // make sure the first reinitialized pose is fixed
    sized_window_optimize();
    // optimize();
  } else {
    abs_pose[current_frame] = pose[current_frame] * key_pose;
    record_pose(abs_pose[current_frame]);  // T_c_w
    cout << "pose of " << current_frame << " recorded" << endl;
  }

  // update next incoming frame:
  Sophus::SE3<double> T_last_w = pose[current_frame + window_size - 1];
  Sophus::SE3<double> T_second_last_w = pose[current_frame + window_size - 2];
  pose[current_frame + window_size] =
      T_last_w * T_second_last_w.inverse() * T_last_w;
  // pose[current_frame + window_size] = T_last_w;

  /*
  optimize(T_c2_c1);
  cout << T_c2_c1.matrix() << endl;
  //current_pose = T_c2_c1 * current_pose;
  current_pose = T_c2_c1 * key_pose;
  record_pose(current_pose);
  */

  // record_pose(pose[current_frame]);

  current_frame++;
  first_pose = false;
  change_display_to_image(FrameCamId(current_frame, 0));
  cout << "" << endl;
  return true;
}

void sized_window_optimize(int offset) {
  if (window_size < 2) {
    cerr << "window size smaller than 2" << endl;
    abort();
  }

  ceres::Problem problem;
  std::vector<ceres::ResidualBlockId> residual_block_ids;

  Mat gray_current =
      imread(timestamp_depth_rgb.at(current_frame + offset)[2], 0);
  normalize(gray_current, gray_current, 1, 0, NORM_MINMAX, CV_32F);
  // gray_current.convertTo(gray_current, CV_32F);
  Sophus::SE3<double>& T_c1_w = pose[current_frame + offset];

  problem.AddParameterBlock(T_c1_w.data(), 7,
                            new Sophus::test::LocalParameterizationSE3);
  if (current_frame + offset == 0 || first_pose == true) {
    cout << "current frame is fixed" << endl;
    problem.SetParameterBlockConstant(T_c1_w.data());
  }
  Mat gray_window, depth;
  int height = gray_current.size().height;
  int width = gray_current.size().width;
  double gray_window_array[height * width];
  for (int j = 0; j < 1; j++) {
    for (size_t frame = 1; frame < window_size; frame++) {
      gray_window =
          imread(timestamp_depth_rgb.at(current_frame + offset + frame)[2], 0);
      depth = imread(
          timestamp_depth_rgb.at(current_frame + offset + frame - 1)[1], 2);
      normalize(gray_window, gray_window, 1, 0, NORM_MINMAX, CV_32F);
      // gray_window.convertTo(gray_window, CV_32F);
      Sophus::SE3<double>& T_c2_w = pose[current_frame + offset + frame];

      problem.AddParameterBlock(T_c2_w.data(), 7,
                                new Sophus::test::LocalParameterizationSE3);

      for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
          gray_window_array[i * width + j] = gray_window.at<float>(i, j);
        }
      }

      for (int u = 0; u < height; u++) {
        for (int v = 0; v < width; v++) {
          Matrix<double, 3, 1> p_c1;
          if (depth.at<uint16_t>(u, v) != 0) {
            p_c1(2) = (depth.at<uint16_t>(u, v) / intr.factor);
            p_c1(0) = (u - intr.cx) * p_c1(2) / intr.fx;
            p_c1(1) = (v - intr.cy) * p_c1(2) / intr.fy;

            // double var1_2 = compute_var(gray1, gray2, depth1, T_w_c1,
            // T_w_c2);
            Matrix<double, 3, 1> p_c1_2 = T_c2_w * T_c1_w.inverse() * p_c1;
            double u_2 = intr.fx * (p_c1_2[0] / p_c1_2[2]) + intr.cx;
            double v_2 = intr.fy * (p_c1_2[1] / p_c1_2[2]) + intr.cy;
            if (u_2 >= 0 && u_2 < height && v_2 >= 0 && v_2 < width) {
              // double residual =gray_window.at<float>(u_2, v_2) -
              // gray_current.at<float>(u, v); cout << residual << endl;

              ceres::CostFunction* cost_function =
                  new ceres::AutoDiffCostFunction<
                      newPhotoConsistencyCostFunctor, 1, 7, 7>(
                      new newPhotoConsistencyCostFunctor(
                          gray_current, intr, u, v, gray_window_array, p_c1));
              ceres::LossFunction* loss;
              loss = new ceres::HuberLoss(0.5);
              // loss = NULL;
              ceres::ResidualBlockId r_id = problem.AddResidualBlock(
                  cost_function, loss, T_c1_w.data(), T_c2_w.data());
              residual_block_ids.push_back(r_id);
            }
          }
        }
      }
    }
  }
  ceres::Problem::EvaluateOptions options;
  options.residual_blocks = residual_block_ids;
  double total_cost = 0.0;
  vector<double> evaluated_residuals;
  // vector<ceres::ResidualBlockId> removeId_vec;
  problem.Evaluate(options, &total_cost, &evaluated_residuals, nullptr,
                   nullptr);

  ceres::Problem::Options pro_options;
  pro_options.enable_fast_removal = true;

  for (size_t i = 0; i < evaluated_residuals.size(); i++) {
    if (evaluated_residuals[i] > 0.2) {
      // problem.RemoveResidualBlock(residual_block_ids[i]);
    }
  }

  cout << "solving" << endl;
  ceres::Solver::Options ceres_options;
  ceres_options.max_num_iterations = 50;
  ceres_options.max_solver_time_in_seconds = 60;
  ceres_options.minimizer_type = ceres::TRUST_REGION;
  ceres_options.linear_solver_type = ceres::SPARSE_SCHUR;
  ceres_options.num_threads = std::thread::hardware_concurrency();
  // ceres_options.update_state_every_iteration = true;
  ceres::Solver::Summary summary;
  ceres::Solve(ceres_options, &problem, &summary);
  cout << summary.BriefReport() << endl;
  // cout << summary.FullReport() << endl;
  cout << summary.total_time_in_seconds << endl;
}

double compute_var(Mat img1, Mat img2, Mat depth) {
  int height = img1.rows;
  int width = img1.cols;
  double residual_sum = 0;
  double var_old = 999;
  double var_new = 1;
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
  while (abs(var_new - var_old) > 0.001 * var_old) {
    var_old = var_new;
    for (int i = 0; i < (int)residuals.size(); i++) {
      double r = residuals[i];  // square
      residual_sum += r * 6 / (5 + r / var_old);
    }
    var_new = residual_sum / residuals.size();
    // cout << "old:" << var_old << " new:" << var_new << endl;
  }

  return var_new;
}

void optimize() {
  Mat gray_current = imread(timestamp_depth_rgb.at(current_frame)[2],
                            0);  // read rgb as gray
  Mat gray_next = imread(timestamp_depth_rgb.at(current_frame + 1)[2], 0);
  Mat depth = imread(timestamp_depth_rgb.at(current_frame)[1], 2);

  normalize(gray_current, gray_current, 1, 0, NORM_MINMAX, CV_32F);
  normalize(gray_next, gray_next, 1, 0, NORM_MINMAX, CV_32F);
  // gray_current.convertTo(gray_current, CV_32F);
  // gray_next.convertTo(gray_next, CV_32F);

  int height = depth.size().height;
  int width = depth.size().width;

  Sophus::SE3<double>& T_c1_w = pose[current_frame];
  Sophus::SE3<double>& T_c2_w = pose[current_frame + 1];

  if (1) {
    key_pose = current_pose;
    key_id = current_frame;
  }
  // cout << "current frame id:" << current_frame << endl;
  //  cout << "compute between " << key_id << " and " << current_frame + 1
  //  << endl;

  double var = compute_var(gray_current, gray_next, depth);
  cout << "var:" << var << endl;

  ceres::Problem problem;
  problem.AddParameterBlock(T_c1_w.data(), 7,
                            new Sophus::test::LocalParameterizationSE3);
  problem.AddParameterBlock(T_c2_w.data(), 7,
                            new Sophus::test::LocalParameterizationSE3);

  // convert Mat to 1D array, maybe easier way?
  double gray_next_array[height * width];
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      gray_next_array[i * width + j] = gray_next.at<float>(i, j);
    }
  }

  for (int u = 0; u < height; u++) {
    for (int v = 0; v < width; v++) {
      Matrix<double, 3, 1> p_c1;
      if (depth.at<uint16_t>(u, v) != 0) {
        p_c1(2) = (depth.at<uint16_t>(u, v) / intr.factor);
        p_c1(0) = (u + intr.cx) * p_c1(2) / intr.fx;
        p_c1(1) = (v + intr.cy) * p_c1(2) / intr.fy;
        Matrix<double, 3, 1> p_c2 = T_c2_w * T_c1_w.inverse() * p_c1;
        double u_next = intr.fx * (p_c2[0] / p_c2[2]) - intr.cx;
        double v_next = intr.fy * (p_c2(1) / p_c2(2)) - intr.cy;

        if (u_next >= 0 && u_next < height && v_next >= 0 && v_next < width) {
          ceres::CostFunction* cost_function =
              new ceres::AutoDiffCostFunction<PhotoConsistencyCostFunctor, 1, 7,
                                              7>(
                  new PhotoConsistencyCostFunctor(gray_current, intr, u, v,
                                                  gray_next_array, p_c1, var));
          ceres::LossFunction* loss;
          // loss = new ceres::HuberLoss(0.5);
          loss = NULL;
          problem.AddResidualBlock(cost_function, loss, T_c1_w.data(),
                                   T_c2_w.data());
        }
      }
    }
  }
  // solve
  ceres::Solver::Options ceres_options;
  ceres_options.max_num_iterations = 50;
  ceres_options.minimizer_type = ceres::TRUST_REGION;
  ceres_options.linear_solver_type = ceres::SPARSE_SCHUR;
  ceres_options.num_threads = std::thread::hardware_concurrency();
  // ceres_options.update_state_every_iteration = true;
  ceres::Solver::Summary summary;
  ceres::Solve(ceres_options, &problem, &summary);
  cout << summary.BriefReport() << endl;
  cout << summary.total_time_in_seconds << endl;
}

void window_optimize(Sophus::SE3<double>& T_c1_w, Sophus::SE3<double>& T_c2_w,
                     Sophus::SE3<double>& T_c3_w, Sophus::SE3<double>& T_c4_w) {
  Mat gray1 = imread(timestamp_depth_rgb.at(current_frame)[2],
                     0);  // read rgb as gray
  Mat gray2 = imread(timestamp_depth_rgb.at(current_frame + 1)[2], 0);
  Mat gray3 = imread(timestamp_depth_rgb.at(current_frame + 2)[2], 0);
  Mat gray4 = imread(timestamp_depth_rgb.at(current_frame + 3)[2], 0);

  Mat depth1 = imread(timestamp_depth_rgb.at(current_frame)[1], 2);
  Mat depth2 = imread(timestamp_depth_rgb.at(current_frame + 1)[1], 2);
  Mat depth3 = imread(timestamp_depth_rgb.at(current_frame + 2)[1], 2);

  // gray_current.convertTo(gray_current, CV_8UC1, 1.0 / 255, 0);  //
  // normalize gray_next.convertTo(gray_next, CV_8UC1, 1.0 / 255, 0);
  normalize(gray1, gray1, 1, 0, NORM_MINMAX, CV_32F);
  normalize(gray2, gray2, 1, 0, NORM_MINMAX, CV_32F);
  normalize(gray3, gray3, 1, 0, NORM_MINMAX, CV_32F);
  normalize(gray4, gray4, 1, 0, NORM_MINMAX, CV_32F);

  ceres::Problem problem;

  int height = depth1.size().height;
  int width = depth1.size().width;

  problem.AddParameterBlock(T_c1_w.data(), 7,
                            new Sophus::test::LocalParameterizationSE3);
  problem.AddParameterBlock(T_c2_w.data(), 7,
                            new Sophus::test::LocalParameterizationSE3);
  problem.AddParameterBlock(T_c3_w.data(), 7,
                            new Sophus::test::LocalParameterizationSE3);
  problem.AddParameterBlock(T_c4_w.data(), 7,
                            new Sophus::test::LocalParameterizationSE3);

  if (current_frame == 0) {
    cout << "current frame is 0" << endl;
    problem.SetParameterBlockConstant(T_c1_w.data());
  }

  // convert Mat to 1D array, maybe easier way?
  double gray_2_array[height * width];
  double gray_3_array[height * width];
  double gray_4_array[height * width];
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      gray_2_array[i * width + j] = gray2.at<float>(i, j);
      gray_3_array[i * width + j] = gray3.at<float>(i, j);
      gray_4_array[i * width + j] = gray4.at<float>(i, j);
    }
  }

  // std::vector<ceres::ResidualBlockId> residual_block_ids;
  // current_point.clear();
  for (int u = 0; u < height; u++) {
    for (int v = 0; v < width; v++) {
      // Frame 1
      Matrix<double, 3, 1> p_c1;
      if (depth1.at<uint16_t>(u, v) != 0) {
        p_c1(2) = (depth1.at<uint16_t>(u, v) / intr.factor);
        p_c1(0) = (u + intr.cx) * p_c1(2) / intr.fx;
        p_c1(1) = (v + intr.cy) * p_c1(2) / intr.fy;

        // Frame 1 and Frame 2
        // double var1_2 = compute_var(gray1, gray2, depth1, T_w_c1,
        // T_w_c2);
        Matrix<double, 3, 1> p_c1_2 = T_c2_w * T_c1_w.inverse() * p_c1;
        double u_2 = intr.fx * (p_c1_2[0] / p_c1_2[2]) - intr.cx;
        double v_2 = intr.fy * (p_c1_2[1] / p_c1_2[2]) - intr.cy;
        if (u_2 >= 0 && u_2 < height && v_2 >= 0 && v_2 < width) {
          // current_point.push_back(p_c2);
          ceres::CostFunction* cost_function =
              new ceres::AutoDiffCostFunction<newPhotoConsistencyCostFunctor, 1,
                                              7, 7>(
                  // new PhotoConsistencyCostFunctor(gray1, intr, u, v,
                  // gray_2_array, p_c1, var1_2));
                  new newPhotoConsistencyCostFunctor(gray1, intr, u, v,
                                                     gray_2_array, p_c1));
          ceres::LossFunction* loss;
          loss = new ceres::HuberLoss(0.5);
          problem.AddResidualBlock(cost_function, loss, T_c1_w.data(),
                                   T_c2_w.data());
        }

        // Frame 1 and Frame 3
        // double var1_3 = compute_var(gray1, gray3, depth1, T_w_c1,
        // T_w_c3);
        Matrix<double, 3, 1> p_c1_3 = T_c3_w * T_c1_w.inverse() * p_c1;
        double u_3 = intr.fx * (p_c1_3[0] / p_c1_3[2]) - intr.cx;
        double v_3 = intr.fy * (p_c1_3[1] / p_c1_3[2]) - intr.cy;
        if (u_3 >= 0 && u_3 < height && v_3 >= 0 && v_3 < width) {
          // current_point.push_back(p_c3);
          ceres::CostFunction* cost_function =
              new ceres::AutoDiffCostFunction<newPhotoConsistencyCostFunctor, 1,
                                              7, 7>(
                  // new PhotoConsistencyCostFunctor(gray1, intr, u, v,
                  // gray_3_array, p_c1, var1_3));
                  new newPhotoConsistencyCostFunctor(gray1, intr, u, v,
                                                     gray_3_array, p_c1));
          ceres::LossFunction* loss;
          loss = new ceres::HuberLoss(0.5);
          problem.AddResidualBlock(cost_function, loss, T_c1_w.data(),
                                   T_c3_w.data());
        }
        // Frame 1 and Frame 4
        // double var1_4 = compute_var(gray1, gray4, depth1, T_w_c1,
        // T_w_c4);
        Matrix<double, 3, 1> p_c1_4 = T_c4_w * T_c1_w.inverse() * p_c1;
        double u_4 = intr.fx * (p_c1_4[0] / p_c1_4[2]) - intr.cx;
        double v_4 = intr.fy * (p_c1_4[1] / p_c1_4[2]) - intr.cy;
        if (u_4 >= 0 && u_4 < height && v_4 >= 0 && v_4 < width) {
          // current_point.push_back(p_c4);
          ceres::CostFunction* cost_function =
              new ceres::AutoDiffCostFunction<newPhotoConsistencyCostFunctor, 1,
                                              7, 7>(
                  // new PhotoConsistencyCostFunctor(gray1, intr, u, v,
                  // gray_4_array, p_c1, var1_4));
                  new newPhotoConsistencyCostFunctor(gray1, intr, u, v,
                                                     gray_4_array, p_c1));
          ceres::LossFunction* loss;
          loss = new ceres::HuberLoss(0.5);
          problem.AddResidualBlock(cost_function, loss, T_c1_w.data(),
                                   T_c4_w.data());
        }
      }

      // Frame2

      Matrix<double, 3, 1> p_c2;
      if (depth2.at<uint16_t>(u, v) != 0) {
        p_c2(2) = (depth2.at<uint16_t>(u, v) / intr.factor);
        p_c2(0) = (u + intr.cx) * p_c2(2) / intr.fx;
        p_c2(1) = (v + intr.cy) * p_c2(2) / intr.fy;
        // Frame 2 and Frame 3
        // double var2_3 = compute_var(gray2, gray3, depth2, T_w_c2,T_w_c3);
        Matrix<double, 3, 1> p_c2_3 = T_c3_w * T_c2_w.inverse() * p_c2;
        double u_3 = intr.fx * (p_c2_3[0] / p_c2_3[2]) - intr.cx;
        double v_3 = intr.fy * (p_c2_3[1] / p_c2_3[2]) - intr.cy;
        if (u_3 >= 0 && u_3 < height && v_3 >= 0 && v_3 < width) {
          // current_point.push_back(p_c2);
          ceres::CostFunction* cost_function =
              new ceres::AutoDiffCostFunction<newPhotoConsistencyCostFunctor, 1,
                                              7, 7>(
                  // new PhotoConsistencyCostFunctor(gray2, intr, u, v,
                  // gray_3_array, p_c2, var2_3));
                  new newPhotoConsistencyCostFunctor(gray2, intr, u, v,
                                                     gray_3_array, p_c2));
          ceres::LossFunction* loss;
          loss = new ceres::HuberLoss(0.5);
          problem.AddResidualBlock(cost_function, loss, T_c2_w.data(),
                                   T_c3_w.data());
        }

        // Frame 2 and Frame 4
        // double var2_4 = compute_var(gray2, gray4, depth2, T_w_c2,T_w_c4);
        Matrix<double, 3, 1> p_c2_4 = T_c4_w * T_c2_w.inverse() * p_c2;
        double u_4 = intr.fx * (p_c2_4[0] / p_c2_4[2]) - intr.cx;
        double v_4 = intr.fy * (p_c2_4[1] / p_c2_4[2]) - intr.cy;
        if (u_4 >= 0 && u_4 < height && v_4 >= 0 && v_4 < width) {
          // current_point.push_back(p_c2);
          ceres::CostFunction* cost_function =
              new ceres::AutoDiffCostFunction<newPhotoConsistencyCostFunctor, 1,
                                              7, 7>(
                  // new PhotoConsistencyCostFunctor(gray2, intr, u, v,
                  // gray_4_array, p_c2, var2_4));
                  new newPhotoConsistencyCostFunctor(gray2, intr, u, v,
                                                     gray_4_array, p_c2));
          ceres::LossFunction* loss;
          loss = new ceres::HuberLoss(0.5);
          problem.AddResidualBlock(cost_function, loss, T_c2_w.data(),
                                   T_c4_w.data());
        }
      }

      // Frame3
      Matrix<double, 3, 1> p_c3;
      if (depth3.at<uint16_t>(u, v) != 0) {
        p_c3(2) = (depth3.at<uint16_t>(u, v) / intr.factor);
        p_c3(0) = (u + intr.cx) * p_c3(2) / intr.fx;
        p_c3(1) = (v + intr.cy) * p_c3(2) / intr.fy;
        // Frame 3 and Frame 4
        // double var3_4 = compute_var(gray3, gray4, depth3, T_w_c3,T_w_c4);
        Matrix<double, 3, 1> p_c3_4 = T_c4_w * T_c3_w.inverse() * p_c3;
        double u_4 = intr.fx * (p_c3_4[0] / p_c3_4[2]) - intr.cx;
        double v_4 = intr.fy * (p_c3_4[1] / p_c3_4[2]) - intr.cy;
        if (u_4 >= 0 && u_4 < height && v_4 >= 0 && v_4 < width) {
          // current_point.push_back(p_c2);
          ceres::CostFunction* cost_function =
              new ceres::AutoDiffCostFunction<newPhotoConsistencyCostFunctor, 1,
                                              7, 7>(
                  // new PhotoConsistencyCostFunctor(gray3, intr, u,
                  // v,gray_4_array, p_c3, var3_4));
                  new newPhotoConsistencyCostFunctor(gray3, intr, u, v,
                                                     gray_4_array, p_c3));
          ceres::LossFunction* loss;
          loss = new ceres::HuberLoss(0.5);
          problem.AddResidualBlock(cost_function, loss, T_c3_w.data(),
                                   T_c4_w.data());
        }
      }

      // end
    }
  }
  cout << "after adding resiual block" << endl;

  // solve
  cout << "sloving" << endl;
  ceres::Solver::Options ceres_options;
  ceres_options.max_num_iterations = 50;
  ceres_options.minimizer_type = ceres::TRUST_REGION;
  ceres_options.linear_solver_type = ceres::SPARSE_SCHUR;
  ceres_options.num_threads = std::thread::hardware_concurrency();
  ceres_options.max_solver_time_in_seconds = 60;
  // ceres_options.update_state_every_iteration = true;
  ceres::Solver::Summary summary;
  ceres::Solve(ceres_options, &problem, &summary);
  cout << summary.BriefReport() << endl;
  // cout << summary.FullReport() << endl;
  cout << summary.total_time_in_seconds << endl;
}

void load_data(const std::string& dataset_path) {
  const string match_txt = dataset_path + "/match.txt";
  ifstream matches(match_txt);
  FrameId id = 0;
  while (matches) {
    string line;
    getline(matches, line);
    if (line.size() < 20) continue;
    string timestamp = line.substr(0, line.find("depth") - 1);
    string depth_name = line.substr(line.find("depth"),
                                    line.find('g') - line.find("depth") + 1);
    string rgb_name = line.substr(line.find("rgb"), line.size());
    vector<string> match = {timestamp, dataset_path + '/' + depth_name,
                            dataset_path + '/' + rgb_name};
    timestamp_depth_rgb[id] = match;
    id++;
  }
  std::cerr << "Loaded " << id << " rgb-depth pairs" << std::endl;
  if (id == 0) abort();

  show_frame1.Meta().range[1] = timestamp_depth_rgb.size() / NUM_CAMS - 1;
  show_frame1.Meta().gui_changed = true;
  show_frame2.Meta().range[1] = timestamp_depth_rgb.size() / NUM_CAMS - 1;
  show_frame2.Meta().gui_changed = true;
}

// Update the image views to a given image id
void change_display_to_image(const FrameCamId& fcid) {
  if (fcid.cam_id == 0) {
    // left view
    show_cam1 = 0;
    show_frame1 = fcid.frame_id;
    show_cam1.Meta().gui_changed = true;
    show_frame1.Meta().gui_changed = true;
  } else {
    // right view
    show_cam2 = fcid.cam_id;
    show_frame2 = fcid.frame_id;
    show_cam2.Meta().gui_changed = true;
    show_frame2.Meta().gui_changed = true;
  }
}

void draw_image_overlay(pangolin::View& v, size_t view_id) {}

// Render the 3D viewer scene of cameras and points
/*
void draw_scene() {
  const FrameCamId fcid1(show_frame1, show_cam1);
  const FrameCamId fcid2(show_frame2, show_cam2);

  const u_int8_t color_camera_current[3]{255, 0, 0};         // red
  const u_int8_t color_camera_left[3]{0, 125, 0};            // dark green
  const u_int8_t color_camera_right[3]{0, 0, 125};           // dark blue
  const u_int8_t color_points[3]{0, 0, 0};                   // black
  const u_int8_t color_old_points[3]{170, 170, 170};         // gray
  const u_int8_t color_selected_left[3]{0, 250, 0};          // green
  const u_int8_t color_selected_right[3]{0, 0, 250};         // blue
  const u_int8_t color_selected_both[3]{0, 250, 250};        // teal
  const u_int8_t color_outlier_observation[3]{250, 0, 250};  // purple

  // cout << "in draw scene" << endl;
  //  render cameras
  if (show_cameras3d) {
    render_camera(current_pose.inverse().matrix(), 2.0f, color_camera_current,
                  0.1f);
  }

  // render points
  if (show_points3d && current_point.size() > 0) {
    glPointSize(3.0);
    glBegin(GL_POINTS);
    for (int i = 0; i < (int)current_point.size(); i++) {
      glColor3ubv(color_points);
      pangolin::glVertex(current_pose.inverse() * current_point[i]);
    }
    glEnd();
  }
}*/

void record_pose(Sophus::SE3<double> current_pose) {
  double tx = current_pose.matrix3x4()(0, 3);
  double ty = current_pose.matrix3x4()(1, 3);
  double tz = current_pose.matrix3x4()(2, 3);

  Quaterniond q;
  Matrix3d Rotation_M = current_pose.so3().matrix();
  q = Rotation_M;
  // cout << "q: " << q.x() << q.y() << q.z() << q.w() << endl;

  pose_file << timestamp_depth_rgb.at(current_frame)[0] << " " << tx << " "
            << ty << " " << tz << " " << q.x() << " " << q.y() << " " << q.z()
            << " " << q.w() << endl;
  cout << timestamp_depth_rgb.at(current_frame)[0] << " x:" << tx << " y:" << ty
       << endl;
}

void initialize_world_cord_and_transformation() {
  world_cord << 0, 0, 0, 1;
  // Matrix3d R;
  // R << 0.0798578, 0.6121341, -0.7867113, 0.9967406, -0.0399786,
  // 0.0700705, 0.0114409, -0.7897427, -0.6133315;

  Quaternion<double> q;
  q.x() = 0.886385;
  q.y() = 0.231053;
  q.z() = -0.089359;
  q.w() = -0.391090;

  Matrix3d R = q.normalized().toRotationMatrix();

  // Sophus::SE3<double> T_c1_w(R, Vector3d(1.314809, 0.847662, 1.519455));
  Sophus::SE3<double> T_c1_w(Matrix3d::Identity(), Vector3d(0, 0, 0));
  current_pose = T_c1_w;
  cout << T_c1_w.matrix3x4() << endl;

  // for (size_t i = 0; i < window_size; i++) pose.push_back(T_c1_w);
  // for (size_t i = 0; i < current_frame+window_size; i++)
  // pose.push_back(T_c1_w);
  for (size_t i = 0; i < window_size; i++) {
    pose[current_frame + i] = T_c1_w;
    // pose.at(current_frame + i) = T_c1_w;
  }
  // keypose_map[key_id] = T_c1_w;
}
