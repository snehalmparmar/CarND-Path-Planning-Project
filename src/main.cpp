#include <uWS/uWS.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "helpers.h"
#include "json.hpp"
#include "spline.h"
#include <math.h>
#include <chrono>
#include <thread>
#include <algorithm>

using namespace std;

// for convenience
//using json = nlohmann::json;
using nlohmann::json;
using std::string;
using std::vector;

const double SPEED_LIMIT = 49.0; // mph
const double SAFETY_DIST = 30.0; // meters
const double MAX_ACCELERATION = .224; // m/s2

// Reference velocity to target
double ref_vel = 0.0; //mph

double UpdateSpeed(double speed_to_match = -1) {
  // By default speed up
  double updated_speed = ref_vel + MAX_ACCELERATION;

  // Otherwise, try to catch lane speed
  if (speed_to_match != -1) {
    speed_to_match *= 2.24 * .98; // mph // .98 to prevent overshooting front car speed

    // Slow down to match lane speed
    if (MAX_ACCELERATION <= ref_vel) {
      updated_speed = max(speed_to_match, ref_vel - 1.5*MAX_ACCELERATION);
    }
    // Accelerate to match lane speed
    else {
      updated_speed = min(speed_to_match, ref_vel + MAX_ACCELERATION);
    }
  }

  // Check to not exceed speed limit
  ref_vel = min(SPEED_LIMIT, updated_speed);

  return ref_vel;
}

bool CheckLane(vector<vector<double>> sensor_fusion, double car_s, double car_d) {
  // Check all cars from the desired lane
  for (int i = 0; i < sensor_fusion.size(); ++i) {
    // Retrieve detected car's state
    double vx = sensor_fusion[i][3];
    double vy = sensor_fusion[i][4];
    double s = sensor_fusion[i][5];

    double speed = sqrt(vx*vx + vy*vy);
    s += (double) speed * .02;

    // Check for cars in front of ego car
    if ((s >= car_s) && (s - car_s <= 2.0*SAFETY_DIST)) {
      return false;
    }

    // Check for cars behind of ego car
    double min_dist = 0.25 * SAFETY_DIST;
    double max_dist = 1.0 * SAFETY_DIST;
    double dist = (1 - ref_vel/SPEED_LIMIT) * max_dist + min_dist;
    if ((s <= car_s) && (car_s - s <= dist)) {
      return false;
    }
  }

  // It's safe!
  return true;
}

int BestLane(vector<vector<double>> sensor_fusion_left, vector<vector<double>> sensor_fusion_right, double car_s) {
  double closest_left_car_dist = INFINITY;
  double closest_right_car_dist = INFINITY;

  // Find left lane closest car
  for (int i = 0; i < sensor_fusion_left.size(); ++i) {
    // Retrieve detected car's state
    double vx = sensor_fusion_left[i][3];
    double vy = sensor_fusion_left[i][4];
    double s = sensor_fusion_left[i][5];

    double speed = sqrt(vx*vx + vy*vy);
    s += (double) speed * .02;

    if ((s > car_s) && (s - car_s < closest_left_car_dist)) {
      closest_left_car_dist = s - car_s;
    }
  }

  // Find right lane closest car
  for (int i = 0; i < sensor_fusion_right.size(); ++i) {
    // Retrieve detected car's state
    double vx = sensor_fusion_right[i][3];
    double vy = sensor_fusion_right[i][4];
    double s = sensor_fusion_right[i][5];

    double speed = sqrt(vx*vx + vy*vy);
    s += (double) speed * .02;

    if ((s > car_s) && (s - car_s < closest_right_car_dist)) {
      closest_right_car_dist = s - car_s;
    }
  }

  // Always go left, if cars are far away in front of ego car
  // Or when the left front car is farest than the right front car
  if ((closest_left_car_dist >= 3*SAFETY_DIST && closest_right_car_dist >= 3*SAFETY_DIST) ||
       closest_left_car_dist >= closest_right_car_dist) {
    return 0;
  }
  // Otherwise, go right
  else {
    return 2;
  }
}

int DetectLaneFromCarPos(double car_d) {
  if (car_d < 4) {
    return 0;
  }
  else if (car_d >= 4 && car_d < 8) {
    return 1;
  }
  else {
    return 2;
  }
}

vector<double> SetEndGoal(double s, double d, double speed) {
  vector<double> end_goal;

  end_goal.push_back(s);
  end_goal.push_back(d);
  end_goal.push_back(speed);

  return end_goal;
}

vector<double> TryChangingLane(vector<vector<double>> car_from_left, vector<vector<double>> car_from_right, double car_s, double car_d, double front_car_speed) {
  // Detect current lane
  int lane = DetectLaneFromCarPos(car_d);

  // Check lanes
  bool is_left_lane_safe = CheckLane(car_from_left, car_s, car_d);
  bool is_right_lane_safe = CheckLane(car_from_right, car_s, car_d);

  // Can't go left, try right
  if ((lane == 0 || (lane == 1 && !is_left_lane_safe)) && is_right_lane_safe) {
    return SetEndGoal(car_s+1.5*SAFETY_DIST, 2+4*(lane+1), UpdateSpeed());
  }
  // Try best lane
  else if (lane == 1 && is_left_lane_safe && is_right_lane_safe) {
    int best = BestLane(car_from_left, car_from_right, car_s);
    return SetEndGoal(car_s+1.5*SAFETY_DIST, 2+4*best, UpdateSpeed());
  }
  // Can't go right, try left
  else if ((lane == 2 || (lane == 1 && !is_right_lane_safe)) && is_left_lane_safe) {
    return SetEndGoal(car_s+1.5*SAFETY_DIST, 2+4*(lane-1), UpdateSpeed());
  }
  // Otherwise, slow down
  else {
    return SetEndGoal(car_s+SAFETY_DIST, 2+4*lane, UpdateSpeed(front_car_speed));
  }
}

vector<double> SimpleBehaviorPlanner(vector<vector<double>> sensor_fusion, double car_s, double car_d) {
  bool should_change_lane = false;
  double front_car_speed;
  double front_car_s = INFINITY;
  vector<vector<double>> car_from_left;
  vector<vector<double>> car_from_right;

  int lane = DetectLaneFromCarPos(car_d);

  for (int i = 0; i < sensor_fusion.size(); ++i) {
    // Retrieve state of detected vehicle
    double vx = sensor_fusion[i][3];
    double vy = sensor_fusion[i][4];
    double s = sensor_fusion[i][5];
    double d = sensor_fusion[i][6];

    // Compute s value for the detected and ego car after step
    double speed = sqrt(vx*vx + vy*vy);
    s += (double) speed * 0.02;

    // Check the car lane
    int detected_car_lane = DetectLaneFromCarPos(d); // TODO: detect cars close to us between lanes (changing...)
    bool in_same_lane = lane == detected_car_lane;
    bool from_left = lane - 1 == detected_car_lane;
    bool from_right = lane + 1 == detected_car_lane;

    // Check if it's close from our car
    bool is_close = (s > car_s) && (s - car_s < SAFETY_DIST);

    // Try to change lane if the car is close from us in the same lane
    if (in_same_lane && is_close) {
      should_change_lane = true;
      if (s < front_car_s) {
        front_car_speed = speed;
        front_car_s = s;
      }
    }
    else if (from_left) {
      car_from_left.push_back(sensor_fusion[i]);
    }
    else if (from_right) {
      car_from_right.push_back(sensor_fusion[i]);
    }
  }

  // Try changing lane, if not possible slow down
  if (should_change_lane) {
    return TryChangingLane(car_from_left, car_from_right, car_s, car_d, front_car_speed);
  }
  // Stay on the same lane and keep going forward
  else {
    return SetEndGoal(car_s+SAFETY_DIST, 2+4*lane, UpdateSpeed());
  }
}

vector<vector<double>> GenerateTrajectory(vector<double> desired_goal, double car_x, double car_y, double car_yaw, vector<double> previous_path_x, vector<double> previous_path_y, vector<double> map_waypoints_x, vector<double> map_waypoints_y, vector<double> map_waypoints_s) {
  vector<double> ptsx;
  vector<double> ptsy;

  double ref_x = car_x;
  double ref_y = car_y;
  double ref_yaw = deg2rad(car_yaw);

  int prev_size = previous_path_x.size();

  if (prev_size < 2) {
    // Use 2 points that make the path tangent to the car
    double prev_car_x = car_x - cos(car_yaw);
    double prev_car_y = car_y - sin(car_yaw);

    ptsx.push_back(prev_car_x);
    ptsx.push_back(car_x);

    ptsy.push_back(prev_car_y);
    ptsy.push_back(car_y);
  }
  else {
    ref_x = previous_path_x[prev_size-1];
    ref_y = previous_path_y[prev_size-1];

    double ref_x_prev = previous_path_x[prev_size-2];
    double ref_y_prev = previous_path_y[prev_size-2];
    ref_yaw = atan2(ref_y-ref_y_prev, ref_x-ref_x_prev);

    ptsx.push_back(ref_x_prev);
    ptsx.push_back(ref_x);

    ptsy.push_back(ref_y_prev);
    ptsy.push_back(ref_y);
  }

  // Define next trajectory point as the desired goal
  vector<double> desired_goal_pts = getXY(desired_goal[0], desired_goal[1], map_waypoints_s, map_waypoints_x, map_waypoints_y);
  ptsx.push_back(desired_goal_pts[0]);
  ptsy.push_back(desired_goal_pts[1]);

  // Shift to car coordinate
  for (size_t i = 0; i < ptsx.size(); ++i) {
    double shiftx = ptsx[i] - ref_x;
    double shifty = ptsy[i] - ref_y;
    ptsx[i] = shiftx*cos(0-ref_yaw) - shifty*sin(0-ref_yaw);
    ptsy[i] = shiftx*sin(0-ref_yaw) + shifty*cos(0-ref_yaw);
  }

  // Use spline to calculate trajectory
  tk::spline s;
  s.set_points(ptsx, ptsy);

  // Choose next waypoints from trajectory
  vector<double> next_x_vals;
  vector<double> next_y_vals;

  // Start from all the remaining previous path points from last time
  for (size_t i = 0; i < prev_size; ++i) {
    next_x_vals.push_back(previous_path_x[i]);
    next_y_vals.push_back(previous_path_y[i]);
  }

  double target_x = 30.0;
  double target_y = s(target_x);
  double target_dist = sqrt(target_x*target_x + target_y*target_y);
  int N = target_dist / (0.02 * desired_goal[2]/2.24); // 2.24 used to convert in m/s

  double x_add_on = 0.0;

  for (size_t i = 0; i < 50-prev_size; ++i) {
    double next_x = x_add_on + target_x/N;
    double next_y = s(next_x);

    x_add_on = next_x;

    // Back to map coordinate
    double x_ref = next_x;
    double y_ref = next_y;
    next_x = x_ref*cos(ref_yaw) - y_ref*sin(ref_yaw);
    next_y = x_ref*sin(ref_yaw) + y_ref*cos(ref_yaw);

    next_x += ref_x;
    next_y += ref_y;

    next_x_vals.push_back(next_x);
    next_y_vals.push_back(next_y);
  }

  vector<vector<double>> trajectory;
  trajectory.push_back(next_x_vals);
  trajectory.push_back(next_y_vals);

  return trajectory;
}

int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  ifstream in_map_(map_file_.c_str(), ifstream::in);

  string line;
  while (getline(in_map_, line)) {
    istringstream iss(line);
    double x;
    double y;
    float s;
    float d_x;
    float d_y;
    iss >> x;
    iss >> y;
    iss >> s;
    iss >> d_x;
    iss >> d_y;
    map_waypoints_x.push_back(x);
    map_waypoints_y.push_back(y);
    map_waypoints_s.push_back(s);
    map_waypoints_dx.push_back(d_x);
    map_waypoints_dy.push_back(d_y);
  }

  h.onMessage([&map_waypoints_x,&map_waypoints_y,&map_waypoints_s,&map_waypoints_dx,&map_waypoints_dy](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);

        string event = j[0].get<string>();

        if (event == "telemetry") {
          // j[1] is the data JSON object

          // Main car's localization Data
          double car_x = j[1]["x"];
          double car_y = j[1]["y"];
          double car_s = j[1]["s"];
          double car_d = j[1]["d"];
          double car_yaw = j[1]["yaw"];
          double car_speed = j[1]["speed"];

          // Previous path data given to the Planner
          auto previous_path_x = j[1]["previous_path_x"];
          auto previous_path_y = j[1]["previous_path_y"];
          // Previous path's end s and d values
          double end_path_s = j[1]["end_path_s"];
          double end_path_d = j[1]["end_path_d"];

          // Sensor Fusion Data, a list of all other cars on the same side of the road
          auto sensor_fusion = j[1]["sensor_fusion"];

          json msgJson;

          // 1. Decide what the car should do based on data from sensor fusion and current state
          vector<double> desired_goal = SimpleBehaviorPlanner(sensor_fusion, car_s, car_d);

          // 2. Generate trajectory
          vector<vector<double>> next_vals = GenerateTrajectory(desired_goal, car_x, car_y, car_yaw, previous_path_x, previous_path_y, map_waypoints_x, map_waypoints_y, map_waypoints_s);

          msgJson["next_x"] = next_vals[0];
          msgJson["next_y"] = next_vals[1];

          auto msg = "42[\"control\","+ msgJson.dump()+"]";

          //this_thread::sleep_for(chrono::milliseconds(1000));
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }  // end websocket if
  }); // end h.onMessage

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    cout << "Connected!!!" << endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    cout << "Disconnected" << endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    cout << "Listening to port " << port << endl;
  } else {
    cerr << "Failed to listen to port" << endl;
    return -1;
  }
  h.run();
}