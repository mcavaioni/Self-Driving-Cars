#include <fstream>
#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "Eigen-3.3/Eigen/Dense"
#include "json.hpp"
#include "spline.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;


// for convenience
using json = nlohmann::json;

tk::spline x_spline;
tk::spline y_spline;
tk::spline dx_spline;
tk::spline dy_spline;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_first_of("}");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

double distance(double x1, double y1, double x2, double y2)
{
  return sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
}
int ClosestWaypoint(double x, double y, vector<double> maps_x, vector<double> maps_y)
{

  double closestLen = 100000; //large number
  int closestWaypoint = 0;

  for(int i = 0; i < maps_x.size(); i++)
  {
    double map_x = maps_x[i];
    double map_y = maps_y[i];
    double dist = distance(x,y,map_x,map_y);
    if(dist < closestLen)
    {
      closestLen = dist;
      closestWaypoint = i;
    }

  }

  return closestWaypoint;

}

int NextWaypoint(double x, double y, double theta, vector<double> maps_x, vector<double> maps_y)
{

  int closestWaypoint = ClosestWaypoint(x,y,maps_x,maps_y);

  double map_x = maps_x[closestWaypoint];
  double map_y = maps_y[closestWaypoint];

  double heading = atan2( (map_y-y),(map_x-x) );

  double angle = abs(theta-heading);

  if(angle > pi()/4)
  {
    closestWaypoint++;
  }

  return closestWaypoint;

}

// Transform from Cartesian x,y coordinates to Frenet s,d coordinates
vector<double> getFrenet(double x, double y, double theta, vector<double> maps_x, vector<double> maps_y)
{
  int next_wp = NextWaypoint(x,y, theta, maps_x,maps_y);

  int prev_wp;
  prev_wp = next_wp-1;
  if(next_wp == 0)
  {
    prev_wp  = maps_x.size()-1;
  }

  double n_x = maps_x[next_wp]-maps_x[prev_wp];
  double n_y = maps_y[next_wp]-maps_y[prev_wp];
  double x_x = x - maps_x[prev_wp];
  double x_y = y - maps_y[prev_wp];

  // find the projection of x onto n
  double proj_norm = (x_x*n_x+x_y*n_y)/(n_x*n_x+n_y*n_y);
  double proj_x = proj_norm*n_x;
  double proj_y = proj_norm*n_y;

  double frenet_d = distance(x_x,x_y,proj_x,proj_y);

  //see if d value is positive or negative by comparing it to a center point

  double center_x = 1000-maps_x[prev_wp];
  double center_y = 2000-maps_y[prev_wp];
  double centerToPos = distance(center_x,center_y,x_x,x_y);
  double centerToRef = distance(center_x,center_y,proj_x,proj_y);

  if(centerToPos <= centerToRef)
  {
    frenet_d *= -1;
  }

  // calculate s value
  double frenet_s = 0;
  for(int i = 0; i < prev_wp; i++)
  {
    frenet_s += distance(maps_x[i],maps_y[i],maps_x[i+1],maps_y[i+1]);
  }

  frenet_s += distance(0,0,proj_x,proj_y);

  return {frenet_s,frenet_d};

}

// Transform from Frenet s,d coordinates to Cartesian x,y
vector<double> getXY(double s, double d, vector<double> maps_s, vector<double> maps_x, vector<double> maps_y)
{
  int prev_wp = -1;

  while(s > maps_s[prev_wp+1] && (prev_wp < (int)(maps_s.size()-1) ))
  {
    prev_wp++;
  }

  int wp2 = (prev_wp+1)%maps_x.size();

  double heading = atan2((maps_y[wp2]-maps_y[prev_wp]),(maps_x[wp2]-maps_x[prev_wp]));
  // the x,y,s along the segment
  double seg_s = (s-maps_s[prev_wp]);

  double seg_x = maps_x[prev_wp]+seg_s*cos(heading);
  double seg_y = maps_y[prev_wp]+seg_s*sin(heading);

  double perp_heading = heading-pi()/2;

  double x = seg_x + d*cos(perp_heading);
  double y = seg_y + d*sin(perp_heading);

  return {x,y};

}


///-------VEHICLE:--------///
class Vehicle{
  public:
  int set_id(int i){
    return this->id = i;
  }

  string convert_d_to_lane(double d){
    if (d > 0.0 && d < 4.0) {
      return "LEFT";
    } else if (d > 4.0 && d < 8.0) {
      return "MIDDLE";
    } else if (d > 8.0 && d < 12.0) {
      return "RIGHT";
    }
    return "OUT OF BOUNDARY";
  }

  void update_status(double s, double d){
    this->s = s;
    this->d = d;
    // this->v = v;
    this->lane = this->convert_d_to_lane(this->d);
  }

  void update_velocity(double v){
    this->v = v;
  }


  void saved_state(double state_s, double state_sdot, double state_d, double state_ddot){
    this->state_s = state_s;
    this->state_sdot = state_sdot;
    this->state_d = state_d;
    this->state_ddot = state_ddot;

  }


  double convert_lane_to_d(string lane){

    double d = -1.0;

    if (lane == "LEFT") {
      d = 2.0;
    } else if (lane == "MIDDLE") {
      d = 6.0;
    } else if (lane == "RIGHT") {
      d = 9.8;
    }
    return d;
  }


  void adjacent_lanes(){

    if (this->lane == "LEFT") {

      this->lane_at_left = "OUT OF BOUNDARY";
      this->lane_at_right = "MIDDLE";

    } else if (this->lane == "MIDDLE") {

      this->lane_at_left = "LEFT";
      this->lane_at_right = "RIGHT";

    } else if (this->lane == "RIGHT") {

      this->lane_at_left = "MIDDLE";
      this->lane_at_right = "OUT OF BOUNDARY";

    } else {

      this->lane = "OUT OF BOUNDARY";
      this->lane_at_left = "OUT OF BOUNDARY";
      this->lane_at_right = "OUT OF BOUNDARY";
    }
  }


  public:
    int id;
    double s;
    double d;
    double v;
    string lane;
    string lane_at_left;
    string lane_at_right;

    double state_s;
    double state_d;
    double state_sdot;
    double state_ddot;

    double front_gap;
    double front_v;
    double front_s;

};
///------END Vehicle--------///
//////////////////////////////


///------BEHAVIOR:-----------///
double FRONT_THRESHOLD = 25.0;
double BACK_THRESHOLD = 10.0;

double gap(Vehicle &myCar, vector<Vehicle>& otherCars, string lane, string from_where){
  if(lane == "OUT OF BOUNDARY"){
    return 0.0001;
  }
  double adjustment;
  if(from_where == "FRONT"){
    adjustment = 1.0;
  }
  else{
    adjustment = -1.0;
  }
  

  double between_gap = 999999;

  for(auto &otherCar: otherCars){

    double gap = (otherCar.s - myCar.s) * adjustment;

    if (otherCar.lane == lane && gap > 0.0001 && gap < between_gap) {
      between_gap = gap;
      myCar.front_v = otherCar.v;

    }
  }
  return between_gap;
}



double cost(double front_gap, double back_gap, string lane){
  double BACK_GAP_FACTOR = 0.4; 
  double FRONT_GAP_FACTOR = 1.0;
  double cost = FRONT_GAP_FACTOR/front_gap + BACK_GAP_FACTOR/back_gap;
  if(lane == "OUT OF BOUNDARY"){
    return 999999;
  }
  if(front_gap < FRONT_THRESHOLD || back_gap < BACK_THRESHOLD){
    return 999999;
  }

  return cost;
}


double in_lane_cost(double gap){
  if(gap < FRONT_THRESHOLD){
    return 999999;
  }
  return 1.0 / gap;
}


string behavior(Vehicle& myCar, vector<Vehicle>& otherCars){

  myCar.front_gap = gap(myCar, otherCars, myCar.lane, "FRONT");

  double front_left = gap(myCar, otherCars, myCar.lane_at_left, "FRONT");
  double back_left = gap(myCar, otherCars, myCar.lane_at_left, "BACK");
  

  double front_right = gap(myCar, otherCars, myCar.lane_at_right, "FRONT");
  double back_right = gap(myCar, otherCars, myCar.lane_at_right, "BACK");

  double stay_cost = in_lane_cost(myCar.front_gap);
  double left_cost = cost(front_left, back_left, myCar.lane_at_left);
  double right_cost = cost(front_right, back_right, myCar.lane_at_right);


  if(left_cost < stay_cost && left_cost < right_cost){
    return "TURN_LEFT";
  }
  if (right_cost < stay_cost && right_cost < left_cost){
    return "TURN_RIGHT";
  }
  // else{
    return "KEEP_LANE";
  // }
}
///-------END Behavior--------///
//////////////////////////////




///---------TRAJECTORY---------///
  //----JMT (Jerk Minimizing Trajectory)---///
double TRANSFER_TIME = 2.0;

Eigen::VectorXd JMT(vector< double> start, vector <double> end, double T){

    int s = start[0];
    int s_dot = start[1];
    int s_double_dot = start[2];
    
    int s_f = end[0];
    int s_f_dot = end[1];
    int s_f_double_dot = end[2];
    
    Eigen::MatrixXd matrix_t(3,3);
    matrix_t << pow(T,3), pow(T,4), pow(T,5),
                3*(pow(T,2)), 4*(pow(T,3)), 5*(pow(T,4)),
                6*T, 12*pow(T,2), 20*pow(T,3);
                                
    Eigen::VectorXd out_eq(3);
    out_eq << (s_f - (s + s_dot*T + 0.5*s_double_dot*T*T)),
              (s_f_dot - (s_dot + s_double_dot*T)),
              (s_f_double_dot - s_double_dot);
                            
    Eigen::VectorXd alpha(3);
    alpha << ((matrix_t).inverse() * out_eq);
    double a_0 = s;
    double a_1 = s_dot;
    double a_2 = s_double_dot;
    double a_3 = alpha[0];
    double a_4 = alpha[1];
    double a_5 = alpha[2];

    Eigen::VectorXd coeff(6);
    coeff<<a_0, a_1, a_2, a_3, a_4, a_5;
    return coeff;

}

double polynomial_result(Eigen::VectorXd coeff, double t){
  Eigen::VectorXd T(6);
  T << 1.0, t, t*t, pow(t,3), pow(t,4), pow(t,5);
  return T.transpose()*coeff;
}



vector<vector<double>> trajectory_s(Vehicle& myCar, string behavior){
 
  double target_s = myCar.state_s + TRANSFER_TIME * myCar.state_sdot; 
  double target_speed = myCar.state_sdot;

  if(behavior == "KEEP_LANE"){

    if((myCar.front_v > 20.75) || (myCar.front_gap > (FRONT_THRESHOLD+10))){
      target_speed = 20;
    }
    else if (myCar.front_v < myCar.state_sdot){
      target_speed = (myCar.front_v - 6);
    }

    target_speed = target_speed > 15 ? target_speed : 15;

    target_s =  myCar.state_s + TRANSFER_TIME * 0.5 * (myCar.state_sdot + target_speed);

  } 

  vector<double> targetState_s = {target_s, target_speed, 0.0};
  vector<double> startState_s = {myCar.state_s, myCar.state_sdot, 0.0};

  vector<vector<double>> start_target_s = {startState_s, targetState_s};

  return start_target_s;
}



vector<vector<double>> trajectory_d(Vehicle& myCar, string behavior){

  double target_d;
  if(behavior == "TURN_RIGHT"){
    target_d =  myCar.convert_lane_to_d(myCar.lane_at_right);
  }
  else if(behavior == "TURN_LEFT"){
    target_d = myCar.convert_lane_to_d(myCar.lane_at_left);
  }
  else{     
    target_d = myCar.convert_lane_to_d(myCar.lane);
  }

  vector<double> targetState_d = {target_d, 0.0, 0.0}; 
  vector<double> startState_d = {myCar.state_d, myCar.state_ddot, 0.0}; 

  vector<vector<double>> start_target_d = {startState_d, targetState_d};
  return start_target_d;
}
///-------END Trajectory--------///
//////////////////////////////




///---------PATH---------///
vector<double> convert_sd_to_XY(double s, double d){
  double max_s = 6945.554;
  double module_s = fmod(s, max_s);
  double x_num = x_spline(module_s);
  double y_num = y_spline(module_s);
  double dx = dx_spline(module_s);
  double dy = dy_spline(module_s);

  double x = x_num + dx * d;
  double y = y_num + dy * d;

  return {x, y};
}


struct Points {
  std::vector<double> x_path;
  std::vector<double> y_path;
  int n;
};


Points path(Eigen::VectorXd jerk_s, Eigen::VectorXd jerk_d, double t, int n){

  vector<double> x_path;
  vector<double> y_path;
  vector<double> p;

  for (int i = 0; i < n; i++) {

    double s = polynomial_result(jerk_s, (i * t));
    double d = polynomial_result(jerk_d, (i * t));

    vector<double> p = convert_sd_to_XY(s, d);

    x_path.push_back(p[0]);
    y_path.push_back(p[1]);
  }

  Points path = {x_path, y_path, n};

  return path;
}
///-------END Path--------///
//////////////////////////////




bool beginning =true;

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
  double max_s =  6945.554; 

  bool first = true;
  double x1, y1, dx1, dy1;

  double x;
  double y;
  float s;
  float d_x;
  float d_y;


  ifstream in_map_(map_file_.c_str(), ifstream::in);
  
  string line;
  while (getline(in_map_, line)) {
    istringstream iss(line);

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

    if (first) {
      x1 = x;
      y1 = y;
      dx1 = d_x;
      dy1 = d_y;

      first = false;   
    }
  }

   if (in_map_.is_open()) {
    in_map_.close();
  }


  map_waypoints_x.push_back(x1);
  map_waypoints_y.push_back(y1);
  map_waypoints_s.push_back(max_s);
  map_waypoints_dx.push_back(dx1);
  map_waypoints_dy.push_back(dy1);


  //fit splines (cubic polynomial curves):
  x_spline.set_points(map_waypoints_s, map_waypoints_x);
  y_spline.set_points(map_waypoints_s, map_waypoints_y);
  dx_spline.set_points(map_waypoints_s, map_waypoints_dx);
  dy_spline.set_points(map_waypoints_s, map_waypoints_dy);


  h.onMessage([&map_waypoints_x,&map_waypoints_y,&map_waypoints_s,&map_waypoints_dx,&map_waypoints_dy](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    //auto sdata = string(data).substr(0, length);
    //cout << sdata << endl;
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

            // Sensor Fusion Data, a list of all other cars on the same side of the road.
            auto sensor_fusion = j[1]["sensor_fusion"];

            // json msgJson;

            vector<double> next_x_vals;
            vector<double> next_y_vals;


            // TODO: define a path made up of (x,y) points that the car will visit sequentially every .02 seconds
            ///--------------SOLUTION--------------///


            //Initialize my Car with status and position
            Vehicle myCar;
            myCar.set_id(999);
            myCar.update_status(car_s, car_d);
            myCar.update_velocity(car_speed);
            myCar.adjacent_lanes();


            int n = previous_path_x.size();
            Points XY_points = {previous_path_x, previous_path_y, n};

            //Start moving. No cars around.
            if(beginning){

              const double TIME_INCREMENT =0.02;
              const int n_new = 225;
              const double t = n_new * TIME_INCREMENT;
              const double target_speed = 20.0;
              const double target_s = myCar.s + 40.0;

              vector<double>  startState_s = {myCar.s, myCar.v, 0.0};
              vector<double>  startState_d = {myCar.d, 0.0, 0.0};

              vector<double>  endState_s = {target_s, target_speed, 0.0};
              vector<double>  endState_d = {myCar.convert_lane_to_d("MIDDLE"), 0.0, 0.0};

              Eigen::VectorXd jmt_s = JMT(startState_s, endState_s, t);
              Eigen::VectorXd jmt_d = JMT(startState_d, endState_d, t);

              myCar.saved_state(endState_s[0], endState_s[1], endState_d[0], endState_d[1]);

              XY_points = path(jmt_s, jmt_d, TIME_INCREMENT, n_new);
              beginning = false;

            }
            /////////////


            else if(n < 10){
  
            //create vector of other cars.
            vector<Vehicle> otherCars;

            for (int i = 0; i < sensor_fusion.size(); i++) {

              int id = sensor_fusion[i][0];
              double s = sensor_fusion[i][5];
              double d = sensor_fusion[i][6];
              double vx = sensor_fusion[i][3];
              double vy = sensor_fusion[i][4];

              Vehicle car;
              car.set_id(id);

              double velocity = sqrt(vx * vx + vy * vy);
              car.update_status(s, d);
              car.update_velocity(velocity);

              otherCars.push_back(car);
            }

            //evaluate the behavior of my car and the other cars
            string my_behavior = behavior(myCar, otherCars);

            // //transform from Eigen to vector:
            // vector<double> new_trajectory_s(my_trajectory_s.data(), my_trajectory_s.data() + my_trajectory_s.size());
            // vector<double> new_trajectory_d(my_trajectory_d.data(), my_trajectory_d.data() + my_trajectory_d.size());

            //Apply trajectory based on feedback on the behavior of my car and other cars 
            vector<vector<double>>  start_target_s = trajectory_s(myCar, my_behavior);
            vector<vector<double>>  start_target_d = trajectory_d(myCar, my_behavior);

            myCar.saved_state(start_target_s[1][0], start_target_s[1][1], start_target_d[1][0], start_target_d[1][1]);

            //Apply Jerk minimizing to the trajectory
            Eigen::VectorXd jerk_s = JMT(start_target_s[0], start_target_s[1], TRANSFER_TIME);
            Eigen::VectorXd jerk_d = JMT(start_target_d[0], start_target_d[1], TRANSFER_TIME);



            
            double TIME_INCREMENT = 0.02;
            double TRANSFER_TIME = 2.0;
            int TOT_POINTS = int(TRANSFER_TIME / TIME_INCREMENT);

            //Create Path:
            Points Next_XY_points = path(jerk_s, jerk_d, TIME_INCREMENT, TOT_POINTS);

            Next_XY_points.n = TOT_POINTS;

            //Append points to be passed to the simulator
            XY_points.x_path.insert(
              XY_points.x_path.end(), Next_XY_points.x_path.begin(), Next_XY_points.x_path.end());
            
            XY_points.y_path.insert(
              XY_points.y_path.end(), Next_XY_points.y_path.begin(), Next_XY_points.y_path.end());


            XY_points.n = XY_points.x_path.size();

            }
            ///************///////////////**************

            

            json msgJson;


            msgJson["next_x"] = XY_points.x_path;
            msgJson["next_y"] = XY_points.y_path;

            auto msg = "42[\"control\","+ msgJson.dump()+"]";

            //this_thread::sleep_for(chrono::milliseconds(1000));
            ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
          

        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
