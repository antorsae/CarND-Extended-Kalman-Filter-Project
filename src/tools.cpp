#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0,0,0,0;
  
  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  if(estimations.size() != ground_truth.size()
     || estimations.size() == 0){
    rmse << -1,-1,-1,-1;
    return rmse;
  }
  
  //accumulate squared residuals
  for(unsigned int i=0; i < estimations.size(); ++i){
    
    VectorXd residual = estimations[i] - ground_truth[i];
    
    //coefficient-wise multiplication
    residual = residual.array() * residual.array();
    rmse += residual;
  }
  
  //calculate the mean
  rmse = rmse/estimations.size();
  
  //calculate the squared root
  rmse = rmse.array().sqrt();
  
  //return the result
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
    MatrixXd Hj(3,4);
  
    //recover state parameters
    double px = x_state(0);
    double py = x_state(1);
    double vx = x_state(2);
    double vy = x_state(3);
    
    //pre-compute a set of terms to avoid repeated calculation
    double c1 = px*px+py*py;
  
    //check division by zero to avoid NaNs
    if(fabs(c1) < 0.0001){
      c1 = 0.0001;
    }
  
    double c2 = sqrt(c1);
    double c3 = (c1*c2);
  
    // Assert none of these are NaNs
    assert(!(isnan(c1+c2+c3)));

    //compute the Jacobian matrix
    Hj <<  (px/c2),               (py/c2),                0,      0,
          -(py/c1),               (px/c1),                0,      0,
          py*(vx*py - vy*px)/c3,  px*(px*vy - py*vx)/c3,  px/c2,  py/c2;
    
    return Hj;
}

// from http://stackoverflow.com/questions/11498169/dealing-with-angle-wrap-in-c-code
double Tools::ConstrainAngle(double x){
  x = fmod(x + M_PI, 2 * M_PI);
  if (x < 0)
    x += 2 * M_PI;
  return x - M_PI;
}
