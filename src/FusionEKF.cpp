#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>
#include <stdlib.h>


using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
  FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_  = MatrixXd(2, 2);
  R_radar_  = MatrixXd(3, 3);
  H_laser_  = MatrixXd(2, 4);
  Hj_       = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0,      0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0,      0,
              0,    0.0009, 0,
              0,    0,      0.09;

  // There's no H_radar_ matrix here b/c we're using
  // the extended Kalman filter for the update step which
  // doesn't use an H matrix
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;
  
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {

    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    
    //state covariance matrix P
    ekf_.P_ = MatrixXd(4, 4);
    ekf_.P_ <<  1, 0, 0,    0,
                0, 1, 0,    0,
                0, 0, 1000, 0,
                0, 0, 0,    1000;
    
    
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      //ekf_.x_ << measurement_pack.raw_measurements_
      float ro      = measurement_pack.raw_measurements_[0];
      float phi     = measurement_pack.raw_measurements_[1];
      float ro_dot  = measurement_pack.raw_measurements_[2];
      
      ekf_.x_ << ro * cos(phi), ro * sin(phi), ro_dot * cos(phi), ro_dot * sin(phi);

    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;

    }
    
    previous_timestamp_ = measurement_pack.timestamp_;
    
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0f;	//dt - expressed in seconds
  previous_timestamp_ = measurement_pack.timestamp_;
  
  float dt_2 = dt * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3 * dt;
  
  //Modify the F matrix so that the time is integrated
  MatrixXd F_ = MatrixXd(4, 4);
  F_ << 1,  0, dt,  0,
        0,  1,  0, dt,
        0,  0,  1,  0,
        0,  0,  0,  1;
  
  //set the process covariance matrix Q
  float noise_ax = 9, noise_ay = 9;
  MatrixXd Q_ = MatrixXd(4, 4);
  Q_ <<   dt_4/4*noise_ax,    0,                dt_3/2*noise_ax,  0,
          0,                  dt_4/4*noise_ay,  0,                dt_3/2*noise_ay,
          dt_3/2*noise_ax,    0,                dt_2*noise_ax,    0,
          0,                  dt_3/2*noise_ay,  0,                dt_2*noise_ay;
  
  // I find naming this method 'Init' is not at all intuitive
  // but left as is... :-)
  ekf_.Init(ekf_.x_,
            ekf_.P_,
            F_,
            H_laser_,
            measurement_pack.sensor_type_ == MeasurementPackage::LASER ? R_laser_ : R_radar_,
            Q_);
  
  // Predict does not use H_ matrix so we can get away with initialization above
  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);

  } else {
    // Laser update
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
