#include "kalman_filter.h"
#include <iostream>
#include <vector>
#include <math.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
    x_ = F_ * x_;
    MatrixXd Ft = F_.transpose();
    P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
    VectorXd err = z - H_ * x_;
    Calc(err);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
    VectorXd new_state = VectorXd(3);
    //state parameters
    float px = x_(0);
    float py = x_(1);
    float vx = x_(2);
    float vy = x_(3);
    float r = sqrt(pow(px,2) + pow(py,2));
    float phi = atan2(py,px);
    float r_dot = (px*vx + py*vy) / r;

    new_state << r, phi, r_dot;
    
    VectorXd err = z - new_state;
    
    //normalize angle
    err[1] = atan2(sin(err[1]),cos(err[1]));
    
    
    Calc(err);

}

//shared update to be used by class
void KalmanFilter::Calc(const VectorXd &z) {
    //from lecture notes
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Si = S.inverse();
    MatrixXd PHt = P_ * Ht;
    MatrixXd K = PHt * Si;
    
    //new estimate
    x_ = x_ + (K * z);
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
}
