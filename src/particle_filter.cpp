/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  default_random_engine gen;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  num_particles = 100;
  for (int i = 0; i < num_particles; ++i){
    Particle particle;

    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1;
    weights.push_back(1);
    particles.push_back(particle);
    is_initialized = true;
  }

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

  default_random_engine gen;
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);
  for (auto &p : particles) {
    if (fabs(yaw_rate) > 0.001) {
      p.x = p.x + velocity / yaw_rate * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta)) + dist_x(gen);
      p.y = p.y + velocity / yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t)) + dist_y(gen);
      p.theta = p.theta + yaw_rate * delta_t + dist_theta(gen);
    } else {
      p.x = p.x + velocity * delta_t * cos(p.theta) + dist_x(gen);
      p.y = p.y + velocity * delta_t * sin(p.theta) + dist_y(gen);
      p.theta = p.theta + dist_theta(gen);
    }
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    for(auto& obs : observations){
      double min_dist = numeric_limits<double>::max();
      for (auto& preLdmk : predicted){
        double distance = dist(preLdmk.x, preLdmk.y, obs.x, obs.y);
        if ( distance < min_dist ){
          min_dist = distance;
          obs.id = preLdmk.id;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
  double std_x = std_landmark[0];
  double std_y = std_landmark[1];
  double norm_factor = 0.0;

  for (int i=0; i < particles.size(); ++i){
      Particle& p = particles[i];
      vector<LandmarkObs> predicted_landmarks;
      for (auto ldmk : map_landmarks.landmark_list) {
        if (dist(p.x, p.y, ldmk.x_f, ldmk.y_f) < sensor_range) {
          LandmarkObs l_pred;
          l_pred.id = ldmk.id_i;
          l_pred.x = ldmk.x_f;
          l_pred.y = ldmk.y_f;

          predicted_landmarks.push_back(l_pred);
        }
      }
        // transformed observations for a particular particle
      vector<LandmarkObs> transformed_obs;
      for (auto obs : observations){
        LandmarkObs trans_obs;
        trans_obs.x = cos(p.theta) * obs.x - sin(p.theta) * obs.y + p.x;
        trans_obs.y = sin(p.theta) * obs.x + cos(p.theta) * obs.y + p.y;

        transformed_obs.push_back(trans_obs);
      }
      dataAssociation(predicted_landmarks, transformed_obs);

    double particle_likelihood = 1.0;

    double mu_x, mu_y;
    for (auto obs : transformed_obs){
      for (auto ldmk : predicted_landmarks) {
        if (obs.id == ldmk.id) {
          mu_x = ldmk.x;
          mu_y = ldmk.y;
          break;
        }
      }
      double norm_factor = 2 * M_PI * std_x * std_y;
      double prob = exp( -( pow(obs.x - mu_x, 2) / (2 * std_x * std_x) + pow(obs.y - mu_y, 2) / (2 * std_y * std_y) ) );

      particle_likelihood *= prob / norm_factor;

      p.weight = particle_likelihood;
    }
    weights[i] = p.weight;
    norm_factor += p.weight;
  }
  // Normalize weights s.t. they sum to one
  for (auto& particle : particles)
    particle.weight /= (norm_factor + numeric_limits<double>::epsilon());
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> d(weights.begin(), weights.end());
  vector<Particle> particles_new;

  particles_new.resize(num_particles);
  for (int n=0; n < num_particles; ++n) {
    particles_new[n] = particles[d(gen)];
  }

  particles = particles_new;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
