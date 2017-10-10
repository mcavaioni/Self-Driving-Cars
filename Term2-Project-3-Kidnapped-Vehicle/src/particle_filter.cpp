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
#include <cmath>

#include "particle_filter.h"


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	num_particles = 50;

	std::default_random_engine gen;
	std::normal_distribution<double> dist_x(x, std[0]);
	std::normal_distribution<double> dist_y(y, std[1]);
	std::normal_distribution<double> dist_theta(theta, std[2]);

	for(int i=0; i<num_particles; i++){

		Particle particle;
		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 1.0;

		particles.push_back(particle);
		weights.push_back(particle.weight);
	}

	is_initialized = true;
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	std::default_random_engine gen;
	std::normal_distribution<double> dist_x(0, std_pos[0]);
	std::normal_distribution<double> dist_y(0, std_pos[1]);
	std::normal_distribution<double> dist_theta(0, std_pos[2]);

	for(int i=0; i<num_particles; i++){
		//for each particle, update the particle location based on the velocity and yaw rate measurements
		//and adding noise.
		Particle particle = particles[i];
		particle.x = particle.x + (velocity/yaw_rate)*(sin(particle.theta+yaw_rate*delta_t)-sin(particle.theta)) + dist_x(gen);
		particle.y = particle.y + (velocity/yaw_rate)*(cos(particle.theta)-cos(particle.theta + yaw_rate*delta_t)) + dist_y(gen);
		particle.theta = particle.theta + yaw_rate*delta_t + dist_theta(gen);

		particles[i] = particle;

	}

}

// void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	
// }

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
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html


	double std_x = std_landmark[0];
	double std_y = std_landmark[1];


	for(int i=0; i<num_particles; i++){
		Particle particle = particles[i];
		
		int tot_observations = observations.size();
		double weight = 1.0;

		LandmarkObs transformed_obs;
		//Each observation is transformed from vehicle coordinates to map coordinates
		for(int j=0; j<tot_observations; j++){
			transformed_obs.x = observations[j].x * cos(particle.theta) - observations[j].y * sin(particle.theta) + particle.x;
			transformed_obs.y = observations[j].x * sin(particle.theta) + observations[j].y * cos(particle.theta) + particle.y;
			transformed_obs.id = observations[j].id;



			int tot_landmarks = map_landmarks.landmark_list.size();

			Map::single_landmark_s closest_land;
			double shortest_dist = sensor_range;
			//Find the transformed measurement that is closest to each landmark and assign the 
			//observed measurement to this particular landmark.
			for(int m=0; m<tot_landmarks; m++){
				double current_dist = dist(transformed_obs.x, transformed_obs.y, map_landmarks.landmark_list[m].x_f, map_landmarks.landmark_list[m].y_f);
				if(current_dist < shortest_dist){
					shortest_dist = current_dist;
					closest_land = map_landmarks.landmark_list[m];
				}
			}

			// calculat Multi-variate Gaussian probability:
			double exp_arg = -0.5 * (pow((transformed_obs.x-closest_land.x_f),2)/(std_x*std_x) + pow((transformed_obs.y-closest_land.y_f),2)/(std_y*std_y));
			double probability = (1/(2*std_x*std_y*M_PI))*exp(exp_arg);

			weight *= probability;
		}

		particle.weight = weight;
		weights[i] = weight;
		particles[i] = particle;
	}


	
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	std::vector<Particle> new_particles;
	

	std::default_random_engine generator;
  std::discrete_distribution<> distribution(weights.begin(), weights.end());

  int index = distribution(generator);
  double beta = 0.0;

  auto result = std::minmax_element(weights.begin(), weights.end());
  double max_weight = (weights[result.second - weights.begin()]);


  for(int i=0; i<num_particles; i++){
  	beta += distribution(generator) * 2.0 * max_weight;
  	while(beta > weights[index]){
  		beta -= weights[index];
  		index = (index + 1)% num_particles;
  	}
  	new_particles.push_back(particles[index]);
  }
  particles = new_particles;

}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}

