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

// Number of Particles
#define NUM_PARTICLES 100
// For well formed input
#define SMALL_NUMBER 0.00001

using namespace std;

default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: [Done] Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	// Standard deviations for x, y, and theta
	// Extract for better readability
	double std_x, std_y, std_theta;

	std_x = std[0];
	std_y = std[1];
	std_theta = std[2];

	// Create a normal (Gaussian) distribution for x, y and theta
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	num_particles = NUM_PARTICLES;
	double initial_weight = 1.0 / num_particles;

	// Initialize particles by sampling from above normal distrubtions
	for (int i = 0; i < num_particles; i++)
	{
		Particle p
		{
			i,
			dist_x(gen),
			dist_y(gen),
			dist_theta(gen),
			initial_weight
		};
		
		particles.push_back(p);
		weights.push_back(p.weight);
	}

	is_initialized = true;
	
	cout << "Debug: Particle Filter Initialized: " << endl;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: [Done] Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// cout << "Debug: Prediction Start" << endl;

	// Standard deviations for x, y, and theta
	// Extract for better readability
	double std_x, std_y, std_theta;

	std_x = std_pos[0];
	std_y = std_pos[1];
	std_theta = std_pos[2];

	// Create a normal (Gaussian) distribution with zero mean and above standard deviations
	normal_distribution<double> dist_x(0, std_x);
	normal_distribution<double> dist_y(0, std_y);
	normal_distribution<double> dist_theta(0, std_theta);

	// predict particles' new state using given velocity and yaw_rate
	for (int i = 0; i < num_particles; i++)
	{
		// avoid division by zero
		if (fabs(yaw_rate) > SMALL_NUMBER)
		{
			particles[i].x += velocity / yaw_rate * ( sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta) );
			particles[i].y += velocity / yaw_rate * ( cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t) );
			particles[i].theta += yaw_rate * delta_t;
		}
		else
		{
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		}
	
		//add noise
		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
	}

	// cout << "Debug: Prediction End" << endl;
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	// cout << "Debug: dataAssociation Start" << endl;

	for (auto& obs: observations)
	{
		// initialize min_dist with max value of double
		double min_dist = std::numeric_limits<float>::max();

		for (auto& pred: predicted)
		{
			// Euclidean distance between predicted and observed
			double dist_pred_obs = dist(obs.x, obs.y, pred.x, pred.y);
			
			if (dist_pred_obs < min_dist)
			{
				// closer prediction found, save
				min_dist = dist_pred_obs;
				obs.id = pred.id;
			}
		}
	}

	// cout << "Debug: dataAssociation End" << endl;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
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

	// cout << "Debug: updateWeights Start" << endl;

	// to be used later for normalization
	double total_weight = 0.0;

	for (auto& particle: particles)
	{
		// reset weight, as it will be recalculated on each update
		particle.weight = 1.0;
		
		std::vector<LandmarkObs> predicted;
		for (auto& landmark: map_landmarks.landmark_list)
		{
			double d = dist(particle.x, particle.y, landmark.x_f, landmark.y_f);

			// add landmarks within sensor_range of particle to the predicted landmarks
			if (d <= sensor_range)
			{
				LandmarkObs obs
				{
					landmark.id_i,
					landmark.x_f,
					landmark.y_f
				};
				
				predicted.push_back(obs);
			}
		}

		// in order to calculate weight, we first need to transform the measurements from
		// car's local co-ordinate system to map's co-ordinate system:
		std::vector<LandmarkObs> transformed_observations;
		for (auto& obs: observations)
		{
			LandmarkObs new_obs;
			new_obs.id = obs.id;
			new_obs.x = particle.x + obs.x * cos(particle.theta) - obs.y * sin(particle.theta);
			new_obs.y = particle.y + obs.x * sin(particle.theta) + obs.y * cos(particle.theta);
			
			transformed_observations.push_back(new_obs);
		}

		// now look for predictions that are nearest to landmark observations:
		for (auto& obs: transformed_observations)
		{
			LandmarkObs nearest_obs;

			// initialize min_dist with max value of double
			double min_dist = std::numeric_limits<float>::max();
	
			for (auto& pred: predicted)
			{
				// Euclidean distance between predicted and observed
				double curr_dist = dist(obs.x, obs.y, pred.x, pred.y);
				
				if (curr_dist < min_dist)
				{
					// closer prediction found, save
					min_dist = curr_dist;
					nearest_obs = pred;
				}
			}

			// compute weight of the particle:
			
			double std_x = std_landmark[0];
			double std_y = std_landmark[1];

			double gauss_norm = 1 / (2 * M_PI * std_x * std_y);
			double exp_term1 = pow(obs.x - nearest_obs.x, 2) / (2 * pow(std_x, 2));
			double exp_term2 = pow(obs.y - nearest_obs.y, 2) / (2 * pow(std_y, 2));
			double exponent = exp_term1 + exp_term2;
			double w = gauss_norm * exp(-1 * exponent);

			particle.weight *= w;
		}

		total_weight += particle.weight;
	}

	// normalizing weights
	int i = 0;
	for (auto& particle: particles)
	{
		particle.weight /= total_weight;
		weights[i] = particle.weight;
		i++;
	}

	// cout << "Debug: updateWeights End" << endl;
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// cout << "Debug: resample Start" << endl;

	discrete_distribution<int> index_generator(weights.begin(), weights.end());
	
	std::vector<Particle> temp_particles;

	for (int i = 0; i < num_particles; i++)
	{
		int index = index_generator(gen);
		temp_particles.push_back(particles[index]);
	}

	particles = temp_particles;

	// cout << "Debug: resample End" << endl;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

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
