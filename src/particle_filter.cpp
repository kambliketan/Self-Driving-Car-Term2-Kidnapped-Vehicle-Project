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

// for well formed input
#define NUM_PARTICLES 500

using namespace std;

// initialize random engine
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
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
	double initial_weight = 1.0;

	// Initialize particles by sampling from above normal distrubtions
	for (int i = 0; i < num_particles; i++)
	{
		Particle p;

		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = initial_weight;
		
		particles.push_back(p);
		weights.push_back(initial_weight);
	}

	is_initialized = true;
	cout << "Particle Filter Initialized: " << endl;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// Standard deviations for x, y, and theta
	// Extract for better readability
	// cout << "Here 0." << endl;
	double std_x, std_y, std_theta;

	std_x = std_pos[0];
	std_y = std_pos[1];
	std_theta = std_pos[2];

	// cout << "Here 0.1." << endl;
	// Create a normal (Gaussian) distribution with zero mean and above standard deviations
	normal_distribution<double> dist_x(0, std_x);
	normal_distribution<double> dist_y(0, std_y);
	normal_distribution<double> dist_theta(0, std_theta);

	// cout << "Here 1." << endl;

	// predict particles' new state using given velocity and yaw_rate
	for (int i = 0; i < num_particles; i++)
	{
		Particle *p = &particles[i];

		// renaming for readability
		double p_x = (*p).x;
		double p_y = (*p).y;
		double v = velocity;
		double yaw = (*p).theta;
		double yawd = yaw_rate;
		
		//predicted state values
		double px_p, py_p, yaw_p;
	
		// cout << "Here 2." << endl;

		//avoid division by zero
		if (fabs(yawd) > 0.001)
		{
			px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
			py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
			yaw_p = yaw + yawd*delta_t;
		}
		else
		{
			px_p = p_x + v*delta_t*cos(yaw);
			py_p = p_y + v*delta_t*sin(yaw);
		}
	
		//add noise
		px_p = px_p + dist_x(gen);
		py_p = py_p + dist_y(gen);
		yaw_p = yaw_p + dist_theta(gen);
	
		// cout << "Here 3." << endl;

		//write predicted sigma point into right column
		(*p).x = px_p;
		(*p).y = py_p;
		(*p).theta = yaw_p;
	}

	// cout << "Particle Filter Prediction Complete." << endl;
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (int i = 0; i < observations.size(); i++)
	{
		LandmarkObs *obs = &observations[i];

		int nearest_obs = 0;

		for (int j = 0; j < predicted.size(); j++)
		{
			LandmarkObs pred = predicted[j];

			// Euclidean distance between predicted and observed
			double dist_pred_obs = dist((*obs).x, (*obs).y, pred.x, pred.y);

			// initialize min_dist with max value of double
			double min_dist = std::numeric_limits<double>::max();
			if (dist_pred_obs < min_dist)
			{
				// closer prediction found, save
				min_dist = dist_pred_obs;
				nearest_obs = pred.id;
			}
		}

		// update observation with the nearest prediction found
		(*obs).id = nearest_obs;
	}
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

	// cout << "Here 100." << endl;
	for (int i = 0; i < num_particles; i++)
	{
		Particle *p = &particles[i];
		
		// renaming for readability
		double p_x = (*p).x;
		double p_y = (*p).y;
		double yaw = (*p).theta;

		std::vector<LandmarkObs> predicted;

		for (int j = 0; j < map_landmarks.landmark_list.size(); j++)
		{
			Map::single_landmark_s land = map_landmarks.landmark_list[j];
			
			int l_id = land.id_i;
			float l_x = land.x_f;
			float l_y = land.y_f;

			// add landmarks within sensor_range of particle to the predicted landmarks
			if (fabs(p_x - l_x) <= sensor_range
				&& fabs(p_y - l_y) <= sensor_range)
			{
				LandmarkObs obs;
				
				obs.id = l_id;
				obs.x = l_x;
				obs.y = l_y;
				
				predicted.push_back(obs);
			}
		}

		// cout << "Here 101." << endl;
		std::vector<LandmarkObs> transformed;

		for (int k = 0; k < observations.size(); k++)
		{
			LandmarkObs obs = observations[k];

			double x, y;

			// transform to map x co-ordinate
			x = p_x + cos(yaw) * obs.x - sin(yaw) * obs.y;
			// transform to map y co-ordinate
			y = p_y + sin(yaw) * obs.x + cos(yaw) * obs.y;

			LandmarkObs new_obs;
			
			new_obs.id = obs.id;
			new_obs.x = x;
			new_obs.y = y;
			
			transformed.push_back(new_obs);
		}

		// cout << "Here 102." << endl;
		dataAssociation(predicted, transformed);

		// cout << "Here 103." << endl;
		for (int m = 0; m < transformed.size(); m++)
		{
			LandmarkObs obs = transformed[m];

			double x, y;

			for (int n = 0; n < predicted.size(); n++)
			{
				LandmarkObs pred = predicted[n];

				if (pred.id == obs.id)
				{
					x = pred.x;
					y = pred.y;
				}
			}

			double std_x = std_landmark[0];
			double std_y = std_landmark[1];

			double gauss_norm = (1 / (2 * M_PI * std_x * std_y));
			double exp_term1 = pow(obs.x - x, 2) / (2 * pow(std_x, 2));
			double exp_term2 = pow(obs.y - y, 2) / (2 * pow(std_y, 2));
			double exponent = exp_term1 + exp_term2;
			double w = gauss_norm * exp(-exponent);

			(*p).weight *= w;
			weights[i] = (*p).weight;
		}

		// cout << "Here 104." << endl;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// using the resampling wheel approach
	uniform_int_distribution<int> uni_int_dist(0, num_particles - 1);
	int index = uni_int_dist(gen);

	double max_w = *std::max_element(weights.begin(), weights.end());
	uniform_real_distribution<double> uni_real_dist(0.0, 2 * max_w);

	double beta = 0.0;
	std::vector<Particle> temp_particles;

	for (int i = 0; i < num_particles; i++)
	{
		beta += uni_real_dist(gen);

		while (beta > weights[index])
		{
			beta -= weights[index];
			index = (index + 1) % num_particles;
		}

		temp_particles.push_back(particles[index]);
	}

	particles = temp_particles;
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
