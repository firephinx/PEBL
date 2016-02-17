/**
* moment_of_inertia.h
*/

#pragma once

#include <pclextras/moment_of_inertia_estimation.h>
#include <pclextras/moment_of_inertia_estimation.cpp>
#include <vector>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <boost/thread/thread.hpp>

namespace pebcl
{
	class MOIBoundingBox
	{
		public:

			// Constructors for the planeSegmentor
			MOIBoundingBox(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);

			// Destructor for planeSegmentor
			~MOIBoundingBox();

			std::vector <float> moment_of_inertia;
			std::vector <float> eccentricity;
			pcl::PointXYZRGB min_point_AABB;
			pcl::PointXYZRGB max_point_AABB;
			pcl::PointXYZRGB min_point_OBB;
			pcl::PointXYZRGB max_point_OBB;
			pcl::PointXYZRGB position_OBB;
			Eigen::Matrix3f rotational_matrix_OBB;
			float major_value, middle_value, minor_value;
			Eigen::Vector3f major_vector, middle_vector, minor_vector;
			Eigen::Vector3f mass_center;

			Eigen::VectorXf sidePlane1;
		    Eigen::VectorXf sidePlane2;
		    Eigen::VectorXf sidePlane3;
		    Eigen::VectorXf sidePlane4;

		private:

	};
}