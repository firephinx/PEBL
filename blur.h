/**
* blur.h
*/

#pragma once

#include <vector>
#include <pcl/point_types.h>

namespace pebcl
{
	class blur
	{
		public:

			// Constructors for blur
			blur();

			// Destructor for blur
			~blur();

			// Functions for saving data
			void setRadius(int Radius);
			void setLargeRadius(int LargeRadius);
			void setBlurredPoints(std::vector<pcl::PointXYZRGB> BlurredPoints);
			void setCenter(pcl::PointXYZRGB Center);
			void setDistancesFromCenter(std::vector<float> distances);

			// Functions for retrieving data
			int getRadius();
			int getLargeRadius();
			std::vector<pcl::PointXYZRGB> getBlurredPoints();
			pcl::PointXYZRGB getCenter();
			std::vector<float> getDistancesFromCenter();

		private:
			std::vector<pcl::PointXYZRGB> blurredPoints;
			pcl::PointXYZRGB center;
			int radius; 
			int largeRadius;
			std::vector<float> pointDistancesFromCenter;
	};
}