#include "blur.h"

using namespace pebcl;

blur::blur()
{

}

blur::~blur()
{

}

void blur::setRadius(int Radius)
{
  this->radius = radius;
}

void blur::setLargeRadius(int LargeRadius)
{
  this->largeRadius = LargeRadius;
}

void blur::setBlurredPoints(std::vector<pcl::PointXYZRGB> BlurredPoints)
{
  this->blurredPoints = BlurredPoints;
}

void blur::setCenter(pcl::PointXYZRGB Center)
{
  this->center = Center;
}

void blur::setDistancesFromCenter(std::vector<float> distances)
{
	this->pointDistancesFromCenter = distances;
}

int blur::getRadius()
{
  return this->radius;
}

int blur::getLargeRadius()
{
  return this->largeRadius;
}

std::vector<pcl::PointXYZRGB> blur::getBlurredPoints()
{
  return this->blurredPoints;
}

pcl::PointXYZRGB blur::getCenter()
{
  return this->center;
}

std::vector<float> blur::getDistancesFromCenter()
{
	return this->pointDistancesFromCenter;
}