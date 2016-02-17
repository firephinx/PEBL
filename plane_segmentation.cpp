#include "plane_segmentation.h"

using namespace pebcl;

planeSegmentor::planeSegmentor(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{

  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  // Create the segmentation object
  pcl::SACSegmentation<pcl::PointXYZRGB> seg;
  // Optional
  seg.setOptimizeCoefficients (true);
  // Mandatory
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setDistanceThreshold (0.02);
  seg.setMaxIterations (100);
  seg.setInputCloud (cloud);
  seg.segment (*inliers, *coefficients);

  if (inliers->indices.size () == 0)
  {
    PCL_THROW_EXCEPTION (pcl::PCLException, "Could not estimate a planar model for the given dataset.");
  }

  this->MC = coefficients;
  this->PI = inliers;

  std::cerr << "Model coefficients: " << coefficients->values[0] << " " 
                                      << coefficients->values[1] << " "
                                      << coefficients->values[2] << " " 
                                      << coefficients->values[3] << std::endl;

  std::cerr << "Number of inliers : " << inliers->indices.size() << std::endl;

}


planeSegmentor::~planeSegmentor()
{

}

  