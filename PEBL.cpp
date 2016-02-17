/* Plane Extraction and Object Clustering for Camera Localization
 *
 * PEBCL.cpp
 * Written by Kevin Zhang
 */

#include <iostream>
#include <math.h>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <pcl/common/distances.h>
#include <pcl/conversions.h>
#include <pcl/console/parse.h>
#include <pcl/exceptions.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/pfh.h>
#include <pcl/features/shot_omp.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/voxel_grid_occlusion_estimation.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/keypoints/susan.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/point_types_conversion.h>
#include <pcl/registration/transforms.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_features.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>

#include "plane_segmentation.h"
#include "plane_segmentation.cpp"
#include "moment_of_inertia.h"
#include "moment_of_inertia.cpp"

std::string filename;
std::string lib_filename;

void
downsample (pcl::PointCloud<pcl::PointXYZRGB>::Ptr &points, float leaf_size,
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr &downsampled_out)
{
  pcl::VoxelGrid<pcl::PointXYZRGB> vox_grid;
  vox_grid.setLeafSize (leaf_size, leaf_size, leaf_size);
  vox_grid.setInputCloud (points);
  vox_grid.filter (*downsampled_out);
}

void
compute_surface_normals (pcl::PointCloud<pcl::PointXYZRGB>::Ptr &points, float normal_radius,
                         pcl::PointCloud<pcl::Normal>::Ptr &normals_out)
{
  pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> norm_est;
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());

  // Use a FLANN-based KdTree to perform neighborhood searches
  norm_est.setSearchMethod (tree);

  // Specify the size of the local neighborhood to use when computing the surface normals
  norm_est.setRadiusSearch (normal_radius);

  // Set the input points
  norm_est.setInputCloud (points);

  // Estimate the surface normals and store the result in "normals_out"
  norm_est.compute (*normals_out);
}

void
detect_keypoints (pcl::PointCloud<pcl::PointXYZRGB>::Ptr &points,
                  float min_scale, int nr_octaves, int nr_scales_per_octave, float min_contrast,
                  pcl::PointCloud<pcl::PointWithScale>::Ptr &keypoints_out)
{
  pcl::SIFTKeypoint<pcl::PointXYZRGB, pcl::PointWithScale> sift_detect;
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());

  // Use a FLANN-based KdTree to perform neighborhood searches
  sift_detect.setSearchMethod (tree);

  // Set the detection parameters
  sift_detect.setScales (min_scale, nr_octaves, nr_scales_per_octave);
  sift_detect.setMinimumContrast (min_contrast);

  // Set the input
  sift_detect.setInputCloud (points);

  // Detect the keypoints and store them in "keypoints_out"
  sift_detect.compute (*keypoints_out);
}

double
computeCloudResolution (const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud)
{
  double res = 0.0;
  int n_points = 0;
  int nres;
  std::vector<int> indices (2);
  std::vector<float> sqr_distances (2);
  pcl::search::KdTree<pcl::PointXYZRGB> tree;
  tree.setInputCloud (cloud);

  for (size_t i = 0; i < cloud->size (); ++i)
  {
    if (! pcl_isfinite ((*cloud)[i].x))
    {
      continue;
    }
    //Considering the second neighbor since the first is the point itself.
    nres = tree.nearestKSearch (i, 2, indices, sqr_distances);
    if (nres == 2)
    {
      res += sqrt (sqr_distances[1]);
      ++n_points;
    }
  }
  if (n_points != 0)
  {
    res /= n_points;
  }
  return res;
}

void
detect_ISSkeypoints (pcl::PointCloud<pcl::PointXYZRGB>::Ptr &points,
                  pcl::PointCloud<pcl::Normal>::Ptr &normals,
                  pcl::PointCloud<pcl::PointXYZRGB>::Ptr &keypoints_out)
{
  double model_resolution;
  model_resolution = computeCloudResolution(points);

  pcl::ISSKeypoint3D<pcl::PointXYZRGB, pcl::PointXYZRGB> iss_detector;
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());

  // Use a FLANN-based KdTree to perform neighborhood searches
  iss_detector.setSearchMethod (tree);

  // Set the detection parameters
  iss_detector.setSalientRadius (6 * model_resolution);
  iss_detector.setNonMaxRadius (4 * model_resolution);
  iss_detector.setThreshold21 (0.975);
  iss_detector.setThreshold32 (0.975);
  iss_detector.setNormals (normals);
  iss_detector.setMinNeighbors (5);

  // Set the input
  iss_detector.setInputCloud (points);

  // Detect the keypoints and store them in "keypoints_out"
  iss_detector.compute (*keypoints_out);
}

void
detect_SUSANkeypoints (pcl::PointCloud<pcl::PointXYZRGB>::Ptr &points,
                  pcl::PointCloud<pcl::Normal>::Ptr &normals,
                  pcl::PointCloud<pcl::PointXYZRGB>::Ptr &keypoints_out)
{
  /*pcl::PointCloud<pcl::PointXYZI>::Ptr grayPoints(new pcl::PointCloud<pcl::PointXYZI>());
  pcl::PointCloudXYZRGBtoXYZI(*points, *grayPoints);*/

  pcl::SUSANKeypoint<pcl::PointXYZRGB, pcl::PointXYZRGB> susan_detector;
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());

  // Use a FLANN-based KdTree to perform neighborhood searches
  susan_detector.setSearchMethod (tree);

  // Set the detection parameters
  if (normals->points.size() == points->points.size())
  {
    susan_detector.setNormals (normals);
  }
  // Set the input
  susan_detector.setInputCloud (points);

  // Detect the keypoints and store them in "keypoints_out"
  susan_detector.compute (*keypoints_out);
}

void visualize_keypoints (const pcl::PointCloud<pcl::PointXYZRGB>::Ptr points,
                          const pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints)
{
  // Add the points to the vizualizer
  std::cout << "Num Keypoints = " << keypoints->size () << std::endl;
  std::cout << "Num Total Points = " << points->size () << std::endl;
  pcl::visualization::PCLVisualizer viz;
  viz.addPointCloud (points, "points");

  // Draw each keypoint as a sphere
  for (size_t i = 0; i < keypoints->size (); ++i)
  {
    // Get the point data
    pcl::PointXYZRGB & p = keypoints->points[i];

    // Pick the radius of the sphere
    pcl::PointXYZI k;
    pcl::PointXYZRGBtoXYZI(p,k);
    float r = k.intensity/100;
    // * Note: the scale is given as the standard deviation of a Gaussian blur, so a
    //   radius of 2*p.scale is a good illustration of the extent of the keypoint

    // Generate a unique string for each sphere
    std::stringstream ss ("keypoint");
    ss << i;

    // Add a sphere at the keypoint
    viz.addSphere (p, r, 1.0, 0.0, 0.0, ss.str ());
  }

  // Give control over to the visualizer
  viz.spin ();
}

void visualize_keypointsScaled (const pcl::PointCloud<pcl::PointXYZRGB>::Ptr points,
                          const pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints)
{
  // Add the points to the vizualizer
  pcl::visualization::PCLVisualizer viz;
  viz.addPointCloud (points, "points");

  // Draw each keypoint as a sphere
  for (size_t i = 0; i < keypoints->size (); ++i)
  {
    // Get the point data
    const pcl::PointWithScale & p = keypoints->points[i];

    // Pick the radius of the sphere *
    float r = 2*p.scale;
    std::cout << "Scale:" << r << std::endl;
    // * Note: the scale is given as the standard deviation of a Gaussian blur, so a
    //   radius of 2*p.scale is a good illustration of the extent of the keypoint

    // Generate a unique string for each sphere
    std::stringstream ss ("keypoint");
    ss << i;

    // Add a sphere at the keypoint
    viz.addSphere (p, r, 1.0, 0.0, 0.0, ss.str ());
  }

  // Give control over to the visualizer
  viz.spin ();
}

void
compute_PFH_features_at_keypoints (pcl::PointCloud<pcl::PointXYZRGB>::Ptr &points, 
                                   pcl::PointCloud<pcl::Normal>::Ptr &normals, 
                                   pcl::PointCloud<pcl::PointWithScale>::Ptr &keypoints, float feature_radius,
                                   pcl::PointCloud<pcl::PFHSignature125>::Ptr &descriptors_out)
{
  // Create a PFHEstimation object
  pcl::PFHEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::PFHSignature125> pfh_est;
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());

  // Set it to use a FLANN-based KdTree to perform its neighborhood searches
  pfh_est.setSearchMethod (tree);

  // Specify the radius of the PFH feature
  pfh_est.setRadiusSearch (feature_radius);

  /* This is a little bit messy: since our keypoint detection returns PointWithScale points, but we want to
   * use them as an input to our PFH estimation, which expects clouds of PointXYZRGB points.  To get around this,
   * we'll use copyPointCloud to convert "keypoints" (a cloud of type PointCloud<PointWithScale>) to 
   * "keypoints_xyzrgb" (a cloud of type PointCloud<PointXYZRGB>).  Note that the original cloud doesn't have any RGB 
   * values, so when we copy from PointWithScale to PointXYZRGB, the new r,g,b fields will all be zero.
   */
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints_xyzrgb (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::copyPointCloud (*keypoints, *keypoints_xyzrgb);

  // Use all of the points for analyzing the local structure of the cloud
  pfh_est.setSearchSurface (points);  
  pfh_est.setInputNormals (normals);  

  // But only compute features at the keypoints
  pfh_est.setInputCloud (keypoints_xyzrgb);

  // Compute the features
  pfh_est.compute (*descriptors_out);
}

void
compute_PFH_features_at_ISSkeypoints (pcl::PointCloud<pcl::PointXYZRGB>::Ptr &points, 
                                   pcl::PointCloud<pcl::Normal>::Ptr &normals, 
                                   pcl::PointCloud<pcl::PointXYZRGB>::Ptr &keypoints, float feature_radius,
                                   pcl::PointCloud<pcl::PFHSignature125>::Ptr &descriptors_out)
{
  // Create a PFHEstimation object
  pcl::PFHEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::PFHSignature125> pfh_est;
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());

  // Set it to use a FLANN-based KdTree to perform its neighborhood searches
  pfh_est.setSearchMethod (tree);

  // Specify the radius of the PFH feature
  pfh_est.setRadiusSearch (feature_radius);

  /* This is a little bit messy: since our keypoint detection returns PointWithScale points, but we want to
   * use them as an input to our PFH estimation, which expects clouds of PointXYZRGB points.  To get around this,
   * we'll use copyPointCloud to convert "keypoints" (a cloud of type PointCloud<PointWithScale>) to 
   * "keypoints_xyzrgb" (a cloud of type PointCloud<PointXYZRGB>).  Note that the original cloud doesn't have any RGB 
   * values, so when we copy from PointWithScale to PointXYZRGB, the new r,g,b fields will all be zero.
   */
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints_xyzrgb (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::copyPointCloud (*keypoints, *keypoints_xyzrgb);

  // Use all of the points for analyzing the local structure of the cloud
  pfh_est.setSearchSurface (points);  
  pfh_est.setInputNormals (normals);  

  // But only compute features at the keypoints
  pfh_est.setInputCloud (keypoints_xyzrgb);

  // Compute the features
  pfh_est.compute (*descriptors_out);
}

void
compute_SHOTColorFeatures_at_keypoints (pcl::PointCloud<pcl::PointXYZRGB>::Ptr &points, 
                                   pcl::PointCloud<pcl::Normal>::Ptr &normals, 
                                   pcl::PointCloud<pcl::PointXYZRGB>::Ptr &keypoints, float feature_radius,
                                   pcl::PointCloud<pcl::SHOT1344>::Ptr &descriptors_out)
{
  pcl::SHOTColorEstimationOMP<pcl::PointXYZRGB, pcl::Normal, pcl::SHOT1344> descr_est;
  descr_est.setRadiusSearch (feature_radius);

  descr_est.setInputCloud (keypoints);
  descr_est.setInputNormals (normals);
  descr_est.setSearchSurface (points);
  descr_est.compute (*descriptors_out);
}

void
find_feature_correspondences (pcl::PointCloud<pcl::SHOT1344>::Ptr &source_descriptors,
                              pcl::PointCloud<pcl::SHOT1344>::Ptr &target_descriptors,
                              std::vector<int> &correspondences_out, std::vector<float> &correspondence_scores_out)
{
  // Resize the output vector
  correspondences_out.resize (source_descriptors->size ());
  correspondence_scores_out.resize (source_descriptors->size ());

  // Use a KdTree to search for the nearest matches in feature space
  pcl::KdTreeFLANN<pcl::SHOT1344> descriptor_kdtree;
  descriptor_kdtree.setInputCloud (target_descriptors);

  // Find the index of the best match for each keypoint, and store it in "correspondences_out"
  const int k = 1;
  std::vector<int> k_indices (k);
  std::vector<float> k_squared_distances (k);
  for (size_t i = 0; i < source_descriptors->size (); ++i)
  {
    descriptor_kdtree.nearestKSearch (*source_descriptors, i, k, k_indices, k_squared_distances);
    correspondences_out[i] = k_indices[0];
    correspondence_scores_out[i] = k_squared_distances[0];
  }
}

void filter_correspondences(const pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints1,
                            const pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints2,
                            const std::vector<int> &correspondences,
                            const std::vector<float> &correspondence_scores,
                            pcl::PointCloud<pcl::PointWithScale>::Ptr &out_keypoints1,
                            std::vector<int> &remaining_correspondences, std::vector<float> &remaining_correspondence_scores)
{
  int numSearch = keypoints1->points.size();
  float largestDistance = 0.0;
  int firstPointIndex;
  int secondPointIndex;

  std::vector<float> temp;

  for(int i = 0; i < numSearch; i++)
  {
    float distanceTotal = 0.0;
    for(int k = 0; k < numSearch; k++)
    {
      distanceTotal += pcl::euclideanDistance(keypoints2->points[correspondences[i]],keypoints2->points[correspondences[k]]);
    }
    float avgDistance = distanceTotal/numSearch;
    temp.push_back(avgDistance);
  }

  std::sort (temp.begin (), temp.end ());
  float median_score = temp[temp.size ()/4];

 /* for(int i = 0; i < numSearch; i++)
  {
    for(int k = i+1; k < numSearch; k++)
    {
      float distanceBetweenKeyPoints = pcl::euclideanDistance(keypoints1->points[i],keypoints1->points[k]);

      if(distanceBetweenKeyPoints > largestDistance)
      {
        firstPointIndex = i;
        secondPointIndex = k;
        largestDistance = distanceBetweenKeyPoints;
      }
    }
  }*/

  //int j = 0;
  //int leastDistance = largestDistance; 
  for(int i = 0; i < numSearch; i++)
  {
    /*float distanceTotal = 0.0;
    for(int k = 0; k < numSearch; k++)
    {
      distanceTotal += pcl::euclideanDistance(keypoints2->points[correspondences[i]],keypoints2->points[correspondences[k]]);
    }
    float avgDistance = distanceTotal/numSearch;*/
    if(temp[i] < median_score)
    {
      //leastDistance = avgDistance;
      out_keypoints1->points.push_back(keypoints1->points[i]);
      remaining_correspondences.push_back(correspondences[i]);
      remaining_correspondence_scores.push_back(correspondence_scores[i]);
      //j++;
    }
  }
}

void filter_ISScorrespondences(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints1,
                            const pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints2,
                            const std::vector<int> &correspondences,
                            const std::vector<float> &correspondence_scores,
                            pcl::PointCloud<pcl::PointXYZRGB>::Ptr &out_keypoints1,
                            std::vector<int> &remaining_correspondences, std::vector<float> &remaining_correspondence_scores)
{
  int numSearch = keypoints1->points.size();
  float largestDistance = 0.0;
  int firstPointIndex;
  int secondPointIndex;

  std::vector<float> scores (correspondence_scores);
  std::sort (scores.begin (), scores.end ());
  float median_score = scores[scores.size ()/2];

  std::vector<float> temp;

  for(int i = 0; i < numSearch; i++)
  {
    float distanceTotal = 0.0;
    for(int k = 0; k < numSearch; k++)
    {
      distanceTotal += pcl::euclideanDistance(keypoints2->points[correspondences[i]],keypoints2->points[correspondences[k]]);
    }
    float avgDistance = distanceTotal/numSearch;
    temp.push_back(avgDistance);
  }

  std::sort (temp.begin (), temp.end ());
  float median_distance = temp[temp.size ()/4];

 /* for(int i = 0; i < numSearch; i++)
  {
    for(int k = i+1; k < numSearch; k++)
    {
      float distanceBetweenKeyPoints = pcl::euclideanDistance(keypoints1->points[i],keypoints1->points[k]);

      if(distanceBetweenKeyPoints > largestDistance)
      {
        firstPointIndex = i;
        secondPointIndex = k;
        largestDistance = distanceBetweenKeyPoints;
      }
    }
  }*/

  //int j = 0;
  //int leastDistance = largestDistance; 
  for(int i = 0; i < numSearch; i++)
  {
    /*float distanceTotal = 0.0;
    for(int k = 0; k < numSearch; k++)
    {
      distanceTotal += pcl::euclideanDistance(keypoints2->points[correspondences[i]],keypoints2->points[correspondences[k]]);
    }
    float avgDistance = distanceTotal/numSearch;*/
    if(temp[i] < median_distance && correspondence_scores[i] < median_score)
    {
      //leastDistance = avgDistance;
      out_keypoints1->points.push_back(keypoints1->points[i]);
      remaining_correspondences.push_back(correspondences[i]);
      remaining_correspondence_scores.push_back(correspondence_scores[i]);
      //j++;
    }
  }
}

void visualize_correspondences (const pcl::PointCloud<pcl::PointXYZRGB>::Ptr points1,
                                const pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints1,
                                const pcl::PointCloud<pcl::PointXYZRGB>::Ptr points2,
                                const pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints2,
                                const std::vector<int> &correspondences,
                                const std::vector<float> &correspondence_scores)
{
  // We want to visualize two clouds side-by-side, so do to this, we'll make copies of the clouds and transform them
  // by shifting one to the left and the other to the right.  Then we'll draw lines between the corresponding points

  // Create some new point clouds to hold our transformed data
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr points_left (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints_left (new pcl::PointCloud<pcl::PointWithScale>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr points_right (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints_right (new pcl::PointCloud<pcl::PointWithScale>);

  // Shift the first clouds' points to the left
  //const Eigen::Vector3f translate (0.0, 0.0, 0.3);
  //const Eigen::Vector3f translate (0.4, 0.0, 0.0);
  //const Eigen::Quaternionf no_rotation (0, 0, 0, 0);
  //pcl::transformPointCloud (*points1, *points_left, -translate, no_rotation);
  //pcl::transformPointCloud (*keypoints1, *keypoints_left, -translate, no_rotation);

  // Shift the second clouds' points to the right
  //pcl::transformPointCloud (*points2, *points_right, translate, no_rotation);
  //pcl::transformPointCloud (*keypoints2, *keypoints_right, translate, no_rotation);

  // Add the clouds to the vizualizer
  pcl::visualization::PCLVisualizer viz;
  viz.addPointCloud (points1, "points_left");
  viz.addPointCloud (points2, "points_right");
  std::cout << "No Segfault yet." << std::endl;

  // Compute the median correspondence score
  /*std::vector<float> temp (correspondence_scores);
  std::sort (temp.begin (), temp.end ());
  float median_score = temp[temp.size ()/2];*/

  // Draw lines between the best corresponding points
  for (size_t i = 0; i < keypoints1->size (); ++i)
  {
    /*if (correspondence_scores[i] > median_score)
    {
      continue; // Don't draw weak correspondences
    }*/

    // Get the pair of points
    const pcl::PointWithScale & p_left = keypoints1->points[i];
    const pcl::PointWithScale & p_right = keypoints2->points[correspondences[i]];

    // Generate a random (bright) color
    double r = (rand() % 100);
    double g = (rand() % 100);
    double b = (rand() % 100);
    double max_channel = std::max (r, std::max (g, b));
    r /= max_channel;
    g /= max_channel;
    b /= max_channel;

    // Generate a unique string for each line
    std::stringstream ss ("line");
    ss << i;

    // Draw the line
    viz.addLine (p_left, p_right, r, g, b, ss.str ());
    std::cout << "Line " << i << std::endl;
  }

  // Give control over to the visualizer
  viz.spin ();
}

void visualize_ISScorrespondences (const pcl::PointCloud<pcl::PointXYZRGB>::Ptr points1,
                                const pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints1,
                                const pcl::PointCloud<pcl::PointXYZRGB>::Ptr points2,
                                const pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints2,
                                const std::vector<int> &correspondences,
                                const std::vector<float> &correspondence_scores)
{
  // We want to visualize two clouds side-by-side, so do to this, we'll make copies of the clouds and transform them
  // by shifting one to the left and the other to the right.  Then we'll draw lines between the corresponding points

  // Create some new point clouds to hold our transformed data
  /*pcl::PointCloud<pcl::PointXYZRGB>::Ptr points_left (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints_left (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr points_right (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints_right (new pcl::PointCloud<pcl::PointXYZRGB>);*/

  // Shift the first clouds' points to the left
  //const Eigen::Vector3f translate (0.0, 0.0, 0.3);
  //const Eigen::Vector3f translate (0.4, 0.0, 0.0);
  //const Eigen::Quaternionf no_rotation (0, 0, 0, 0);
  //pcl::transformPointCloud (*points1, *points_left, -translate, no_rotation);
  //pcl::transformPointCloud (*keypoints1, *keypoints_left, -translate, no_rotation);

  // Shift the second clouds' points to the right
  //pcl::transformPointCloud (*points2, *points_right, translate, no_rotation);
  //pcl::transformPointCloud (*keypoints2, *keypoints_right, translate, no_rotation);

  // Add the clouds to the vizualizer
  pcl::visualization::PCLVisualizer viz;
  viz.addPointCloud (points1, "points_left");
  viz.addPointCloud (points2, "points_right");
  std::cout << "No Segfault yet." << std::endl;

  // Compute the median correspondence score
  std::vector<float> temp (correspondence_scores);
  std::sort (temp.begin (), temp.end ());
  float median_score = temp[temp.size ()/2];

  // Draw lines between the best corresponding points
  for (size_t i = 0; i < keypoints1->size (); ++i)
  {
    /*if (correspondence_scores[i] > median_score)
    {
      continue; // Don't draw weak correspondences
    }*/

    // Get the pair of points
    const pcl::PointXYZRGB & p_left = keypoints1->points[i];
    const pcl::PointXYZRGB & p_right = keypoints2->points[correspondences[i]];

    // Generate a random (bright) color
    double r = (rand() % 100);
    double g = (rand() % 100);
    double b = (rand() % 100);
    double max_channel = std::max (r, std::max (g, b));
    r /= max_channel;
    g /= max_channel;
    b /= max_channel;

    // Generate a unique string for each line
    std::stringstream ss ("line");
    ss << i;

    // Draw the line
    viz.addLine (p_left, p_right, r, g, b, ss.str ());
  }

  // Give control over to the visualizer
  viz.spin ();
}

int main (int argc, char** argv)
{
  //Model & scene filenames
  std::vector<int> filenames;
  filenames = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");
  if (filenames.size () != 2)
  {
    std::cout << "Filenames missing.\n";
    exit (-1);
  }

  filename = argv[filenames[0]];
  lib_filename = argv[filenames[1]];

  // Read in the cloud data
  pcl::PCDReader reader;
  pcl::PCDReader libreader;
  pcl::PCLPointCloud2 input, libInput;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>), model (new pcl::PointCloud<pcl::PointXYZRGB>);
  reader.read (filename, input);
  libreader.read (lib_filename, libInput);
  pcl::fromPCLPointCloud2(input,*cloud);
  pcl::fromPCLPointCloud2(libInput,*model);
  std::cout << "PointCloud before filtering has: " << cloud->points.size () << " data points." << std::endl; //*

  /* Filters */

  // Statistical Outlier Removal
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr sorOutput(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
  sor.setInputCloud(cloud);
  sor.setMeanK(10);
  sor.setStddevMulThresh(0.5);
  sor.filter(*sorOutput);

  // Voxel Grid
  // Create the filtering object: downsample the dataset using a leaf size of 1cm
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_f(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::VoxelGrid<pcl::PointXYZRGB> vg;
  vg.setInputCloud (sorOutput);
  vg.setLeafSize (0.01f, 0.01f, 0.01f);
  vg.filter (*cloud_f);

  std::cout << "PointCloud after filtering has: " << cloud_f->points.size ()  << " data points." << std::endl; 

  // Storage of outputted and filtered clouds
  pcl::ModelCoefficients::Ptr modelC[100];
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr planeClouds[100];
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filter (new pcl::PointCloud<pcl::PointXYZRGB>), outputCloud (new pcl::PointCloud<pcl::PointXYZRGB>), planeCloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PCDWriter writer;

  // Uncomment to save the filtered pointcloud
  /*std::stringstream ss2;
  ss2 << "filtered_cloud_cluster.pcd";
  writer.write<pcl::PointXYZRGB> (ss2.str (), *cloud_f, false); */

  int num_filtered_points = (int) cloud_f->points.size();
  int i = 0;
  //std::vector<blur> bls;

  pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints (new pcl::PointCloud<pcl::PointWithScale>);

  /*// Compute keypoints
  const float min_scale = 0.01;
  const int nr_octaves = 3;
  const int nr_octaves_per_scale = 3;
  const float min_contrast = 10.0;
  detect_keypoints (cloud_f, min_scale, nr_octaves, nr_octaves_per_scale, min_contrast, keypoints);

  visualize_keypoints (cloud_f, keypoints);*/

  while(cloud_f->points.size() > 0.33 * num_filtered_points)
  {
    try
    {
      planeSegmentor pS(cloud_f);
      modelC[i] = pS.MC;

      // Extract the planar inliers from the input cloud
      pcl::ExtractIndices<pcl::PointXYZRGB> extract;
      extract.setInputCloud(cloud_f);
      extract.setIndices(pS.PI);
      extract.setNegative (false);

      //Get the points associated with the planar surface
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZRGB> ());
      extract.filter (*cloud_plane);

      int numPointsOnPlane = cloud_plane->points.size();
      for (int k = 0; k < numPointsOnPlane; ++k)
      {
        pcl::PointXYZRGB tempPoint(cloud_plane->points[k].r, cloud_plane->points[k].g, cloud_plane->points[k].b);
        tempPoint.x = cloud_plane->points[k].x;
        tempPoint.y = (cloud_plane->points[k].y)*(-1.0);
        tempPoint.z = (cloud_plane->points[k].z)*(-1.0);
        planeCloud->points.push_back(tempPoint);
      }

      // Perform Euclidean Cluster Extraction on the planar surface
      pcl::search::KdTree<pcl::PointXYZRGB>::Ptr plane_t (new pcl::search::KdTree<pcl::PointXYZRGB>);
      plane_t->setInputCloud (cloud_plane);

      std::vector<pcl::PointIndices> cluster_indices;
      pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
      ec.setClusterTolerance (0.02); // 2cm
      ec.setMinClusterSize (100);
      ec.setMaxClusterSize (cloud_plane->points.size());
      ec.setSearchMethod (plane_t);
      ec.setInputCloud (cloud_plane);
      ec.extract (cluster_indices);

      // Find the largest cluster of points in the plane and extract the remaining points.
      int largestCluster = 0;
      pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr largestPlane(new pcl::PointCloud<pcl::PointXYZRGB>());
      int numClusterIndices = cluster_indices.size();
      for (int a = 0; a < numClusterIndices; ++a)
      {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);
        int numIndices = cluster_indices[a].indices.size();
        for (int f = 0; f < numIndices; ++f)
        {
          cloud_cluster->points.push_back (cloud_plane->points[cluster_indices[a].indices[f]]); //*
        }
        cloud_cluster->width = cloud_cluster->points.size ();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        if(cloud_cluster->points.size() > largestCluster)
        {
          *largestPlane = *cloud_cluster;
          largestCluster = cloud_cluster->points.size();
          *inliers = cluster_indices[a];
        }
      }

      // Uncomment to save the largest cluster in the plane to the planeClouds array and also to a point cloud file
      std::cout << "PointCloud representing the plane: " << largestPlane->points.size () << " data points." << std::endl;
      planeClouds[i] = largestPlane;
      std::stringstream ss;
      ss << "cloud_plane_" << i << ".pcd";
      writer.write<pcl::PointXYZRGB> (ss.str (), *largestPlane, false);

      // Find a moment of inertia bounding box of the plane.
      MOIBoundingBox MOIBB(planeClouds[i]);

      pcl::PointXYZRGB center = MOIBB.position_OBB;
      Eigen::VectorXf sidePlane1 = MOIBB.sidePlane1;
      Eigen::VectorXf sidePlane2 = MOIBB.sidePlane2;
      Eigen::VectorXf sidePlane3 = MOIBB.sidePlane3;
      Eigen::VectorXf sidePlane4 = MOIBB.sidePlane4;

      // Determine which side is considered the inliers of the plane by flipping each sidePlane's sign
      int sidePlane1Sign, sidePlane2Sign, sidePlane3Sign, sidePlane4Sign;

      if(center.x * sidePlane1(0) + center.y * sidePlane1(1) + center.z * sidePlane1(2) + sidePlane1(3) > 0)
      {
        sidePlane1Sign = 1;
      }
      else
      {
        sidePlane1Sign = -1;
      }

      if(center.x * sidePlane2(0) + center.y * sidePlane2(1) + center.z * sidePlane2(2) + sidePlane2(3) > 0)
      {
        sidePlane2Sign = 1;
      }
      else
      {
        sidePlane2Sign = -1;
      }

      if(center.x * sidePlane3(0) + center.y * sidePlane3(1) + center.z * sidePlane3(2) + sidePlane3(3) > 0)
      {
        sidePlane3Sign = 1;
      }
      else
      {
        sidePlane3Sign = -1;
      }

      if(center.x * sidePlane4(0) + center.y * sidePlane4(1) + center.z * sidePlane4(2) + sidePlane4(3) > 0)
      {
        sidePlane4Sign = 1;
      }
      else
      {
        sidePlane4Sign = -1;
      }

      float averageRed = 0;
      float averageGreen = 0;
      float averageBlue = 0;
      int numPlanePoints = planeClouds[i]->points.size();

      for(int h = 0; h < numPlanePoints; h++)
      {
        averageRed += planeClouds[i]->points[h].r; 
        averageGreen += planeClouds[i]->points[h].g; 
        averageBlue += planeClouds[i]->points[h].b; 
      }

      std::cout << "avgRed =" << rint(averageRed/numPlanePoints) << " avgGreen =" << rint(averageGreen/numPlanePoints) << " avgBlue =" << rint(averageBlue/numPlanePoints) << std::endl; 

      /*// Flann KdTree creation
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr flannPC (new pcl::PointCloud<pcl::PointXYZRGB>);
      *flannPC = *largestPlane;
      pcl::KdTreeFLANN<pcl::PointXYZRGB> flannTree;
      flannTree.setInputCloud(planeClouds[i]);

      // Neighbors within radius search
      std::vector<int> pointIdxRadiusSearch;
      std::vector<float> pointRadiusSquaredDistance;

      float largerRadius = 0.075; //Find neighbors around the center within a 7.5cm radius
      float radius = 0.05; //Find neighbors around the randomly selected points within a 5cm radius

      blur bl;
      bl.setCenter(center);
      bl.setRadius(radius);
      bl.setLargeRadius(largerRadius);

      if ( flannTree.radiusSearch (center, largerRadius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 )
      {
        std::vector<pcl::PointXYZRGB> blurPoints;
        std::vector<float> pointDistancesFromCenter;
        for (int z = 0; z < 9; z++)
        {
          int r = rand() % pointIdxRadiusSearch.size ();
          std::vector<int> currentPointRadiusSearch;
          std::vector<float> currentPointRadiusSquaredDistance;
          float averageRed = 0;
          float averageGreen = 0;
          float averageBlue = 0;
          if ( flannTree.radiusSearch (flannPC->points[ pointIdxRadiusSearch[r] ], radius, currentPointRadiusSearch, currentPointRadiusSquaredDistance) > 0 )
          {
            for (size_t i = 0; i < currentPointRadiusSearch.size (); ++i)
            {          
              averageRed = (averageRed*i + flannPC->points[ currentPointRadiusSearch[i] ].r)/(i+1); 
              averageGreen = (averageGreen*i + flannPC->points[ currentPointRadiusSearch[i] ].g)/(i+1);
              averageBlue = (averageBlue*i + flannPC->points[ currentPointRadiusSearch[i] ].b)/(i+1);
            }
          }
          pcl::PointXYZRGB blob;
          blob.x = flannPC->points[ pointIdxRadiusSearch[r] ].x;
          blob.y = flannPC->points[ pointIdxRadiusSearch[r] ].y;
          blob.z = flannPC->points[ pointIdxRadiusSearch[r] ].z;
          blob.r = rint(averageRed);
          blob.b = rint(averageBlue);
          blob.g = rint(averageGreen);

          std::cout << currentPointRadiusSearch.size ()
                << " neighbors within radius search at (" << blob.x 
                << " " << blob.y 
                << " " << blob.z
                << ") with radius=" << radius << std::endl;

          blurPoints.push_back(blob);
          pointDistancesFromCenter.push_back(sqrt(pointRadiusSquaredDistance[r]));
        }
        bl.setBlurredPoints(blurPoints);
        bl.setDistancesFromCenter(pointDistancesFromCenter);
        bls.push_back(bl);
      }*/

      /*pcl::PointCloud<pcl::PointXYZRGB>::Ptr model (new pcl::PointCloud<pcl::PointXYZRGB> ());
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_keypoints (new pcl::PointCloud<pcl::PointXYZRGB> ());
      pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());

      *model = *planeClouds[i];

      pcl::PointCloud<pcl::PointXYZRGB>::Ptr planeCopy (new pcl::PointCloud<pcl::PointXYZRGB> ());
      *planeCopy = *planeClouds[i];
      double model_resolution;
      int n_points = 0;
      int nres;
      std::vector<int> indices (2);
      std::vector<float> sqr_distances (2);
      pcl::search::KdTree<pcl::PointXYZRGB> kdtrees;
      kdtrees.setInputCloud(planeCopy);

      for(size_t i = 0; i < planeCopy->size (); ++i)
      {
        if (! pcl_isfinite ((*planeCopy)[i].x))
        {
          continue;
        }
        //Considering the second neighbor since the first is the point itself.
        nres = kdtrees.nearestKSearch (i, 2, indices, sqr_distances);
        if (nres == 2)
        {
          model_resolution += sqrt(sqr_distances[1]);
          ++n_points;
        }
      }
      if (n_points != 0)
      {
        model_resolution /= n_points;
      }

      pcl::ISSKeyPoint3D<pcl::PointXYZRGB, pcl::PointXYZRGB> iss_detector;

      iss_detector.setSearchMethod(tree);
      iss_detector.setSalientRadius(6*model_resolution);
      iss_detector.setNonMaxRadius(4*model_resolution);
      iss_detector.setThreshold21(0.975);
      iss_detector.setThreshold32(0.975);
      iss_detector.setMitNeighbors (5);
      iss_detector.setNumberOfThreads (4);
      iss_detector.setInputCloud (model);
      iss_detector.compute (*model_keypoints);*/

      //Extracting the current plane from the overall pointcloud
      extract.setNegative (true);
      extract.filter(*cloud_filter);
      *cloud_f = *cloud_filter;

      int numpoints = cloud_f->points.size();
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr planeObjects (new pcl::PointCloud<pcl::PointXYZRGB>);
      *planeObjects = *planeClouds[i];

      for (int n = 0; n < numpoints; n++)
      {
        float x = cloud_f->points[n].x;
        float y = cloud_f->points[n].y;
        float z = cloud_f->points[n].z;
        if((x * sidePlane1(0) + y * sidePlane1(1) + 
          z * sidePlane1(2) + sidePlane1(3)) * sidePlane1Sign > 0 && 
           (x * sidePlane2(0) + y * sidePlane2(1) + 
          z * sidePlane2(2) + sidePlane2(3)) * sidePlane2Sign > 0 &&
           (x * sidePlane3(0) + y * sidePlane3(1) + 
          z * sidePlane3(2) + sidePlane3(3)) * sidePlane3Sign > 0 &&
           (x * sidePlane4(0) + y * sidePlane4(1) + 
          z * sidePlane4(2) + sidePlane4(3)) * sidePlane4Sign > 0 &&
          modelC[i]->values[0]*x + modelC[i]->values[1]*y + modelC[i]->values[2]*z + modelC[i]->values[3] < 1)
        {
          outputCloud->points.push_back(cloud_f->points[n]);
          planeObjects->points.push_back(cloud_f->points[n]);
        }
      }

      // Creating the KdTree object for the search method of the extraction
      pcl::search::KdTree<pcl::PointXYZRGB>::Ptr ktree (new pcl::search::KdTree<pcl::PointXYZRGB>);
      ktree->setInputCloud (planeObjects);

      std::vector<pcl::PointIndices> clusterIndices;
      pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ece;
      ece.setClusterTolerance (0.02); // 2cm
      ece.setMinClusterSize (100);
      ece.setMaxClusterSize (planeObjects->points.size());
      ece.setSearchMethod (ktree);
      ece.setInputCloud (planeObjects);
      ece.extract (clusterIndices);

      int j = 0;
      for (std::vector<pcl::PointIndices>::const_iterator it = clusterIndices.begin (); it != clusterIndices.end (); ++it)
      {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
        cloud_cluster->points.push_back (planeObjects->points[*pit]); //*
        cloud_cluster->width = cloud_cluster->points.size ();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        // Uncomment to save the current plane_object to a point cloud file
        /*std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;
        std::stringstream ss;
        ss << "plane_" << i << "_object_" << j << ".pcd";
        writer.write<pcl::PointXYZRGB> (ss.str (), *cloud_cluster, false); //**/
        j++;
      }
    }
    catch (std::exception& e)
    {
      break;
    }
    i++;
  }

  /*pcl::PCDReader libraryReader;
  pcl::PCLPointCloud2 lib;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudLibrary (new pcl::PointCloud<pcl::PointXYZRGB>);
  reader.read ("cloud_library.pcd", lib);
  pcl::fromPCLPointCloud2(lib,*cloudLibrary);

  // Create the container for the different colored XYZRGB Points
  std::vector<std::vector<pcl::PointXYZRGB> > libraryOfBlurs;
  for(int r = 0; r < 128; r++)
  {
    for(int g = 0; g < 128; g++)
    {
      for(int b = 0; b < 128; b++)
      {
        std::vector<pcl::PointXYZRGB> rgb;
        libraryOfBlurs.push_back(rgb);
      }
    }
  }

  int numLibPoints = cloud_library->points.size();
  for(int y = 0; y < numLibPoints; y++)
  {
    int red = cloud_library->points[y].r;
    int green = cloud_library->points[y].g;
    int blue = cloud_library->points[y].b;
    int j = (red/2)*16384 + (green/2)*128 + (blue/2);
    libraryOfBlurs[j].push_back(cloud_library->points[y]);
  }

  std::vector<std::vector<pcl::PointXYZRGB> > potentialPointsForBlurs;
  // For each of the planes and blurred points, try to do a search to find the corresponding points
  for(int m = 0; m < i; m++)
  {
    blur blu = bls[m];
    std::vector<pcl::PointXYZRGB> potentialPoints;
    std::vector<pcl::PointXYZRGB> blurPoints = blu.getBlurredPoints();
    for (int x = 0; x < 9; x++)
    {
      int n = (blurPoints[x].r/2)*16384 + (blurPoints[x].g/2)*128 + (blurPoints[x].b/2);
      potentialPoints.insert(potentialPoints.end(),libraryOfBlurs[n].begin(), libraryOfBlurs[n].end);
    }
    potentialPointsForBlurs.push_back(potentialPoints);
  }*/

  /*
  // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
  tree->setInputCloud (outputCloud);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
  ec.setClusterTolerance (0.02); // 2cm
  ec.setMinClusterSize (100);
  ec.setMaxClusterSize (25000);
  ec.setSearchMethod (tree);
  ec.setInputCloud (outputCloud);
  ec.extract (cluster_indices);

  int j = 0;
  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
  {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);
    for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
    cloud_cluster->points.push_back (outputCloud->points[*pit]); //*
    cloud_cluster->width = cloud_cluster->points.size ();
    cloud_cluster->height = 1;
    cloud_cluster->is_dense = true;

    std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;
    std::stringstream ss;
    ss << "cloud_cluster_" << j << ".pcd";
    writer.write<pcl::PointXYZRGB> (ss.str (), *cloud_cluster, false); //*
    j++;
  }*/

  // Saves the pointcloud containing all of the planes to plane_cloud.pcd
  pcl::PCDWriter planeWriter;
  std::cout << "PointCloud representing the Plane Cluster: " << planeCloud->points.size () << " data points." << std::endl;
  planeCloud->width = planeCloud->points.size ();
  planeCloud->height = 1;
  planeCloud->is_dense = true;
  std::stringstream ss;
  ss << "plane_cloud.pcd";
  planeWriter.write<pcl::PointXYZRGB> (ss.str (), *planeCloud, false);

  // Create some new point clouds to hold our data
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr points1 (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled1 (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::Normal>::Ptr normals1 (new pcl::PointCloud<pcl::Normal>);
  pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints1 (new pcl::PointCloud<pcl::PointWithScale>);
  pcl::PointCloud<pcl::PointWithScale>::Ptr remainingkeypoints1 (new pcl::PointCloud<pcl::PointWithScale>);
  pcl::PointCloud<pcl::SHOT1344>::Ptr descriptors1 (new pcl::PointCloud<pcl::SHOT1344>);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr points2 (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled2 (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::Normal>::Ptr normals2 (new pcl::PointCloud<pcl::Normal>);
  pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints2 (new pcl::PointCloud<pcl::PointWithScale>);
  pcl::PointCloud<pcl::SHOT1344>::Ptr descriptors2 (new pcl::PointCloud<pcl::SHOT1344>);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr issKeypoints1 (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr issKeypoints2 (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr remainingISSKeypoints1 (new pcl::PointCloud<pcl::PointXYZRGB>);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr susanKeypoints1 (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr susanKeypoints2 (new pcl::PointCloud<pcl::PointXYZRGB>);

  *points1 = *planeCloud;
  *points2 = *model;

  // Downsample the clouds
  const float voxel_grid_leaf_size = 0.01;
  downsample (points1, voxel_grid_leaf_size, downsampled1);
  downsample (points2, voxel_grid_leaf_size, downsampled2);
  std::cout << "Done downsampling" << std::endl;

  // Compute surface normals
  const float normal_radius = 0.03;
  compute_surface_normals (downsampled1, normal_radius, normals1);
  compute_surface_normals (downsampled2, normal_radius, normals2);
  std::cout << "Done computing surface normals" << std::endl;

  // Compute keypoints
  /*const float min_scale = 0.01;
  const int nr_octaves = 5;
  const int nr_octaves_per_scale = 5;
  const float min_contrast = 10.0;
  detect_keypoints (points1, min_scale, nr_octaves, nr_octaves_per_scale, min_contrast, keypoints1);
  detect_keypoints (points2, min_scale, nr_octaves, nr_octaves_per_scale, min_contrast, keypoints2);
  std::cout << "Done detecting keypoints" << std::endl;

  visualize_keypointsScaled(points2,keypoints2);*/

  detect_SUSANkeypoints(points1, normals1, susanKeypoints1);
  detect_SUSANkeypoints(points2, normals2, susanKeypoints2);
  std::cout << "Done detecting keypoints" << std::endl;

  //visualize_keypoints (points2, susanKeypoints2);

  // Compute PFH features
  const float feature_radius = 0.08;
  compute_SHOTColorFeatures_at_keypoints (downsampled1, normals1, susanKeypoints1, feature_radius, descriptors1);
  compute_SHOTColorFeatures_at_keypoints (downsampled2, normals2, susanKeypoints2, feature_radius, descriptors2);
  /*compute_PFH_features_at_keypoints (downsampled1, normals1, keypoints1, feature_radius, descriptors1);
  compute_PFH_features_at_keypoints (downsampled2, normals2, keypoints2, feature_radius, descriptors2);
  compute_PFH_features_at_ISSkeypoints (downsampled1, normals1, susanKeypoints1, feature_radius, descriptors1);
  compute_PFH_features_at_ISSkeypoints (downsampled2, normals2, susanKeypoints2, feature_radius, descriptors2);*/
  std::cout << "Done computing features at keypoints" << std::endl;

  // Find feature correspondences
  std::vector<int> correspondences;
  std::vector<float> correspondence_scores;
  find_feature_correspondences (descriptors1, descriptors2, correspondences, correspondence_scores);
  /*pcl::registration::CorrespondenceEstimation<pcl::PointXYZRGB, pcl::PointXYZRGB, int> est;
  est.setInputSource(downsampled1);
  est.setInputTarget(downsampled2);
  pcl::Correspondences all_correspondences, remaining_correspondences;
  est.determineReciprocalCorrespondences(all_correspondences);*/
  std::cout << "Done finding feature correspondences" << std::endl;

  // Filter bad correspondences
  /*std::vector<int> remaining_correspondences;
  std::vector<float> remaining_correspondence_scores;
  filter_correspondences(keypoints1, keypoints2, correspondences, correspondence_scores, remainingkeypoints1, remaining_correspondences, remaining_correspondence_scores);*/


  /*std::vector<int> remaining_correspondences;
  std::vector<float> remaining_correspondence_scores;
  filter_ISScorrespondences(susanKeypoints1, susanKeypoints2, correspondences, correspondence_scores, remainingISSKeypoints1, remaining_correspondences, remaining_correspondence_scores);
  std::cout << "Done filtering correspondences" << std::endl;*/

  /*pcl::registration::CorrespondenceRejectorFeatures CRF;
  std::string sr = "source";
  std::string md = "model";
  CRF.setSourceFeature(descriptors1, sr);
  CRF.setTargetFeature(descriptors2, md);
  CRF.getRemainingCorrespondences(all_correspondences,remaining_correspondences);*/

  // Print out ( number of keypoints / number of points )
  std::cout << "First cloud: Found " << susanKeypoints1->size () << " keypoints "
            << "out of " << downsampled1->size () << " total points." << std::endl;
  std::cout << "Second cloud: Found " << susanKeypoints2->size () << " keypoints "
            << "out of " << downsampled2->size () << " total points." << std::endl;

  /*pcl::visualization::PCLVisualizer viz;
  viz.addPointCloud (points1, "points_left");
  viz.addPointCloud (points2, "points_right");
  viz.addCorrespondences(keypoints1, keypoints2, remaining_correspondences);
  viz.spin();*/

  // Visualize the two point clouds and their feature correspondences
  //visualize_correspondences (points1, remainingkeypoints1, points2, keypoints2, remaining_correspondences, remaining_correspondence_scores);
  visualize_ISScorrespondences (points1, susanKeypoints1, points2, susanKeypoints2, correspondences, correspondence_scores);

  return (0);
}
