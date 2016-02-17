# PEBL

## Table of Contents

* [Description](README.md#description)
* [Requirements](README.md#requirements)
* [Maintainers](README.md#maintainers)
* [Installation](README.md#installation)
  * [Linux](README.md#linux)

## Description

Plane Extraction Based Localization

This application:
* Reads in two point clouds, the first of which is a smaller RGB point cloud within a space and the second which is a model point cloud.
* Performs Statistical Outlier Removal on both point clouds
* Uses a voxelgrid to filter each point cloud
* Identify planes from each cloud and then identify clusters on the planar surface that are either extending or regressing into the plane
* Segment the clusters and identify keypoints
* Generate features at those keypoints including using the normals and the average color of the plane the object is on
* Match keypoints between the model and the test point cloud and display their correspondences

Missing features/TODO:
* Integrate into ROS (Currently standalone so that it is faster to test with the same dataset)
* Output the location in the model space as determined by the strongest correspondences between features of keypoints (Very simple to implement, only need to pinpoint the location and then scale the distances to the model space)
* Implement rotating the head and tilting to provide more accuracy with initial localization (HERB's head is not great and will be upgraded soon)
* Update periodically relocalization within the model when no other tasks are running to increase accuracy after the robot or camera has moved.  

## Requirements

### Operating system requirements

* Ubuntu 14.04 or newer

### Additional Requirements

* Cmake version 2.8 or higher
* PCL v1.2 or higher

## Maintainers

* Kevin Zhang <kevinleezhang@gmail.com> <kevinleezhang@cmu.edu>

## Installation

### Linux

* Download PEBL source

    ```
git clone https://github.com/firephinx/PEBL.git
cd PEBL
cmake CMakeLists.txt
make
```
