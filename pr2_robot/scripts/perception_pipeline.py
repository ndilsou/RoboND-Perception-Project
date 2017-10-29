#!/usr/bin/env python

# Import modules
import os
import sys
import argparse
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import extract_features
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import pcl

import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from std_srvs.srv import Empty
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Constants
LEAF_SIZE = 0.005
Z_AXIS_MIN = 0.605
Z_AXIS_MAX = 1.2
Y_AXIS_MIN = -0.4
Y_AXIS_MAX = 0.4
MAX_DISTANCE = 0.01
# Outlier removal params
MEAN_K = 200
STD_DEV = 1
# Cluster params
TOLERANCE = 0.05
MIN_SIZE = 50
MAX_SIZE = 1500

# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster


# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(current_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["current_scene_num"] = current_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict


# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)


def set_position_from_list(pose, position):
    """Helper method to set the x, y , z field of a Pose.Position object.
    """
    pose.position.x = position[0]
    pose.position.y = position[1]
    pose.position.z = position[2]


def pcl_callback(pcl_msg):
    # PART 1 : FILTERING AND RANSAC

    # Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)

    print("PassThrough Filter")
    # PassThrough Filter
    passthrough = cloud.make_passthrough_filter()
    filter_axis = "z"
    passthrough.set_filter_field_name(filter_axis)
    passthrough.set_filter_limits(Z_AXIS_MIN, Z_AXIS_MAX)
    cloud_filtered = passthrough.filter()
    
    passthrough = cloud_filtered.make_passthrough_filter()
    filter_axis = "y"
    passthrough.set_filter_field_name(filter_axis)
    passthrough.set_filter_limits(Y_AXIS_MIN, Y_AXIS_MAX)
    cloud_filtered = passthrough.filter()

    print("Outlier removal")
    # Outlier removal
    outlier_filter = cloud_filtered.make_statistical_outlier_filter()
    outlier_filter.set_mean_k(MEAN_K)
    outlier_filter.set_std_dev_mul_thresh(STD_DEV)
    cloud_filtered = outlier_filter.filter()

    print("Voxel Grid Downsampling")
    # Voxel Grid Downsampling
    vox = cloud_filtered.make_voxel_grid_filter()
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_filtered = vox.filter()

    print("RANSAC Plane Segmentation")
    # RANSAC Plane Segmentation
    seg = cloud_filtered.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(MAX_DISTANCE)
    inliers, coefficient = seg.segment()

    print("Extract inliers and outliers")
    # Extract inliers and outliers
    extracted_inliers = cloud_filtered.extract(inliers, negative=False)
    extracted_outliers = cloud_filtered.extract(inliers, negative=True)

    # PART 2 : CLUSTERING FOR SEGMENTATION

    print("Euclidean Clustering")
    # Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(extracted_outliers)
    tree = white_cloud.make_kdtree()

    print("Create Cluster-Mask Point Cloud to visualize each cluster separately")
    # Create Cluster-Mask Point Cloud to visualize each cluster separately
    ec = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold
    # as well as minimum and maximum cluster size (in points)
    ec.set_ClusterTolerance(TOLERANCE)
    ec.set_MinClusterSize(MIN_SIZE)
    ec.set_MaxClusterSize(MAX_SIZE)

    print("Search the k-d tree for clusters")
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()
    # Assign a color corresponding to each segmented object in scene
    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                             white_cloud[indice][1],
                                             white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    # Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)

    # Convert PCL data to ROS messages
    ros_cloud_table = pcl_to_ros(extracted_inliers)
    ros_cloud_objects = pcl_to_ros(extracted_outliers)

    # Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cluster_cloud)

    # PART 3 CLASSIFICATION

    print("Classify the clusters!")
    # Classify the clusters!
    separate_objects_cloud = {"table": extracted_inliers} # Stores the point cloud before sending them to the Pr2Mover.
    detected_objects_labels = []
    detected_objects = []
    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster from the extracted outliers (cloud_objects)
        pcl_cluster = extracted_outliers.extract(pts_list)
        # Tconvert the cluster from pcl to ROS using helper function
        ros_cluster = pcl_to_ros(pcl_cluster)

        # Extract histogram features
        feature = extract_features(ros_cluster, get_normals)

        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        X = feature.reshape(1, -1)
        if scaler:
            X = scaler.transform(X)
        prediction = clf.predict(X)
        label = encoder.inverse_transform(prediction)[0]
        separate_objects_cloud[label] = pcl_cluster
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label, label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)

    if len(separate_objects_cloud) > 1:
        if pickup:  # perform pick and place.
            try:
                pr2_mover.move_next(separate_objects_cloud)
            except rospy.ROSInterruptException:
                pass
        else:  # just output to result file.
            send_all_to_yaml(separate_objects_cloud)


def get_cloud_centroid(cloud):
    """Identify the centroid of a point cloud as the mean over each dimension.
    """
    arr_cloud = cloud.to_array()
    return np.mean(arr_cloud, axis=0)[:3]


def aggregate_clouds(clouds):
    """ Forms a single point cloud object from an iterable of clouds.
    """
    cloud = pcl.PointCloud_PointXYZRGB()
    objects = tuple(c.to_array() for c in clouds)
    arr = np.concatenate(objects)
    cloud.from_array(arr)
    return cloud


def get_output(table_object, object_dict, dropbox_map):
    """
    Collects all the information required to send a pick and place request for a single object.

    This function will obtain the object label, arm to use, and poses for the object and target box.
    """
    object_name = String()
    object_name.data = table_object["name"]
    print("object_name")
    print(object_name)
    arm_name = String()
    arm_name.data = dropbox_map[table_object['group']]['name']
    print("which arm")
    print(arm_name)
    # Get the PointCloud for a given object and obtain it's centroid
    pcl_cloud = object_dict[table_object["name"]]
    centroid = get_cloud_centroid(pcl_cloud)
    pick_pose = Pose()
    set_position_from_list(pick_pose, centroid.tolist())
    print("pick_pose")
    print(pick_pose)
    # Create 'place_pose' for the object
    place_pose = Pose()
    # Assign the arm to be used for pick_place
    set_position_from_list(place_pose, dropbox_map[table_object['group']]['position'])
    print("place_pose")
    print(place_pose)
    return object_name, arm_name, pick_pose, place_pose


def send_all_to_yaml(object_dict):
    """Push all the objects recognized in the scene to a yaml file.
    """
    object_list_param = rospy.get_param('/object_list')
    dropbox_param = rospy.get_param("/dropbox")
    dropbox_map = {dropbox['group']: dropbox for dropbox in dropbox_param}
    output_list = []
    print(object_list_param)
    for i, table_object in enumerate(object_list_param):
        object_name, arm_name, pick_pose, place_pose = get_output(table_object, object_dict, dropbox_map)

        # Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        output = make_yaml_dict(current_scene_num, arm_name, object_name, pick_pose, place_pose)
        output_list.append(output)

    # Output your request parameters into output yaml file
    send_to_yaml("output/output_{}.yaml".format(current_scene), output_list)


def clear_octomap():
    """Clears the content of the collision map.
    """
    rospy.wait_for_service('/clear_octomap')
    try:
        clear_octomap_srv = rospy.ServiceProxy('/clear_octomap', Empty)
        clear_octomap_srv()
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)


def scan():
    """Rotates PR2 in place to capture side tables for the collision map
    The initial plan was to send the rotation command and wait for the robot to reach the
    position by polling the /pr2/joint_states topic and looking up the value for the base joint
    until the target is reached.
    """
    base_joint_controller_pub.publish(Float64(-np.pi / 2))
    #rospy.wait_for_message("/pr2/joint_states")
    base_joint_controller_pub.publish(Float64(np.pi / 2))

    # return to normal position.
    base_joint_controller_pub.publish(Float64(0))


def pickup(object_name, arm_name, pick_pose, place_pose):
    """Send a pickup request to the pick_place_routine service.
    """
    print("sending pickup")
    rospy.wait_for_service('pick_place_routine')
    try:
        pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

        # Insert your message variables to be sent as a service request
        resp = pick_place_routine(current_scene_num, object_name, arm_name, pick_pose, place_pose)

        print("Response: ", resp.success)
        pass
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)


class Pr2Mover:
    """
    Facilitates sending the pick & place request to the robot by keeping 
    track of which items are left to pickup.
    """
    def __init__(self):
        """Load the object list and drop boxes position from param server."""
        self.object_list_param = rospy.get_param('/object_list')
        self.object_list_queue = list(self.object_list_param)
        dropbox_param = rospy.get_param("/dropbox")
        self.dropbox_map = {dropbox['group']: dropbox for dropbox in dropbox_param}

    def move_next(self, object_dict):
        """Update the collision map and send a request to pick up the next object in the queue for this world.
        """
        if len(self.object_list_queue) > 0:
            table_object = self.object_list_queue.pop(0)

            print("moving {} next".format(table_object['name']))
            clear_octomap()

            # update octomap
            collision_dict = object_dict.copy()
            collision_dict.pop(table_object['name'])
            collision_cloud = aggregate_clouds(object_dict.values())
            collision_avoidance_pub.publish(pcl_to_ros(collision_cloud))
            scan()
            # retrieve object info
            object_name, arm_name, pick_pose, place_pose = get_output(table_object,
                                                                      object_dict,
                                                                      self.dropbox_map)
            # send pickup request.
            pickup(object_name, arm_name, pick_pose, place_pose)
        else:
            print("no remaining object to pickup")


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("scene", type=int, help="index of the world scene currently in use in Gazebo")
    argument_parser.add_argument("model", type=str, help="name of the model to use.")
    argument_parser.add_argument("--pickup", dest="pickup", default=False, action="store_true",
                                 help="flag indicating if the robot should pickup or just write output to yaml")

    arguments = argument_parser.parse_args()
    current_scene = arguments.scene
    current_scene_num = Int32(current_scene)
    pickup = arguments.pickup
    model_name = arguments.model

    # ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", PointCloud2, pcl_callback, queue_size=1)

    # Create Publishers
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)
    collision_avoidance_pub = rospy.Publisher("/pr2/3d_map/points", PointCloud2, queue_size=1)
    base_joint_controller_pub = rospy.Publisher("/pr2/world_joint_controller/command", Float64, queue_size=1)

    pr2_mover = Pr2Mover()
    # Load Model From disk
    model = pickle.load(open('estimators/model_{}.sav'.format(model_name), 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model["scaler"]
    # Initialize color_list
    get_color_list.color_list = []

    print("starting.")
    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
