#!/usr/bin/env python
import numpy as np
import pickle
import sys
import rospy

from sensor_stick.pcl_helper import *
from sensor_stick.training_helper import spawn_model
from sensor_stick.training_helper import delete_model
from sensor_stick.training_helper import initial_setup
from sensor_stick.training_helper import capture_sample
from sensor_stick.features import extract_features
from sensor_stick.srv import GetNormals
from geometry_msgs.msg import Pose
from sensor_msgs.msg import PointCloud2


def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

if __name__ == '__main__':
    arguments = sys.argv
    if len(arguments) > 1:
        nb_point = int(arguments[1])
    else:
        nb_point = 5

    rospy.init_node('capture_node')

    models = [
        "biscuits",
        "soap",
        "soap2",
        "book",
        "glue",
        "sticky_notes",
        "snacks",
        "eraser"
    ]

    # Disable gravity and delete the ground plane
    initial_setup()
    labeled_features = []

    for model_name in models:
        spawn_model(model_name)
        print(model_name)
        for i in range(nb_point):
            # make five attempts to get a valid a point cloud then give up
            sample_was_good = False
            try_count = 0
            while not sample_was_good and try_count < 5:
                sample_cloud = capture_sample()
                sample_cloud_arr = ros_to_pcl(sample_cloud).to_array()

                # Check for invalid clouds.
                if sample_cloud_arr.shape[0] == 0:
                    print('Invalid cloud detected')
                    try_count += 1
                else:
                    sample_was_good = True
            # Save point cloud
            # Extract histogram features
            feature = extract_features(sample_cloud, get_normals)
            labeled_features.append([feature, model_name])

        delete_model()

    pickle.dump(labeled_features, open('training_set.sav', 'wb'))

