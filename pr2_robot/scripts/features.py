import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from pcl_helper import *

DEBUG = False

def rgb_to_hsv(rgb_list):
    rgb_normalized = [1.0*rgb_list[0]/255, 1.0*rgb_list[1]/255, 1.0*rgb_list[2]/255]
    hsv_normalized = matplotlib.colors.rgb_to_hsv([[rgb_normalized]])[0][0]
    return hsv_normalized


def show_hist_range(name, hist):
    print("{} ({}, {})".format(name, hist.min(), hist.max()))


def extract_features(cloud,
                     get_normals,
                     bins=32,
                     using_hsv=True,
                     cbins_range=(0, 256),
                     nbins_range=(-1, 1),
                     normed=True):
    chists = compute_color_histograms(cloud, using_hsv=using_hsv,
                                      bins=bins, bins_range=cbins_range,
                                      normed=normed)
    normals = get_normals(cloud)
    nhists = compute_normal_histograms(normals, bins=bins,
                                       bins_range=nbins_range, normed=normed)
    feature = np.concatenate((chists, nhists), axis=0)
    return feature


def compute_color_histograms(cloud, using_hsv=False, bins=50, bins_range=(0, 256), normed=False):

    # Compute histograms for the clusters
    point_colors_list = []

    # Step through each point in the point cloud
    for i, point in enumerate(pc2.read_points(cloud, skip_nans=True)):
        rgb_list = float_to_rgb(point[3])
        if using_hsv:
            point_colors_list.append(rgb_to_hsv(rgb_list) * 255)
        else:
            point_colors_list.append(rgb_list)

    # Populate lists with color values
    channel_1_vals = []
    channel_2_vals = []
    channel_3_vals = []

    for color in point_colors_list:
        channel_1_vals.append(color[0])
        channel_2_vals.append(color[1])
        channel_3_vals.append(color[2])

    # TODO: Compute histograms
    channel_1_hist, hist_range_1 = np.histogram(channel_1_vals, bins=bins, range=bins_range, normed=normed)
    channel_2_hist, hist_range_2 = np.histogram(channel_2_vals, bins=bins, range=bins_range, normed=normed)
    channel_3_hist, hist_range_3 = np.histogram(channel_3_vals, bins=bins, range=bins_range, normed=normed)

    if DEBUG:
        show_hist_range("channel_1_hist", hist_range_1)
        show_hist_range("channel_2_hist", hist_range_2)
        show_hist_range("channel_3_hist", hist_range_3)

    # Concatenate and normalize the histograms
    hist_feature = np.concatenate((channel_1_hist, channel_2_hist, channel_3_hist))
    normed_features = hist_feature / np.sum(hist_feature)
    return normed_features 


def compute_normal_histograms(normal_cloud, bins=50, bins_range=(-1, 1), normed=False):
    norm_x_vals = []
    norm_y_vals = []
    norm_z_vals = []

    for norm_component in pc2.read_points(normal_cloud,
                                          field_names=('normal_x', 'normal_y', 'normal_z'),
                                          skip_nans=True):
        norm_x_vals.append(norm_component[0])
        norm_y_vals.append(norm_component[1])
        norm_z_vals.append(norm_component[2])

    # Compute histograms of normal values
    norm_x_hist, hist_range_x = np.histogram(norm_x_vals, bins=bins, range=bins_range, normed=normed)
    norm_y_hist, hist_range_y = np.histogram(norm_y_vals, bins=bins, range=bins_range, normed=normed)
    norm_z_hist, hist_range_z = np.histogram(norm_z_vals, bins=bins, range=bins_range, normed=normed)

    if DEBUG:
        show_hist_range("norm_x_hist", hist_range_x)
        show_hist_range("norm_y_hist", hist_range_y)
        show_hist_range("norm_z_hist", hist_range_z)

    # Concatenate and normalize the histograms
    hist_feature = np.concatenate((norm_x_hist, norm_y_hist, norm_z_hist))
    normed_features = hist_feature / np.sum(hist_feature)

    return normed_features
