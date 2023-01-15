#!/Yan/miniforge3/envs/
# -*- coding:utf-8 -*-

from numpy import random, shape, array
from scipy import stats


def read_csv2array(filepath):
    """read label.csv and return a list"""

    file = open(filepath, 'r', encoding="utf-8")
    context = file.read()  # as str
    list_result = context.split("\n")[1:-1]
    length = len(list_result)
    points = random.normal(0, 1, [length, 4])
    for i in range(length):
        for j in range(4):
            points[i, j] = float(list_result[i].split(",")[j+1])
    file.close()  # file must be closed after manipulating
    return points


def get_distribution(points, cat):
    latent_points = read_csv2array('./latent_point.csv')
    mix_dense = array([0, 0], dtype='float64')
    count = 0
    for i in range(shape(latent_points)[0]):
        if latent_points[i, 3] == cat:
            gaus_dist = stats.multivariate_normal(mean=[latent_points[i, 0], latent_points[i, 1], latent_points[i, 2]],
                                                  cov=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            gaus_dense = gaus_dist.pdf(points)
            mix_dense += gaus_dense
            count += 1
    return mix_dense / count






