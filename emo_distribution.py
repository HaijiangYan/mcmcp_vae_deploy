#!/Yan/miniforge3/envs/
# -*- coding:utf-8 -*-
from numpy import random, array, argwhere, mean, sum, zeros, square, sqrt
from scipy import stats


class AffectiveFace:
    """calculate the distribution of vae's prior knowledge of the specific emotion"""

    @staticmethod
    def _read_csv2array(filepath):
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

    def __init__(self):
        self.latent_points = self._read_csv2array('./latent_point.csv')
        self.cat_center = zeros((7, 3))
        self.weight = zeros((self.latent_points.shape[0], 1))
        for category_id in range(7):
            idx = argwhere(self.latent_points[:, 3] == category_id).squeeze()
            mean_cat = mean(self.latent_points[idx, 0:3], axis=0)
            # the center of each emotion cluster
            self.cat_center[category_id, :] = mean_cat
            # the distance from each point to their center
            distance = 1./sqrt(square(self.latent_points[idx, 0]-mean_cat[0])
                               + square(self.latent_points[idx, 1]-mean_cat[1])
                               + square(self.latent_points[idx, 2]-mean_cat[2]))
            self.weight[idx, 0] = distance / sum(distance)

    def get_density(self, points, category_id):
        """weighted GMM to infer the probability density given a point"""

        gmm_density = array([0, 0], dtype='float64')
        idx = argwhere(self.latent_points[:, 3] == category_id).squeeze()

        for i in range(idx.shape[0]):
            gaus_dist = stats.multivariate_normal(mean=[self.latent_points[idx[i], 0], self.latent_points[idx[i], 1],
                                                        self.latent_points[idx[i], 2]],
                                                  cov=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            gaus_dense = gaus_dist.pdf(points)
            gmm_density += gaus_dense * self.weight[idx[i], 0]

        return gmm_density






