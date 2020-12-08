
import numpy as np
import scipy.stats as stat


class NormalDistribution:
    #
    # matr = None
    #
    # mean_vec = None

    mean_logN = None
    #
    # distr = None

    # def pdf(self, x):
    #     coeff = pow(2 * np.pi, len(self.mean_vec))
    #     det = np.linalg.det(self.matr)
    #     matr_rev = np.linalg.inv(self.matr)
    #     coeff = np.sqrt(det * coeff)
    #
    #     v1 = np.dot(matr_rev, x - self.mean_vec)
    #     v2 = np.dot(x - self.mean_vec, v1)
    #
    #     res = np.exp(-0.5 * v2) / coeff
    #     return res

    #rows = samples cols variables
    def __init__(self,data):
        data = np.array(data)
        mean = []
        for i in range(data.shape[1]):
            mean.append(data[:,i].mean())
        pass

        mean = np.array(mean)
        mean_vec = mean
        cov = np.cov(data.transpose())
        matr = cov

        self.distr = stat.multivariate_normal(mean=mean, cov=cov)

        self.mean_logN = self.distr.logpdf(self.distr.mean)


    def __call__(self, *args, **kwargs):

        x = args[0]
        return self.distr.logpdf(x) - self.mean_logN











