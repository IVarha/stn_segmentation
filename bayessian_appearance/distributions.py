import numpy as np
import scipy.stats as stat
import scipy as scp
import scipy.optimize as opt
import sklearn.neighbors as neib

import sklearn.covariance as rob_cov
import sklearn.decomposition as decomp
import bayessian_appearance.settings as settings

import sklearn.svm as svm


class NormalDistribution:
    #
    # matr = None
    #
    # mean_vec = None

    mean_logN = None

    _internal_pca = None

    # rows = samples
    # cols = variables
    def __init__(self, data):
        data = np.array(data)

        rob = rob_cov.EllipticEnvelope(random_state=0)

        rob.fit(X=data)

        mean = rob.location_

        cov = rob.covariance_

        pca = decomp.PCA(n_components=settings.settings.pca_precision, svd_solver='full')

        pca.fit(X=data)

        self._internal_pca = pca

        self.distr = stat.multivariate_normal(mean=mean, cov=cov, allow_singular=True)

        self.mean_logN = self.distr.logpdf(self.distr.mean)

    def __call__(self, *args, **kwargs):
        x = args[0]
        return self.distr.logpdf(x)

    @staticmethod
    def calculate_median(data):

        np_med = []
        for arr in data:

            a = NormalDistribution(arr)

            arr_vals = []
            for i in range(arr.shape[0]):
                arr_vals.append(a.distr.logpdf(arr[i, :]))

            arr_vals = np.array(arr_vals)

            # get median index
            arr2 = arr_vals.copy()
            arr2.sort()
            ind_1 = arr2.shape[0] // 2

            ind = np.where(arr_vals == arr2[ind_1])


            np_med.append(arr[ind, :])

        # for arr in data:
        #
        #
        #     if np_med is None:
        #         np_med = np.median(arr, 0)
        #     else:
        #         np_med = np.concatenate((np_med,np.median(arr, 0)),axis=-1)

        return np_med


class ProductJoined_ShInt_Distribution:

    _norm1 = None
    _norm2 = None
    _pca_main = None
    def __init__(self, data_main, data_condition, tol=-1):
        if tol == -1:
            tol = 0

        pcaS = decomp.PCA(n_components=settings.settings.pca_precision, svd_solver='full')
        pcaS.fit(data_main)


        norm1 = rob_cov.EllipticEnvelope()
        norm1.fit(data_main)

        norm2 = rob_cov.EllipticEnvelope()
        norm2.fit(data_condition)

        self._norm1 = norm1
        self._norm2 = norm2
        self._pca_main = pcaS

    def __call__(self, *args, **kwargs):
        x0 = args[0]
        x1 = args[1]



        return -(self._norm1.mahalanobis(x0.reshape((1,x0.shape[0])))[0]
                + self._norm2.mahalanobis(x1.reshape((1,x1.shape[0])))[0])

    def decompose_coords_to_eigcords(self, X):
        X1 = X.reshape((1, X.shape[0]))
        return self._pca_main.transform(X1)[0]

    def vector_2_points(self, X):

        X1 = X.reshape((1, X.shape[0]))
        return self._pca_main.inverse_transform(X1)[0]

    def generate_bounds(self, n):
        res = []

        expl_var = self._pca_main.explained_variance_
        for i in range(expl_var.shape[0]):
            m = np.sqrt(expl_var[i])
            res.append([-n * m, n * m])
        return res

    pass

def _eigvalsh_to_eps(spectrum, cond=None, rcond=None):
    """
    Determine which eigenvalues are "small" given the spectrum.

    This is for compatibility across various linear algebra functions
    that should agree about whether or not a Hermitian matrix is numerically
    singular and what is its numerical matrix rank.
    This is designed to be compatible with scipy.linalg.pinvh.

    Parameters
    ----------
    spectrum : 1d ndarray
        Array of eigenvalues of a Hermitian matrix.
    cond, rcond : float, optional
        Cutoff for small eigenvalues.
        Singular values smaller than rcond * largest_eigenvalue are
        considered zero.
        If None or -1, suitable machine precision is used.

    Returns
    -------
    eps : float
        Magnitude cutoff for numerical negligibility.

    """
    if rcond is not None:
        cond = rcond
    if cond in [None, -1]:
        t = spectrum.dtype.char.lower()
        factor = {'f': 1E3, 'd': 1E6}
        cond = factor[t] * np.finfo(t).eps
    eps = cond * np.max(abs(spectrum))
    return eps


def _pinv_1d(v, eps=1e-5):
    """
    A helper function for computing the pseudoinverse.

    Parameters
    ----------
    v : iterable of numbers
        This may be thought of as a vector of eigenvalues or singular values.
    eps : float
        Values with magnitude no greater than eps are considered negligible.

    Returns
    -------
    v_pinv : 1d float ndarray
        A vector of pseudo-inverted numbers.

    """
    return np.array([0 if abs(x) <= eps else 1 / x for x in v], dtype=float)


class NormalConditional:
    _mean1 = None
    _mean2 = None
    _m1_m2 = None

    _cov_12 = None
    _l_12 = None
    _l_cov = None
    _dist = None
    _gaus_kernel = None
    _main_vecs = None

    _eig_vec = None
    _eig_val = None

    _pca = None


    def __init__(self, data_main, data_condition, tol=-1):
        if tol == -1:
            tol = 0

        pca = decomp.PCA(n_components=settings.settings.pca_precision, svd_solver='full')
        pca.fit(np.concatenate((data_main, data_condition), axis=-1))

        cov_all = pca.get_covariance()
        pca2 = decomp.PCA(n_components=settings.settings.pca_precision, svd_solver='full') #used only for recalculate
        pca2.fit(data_main)
        self._pca = pca2
        mean1 = pca.mean_[:data_main.shape[1]]
        mean2 = pca.mean_[data_main.shape[1]:]
        self._mean1 = mean1
        self._mean2 = mean2
        num_of_pts = mean1.shape[0]

        cov11 = cov_all[:num_of_pts, :num_of_pts]
        self._cov_11 = cov11

        # s, u = scp.linalg.eigh(cov11, lower=True, check_finite=True)

        # eps = _eigvalsh_to_eps(s)
        # s[s < eps] = 0
        # self._eig_vec = u
        # self._eig_val = s
        # ###########################
        # s, u = scp.linalg.eigh(cov_all, lower=True, check_finite=True)
        # s_pinv = self._pinv_1d(s, eps)
        #
        # U = np.multiply(u, np.sqrt(s_pinv))
        #
        # prec = np.dot(U, U.transpose())[:num_of_pts, num_of_pts:]
        #
        # self._prec_12 = prec
        # ################ =============================================
        # if tol>eps:
        #     eps = tol
        # self._eig_val[self._eig_val < eps] = 0
        # # self._l_12 = np.dot(U,U.transpose())
        # self._main_vecs = self._eig_vec[:, self._eig_val > 0]
        # self._l_cov = np.dot(cov11, self._prec_12)
        self._prec_12 = pca.get_precision()[:num_of_pts, num_of_pts:]
        self._l_cov = np.dot(cov11, self._prec_12)
        self._dist = stat.multivariate_normal(mean=mean1, cov=cov11, allow_singular=True)
        #self._gaus_kernel = neib.KernelDensity()
        #self._gaus_kernel.fit(X=np.concatenate((data_main, data_condition), axis=1))

    def __call__(self, *args, **kwargs):
        x0 = args[0]
        x1 = args[1]

        a = np.concatenate((x0,x1),axis=-1)
        #return self._gaus_kernel.score(a.reshape((1,a.shape[0])))
        self._dist.mean = self._mean1 - np.dot(self._l_cov, x1 - self._mean2)
        return self._dist.logpdf(x0)

    def get_mean_conditional(self,shape2):
        return self._mean1 - np.dot(self._l_cov, shape2 - self._mean2)
    def decompose_coords_to_eigcords(self, X):
        X1 = X.reshape((1, X.shape[0]))
        return self._pca.transform(X1)[0]

    def vector_2_points(self, X):

        X1 = X.reshape((1, X.shape[0]))
        return self._pca.inverse_transform(X1)[0]

    def generate_bounds(self, n):
        res = []

        expl_var = self._pca.explained_variance_
        for i in range(expl_var.shape[0]):
            m = np.sqrt(expl_var[i])
            res.append([-n * m, n * m])
        return res


class NormalConditionalBayes():
    _mean1 = None
    _mean2 = None
    _m1_m2 = None
    _pca1 = None

    _cov_12 = None
    _l_12 = None
    _l_cov = None
    _dist = None

    _main_vecs = None

    _eig_vec = None
    _eig_val = None

    _pdf_prior = None

    def __init__(self, data_main, data_condition):

        pca = decomp.PCA(n_components=settings.settings.pca_precision, svd_solver='full')

        pca.fit(np.concatenate((data_main, data_condition), axis=-1))
        self._mean1 = pca.mean_[:data_main.shape[1]]
        self._mean2 = pca.mean_[data_main.shape[1]:]
        cov_all = pca.get_covariance()
        num_of_pts = self._mean1.shape[0]
        self._cov_11 = cov_all[:num_of_pts, :num_of_pts]

        self._pdf_prior = stat.multivariate_normal(mean=self._mean1, cov=self._cov_11, allow_singular=True)


        prec_all = pca.get_precision()


        #self._main_vecs = self._eig_vec[:, self._eig_val > 0]
        self._l_cov = np.dot(cov_all[num_of_pts:, num_of_pts:], prec_all[num_of_pts:, :num_of_pts])

        self._dist = stat.multivariate_normal(mean=self._mean2, cov=prec_all[num_of_pts:, num_of_pts:],
                                              allow_singular=True)

    def __call__(self, *args, **kwargs):
        x0 = args[0]
        x1 = args[1]

        self._dist.mean = self._mean2 - np.dot(self._l_cov, x0 - self._mean1)
        return self._dist.logpdf(x1) + self._pdf_prior.logpdf(x0)

    def vector_2_points(self, X):
        return self._mean1 + np.dot(self._main_vecs, X)


class JointDependentDistribution:
    dist_I_s1 = None
    dist_s1s2 = None
    _mean_s2 = None
    _pca_main = None

    def __init__(self,data_s1,data_s2,data_I1):
        self._mean_s2 = None

        #########
        pcaS = decomp.PCA(n_components=settings.settings.pca_precision, svd_solver='full')
        pcaS.fit(data_s1)


        norm1 = rob_cov.EllipticEnvelope()
        norm1.fit(np.concatenate((data_s1,data_s2),axis=-1))


        #self.dist_s1s2 = norm1

        self._pca_main = pcaS

        ###########
        self.dist_s1s2 = NormalConditional(data_main=data_s1, data_condition=data_s2)
        #self.dist_I_s1 = NormalConditional(data_main=data_I1,data_condition=data_s1)
        a = rob_cov.EllipticEnvelope()
        a.fit(data_I1)
        self.dist_I_s1 = a
        pass

    def set_S2(self, pts):
        self._mean_s2 = np.array(pts)

    def __call__(self, *args, **kwargs):
        shape = args[0]

        intenstities = args[1]

        shape = np.concatenate((shape,self._mean_s2),axis=-1)
        shape = shape.reshape((1,shape.shape[0]))

        return -(self.dist_I_s1.mahalanobis(intenstities.reshape((1,intenstities.shape[0])))[0]
               + self.dist_s1s2(args[0],self._mean_s2))

    def get_mean_conditional(self ):
        return self.dist_s1s2.get_mean_conditional(self._mean_s2)


    # def vector_2_points(self, X):
    #     return self.dist_s1s2.vector_2_points(X)
    #
    # def get_num_eigenvecs(self):
    #     return self.dist_s1s2._pca.components_.shape[0]
    #
    # def generate_bounds(self, n):
    #     return self.dist_s1s2.generate_bounds(n)
    def decompose_coords_to_eigcords(self, X):
        X1 = X.reshape((1, X.shape[0]))
        return self._pca_main.transform(X1)[0]

    def vector_2_points(self, X):

        X1 = X.reshape((1, X.shape[0]))
        return self._pca_main.inverse_transform(X1)[0]

    def generate_bounds(self, n):
        res = []

        expl_var = self._pca_main.explained_variance_
        for i in range(expl_var.shape[0]):
            m = np.sqrt(expl_var[i])
            res.append([-n * m, n * m])
        return res
    def get_num_eigenvecs(self):
        return self._pca_main.components_.shape[0]