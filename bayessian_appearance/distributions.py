import numpy as np
import scipy.stats as stat
import scipy as scp
import scipy.optimize as opt

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

        pca = decomp.PCA(n_components=settings.settings.pca_precision,svd_solver='full')

        pca.fit(X=data)

        self._internal_pca = pca

        self.distr = stat.multivariate_normal(mean=mean, cov=cov, allow_singular=True)

        self.mean_logN = self.distr.logpdf(self.distr.mean)

    def __call__(self, *args, **kwargs):
        x = args[0]
        return self.distr.logpdf(x)

    @staticmethod
    def calculate_median(data):



        np_med = None
        for arr in data:

            a = NormalDistribution(arr)

            arr_vals = []
            for i in range(arr.shape[0]):

                arr_vals.append(a.distr.logpdf(arr[i,:]))

            arr_vals = np.array(arr_vals)

            #get median index
            arr2 = arr_vals.copy()
            arr2.sort()
            ind_1 = arr2.shape[0]//2

            ind = np.where(  arr_vals == arr2[ind_1])


            if np_med is None:
                np_med = np.reshape(arr[ind,:],(arr.shape[1]))

            else:
                np_med = np.concatenate( (np_med,np.reshape(arr[ind,:],(arr.shape[1]))),axis=-1)



        # for arr in data:
        #
        #
        #     if np_med is None:
        #         np_med = np.median(arr, 0)
        #     else:
        #         np_med = np.concatenate((np_med,np.median(arr, 0)),axis=-1)

        return np_med


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

    _main_vecs = None

    _eig_vec = None
    _eig_val = None

    def _pinv_1d(self, v, eps=1e-5):
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

    def __init__(self, mean1, mean2, cov_all,tol =-1):
        if tol ==-1:
            tol = 0
        self._mean1 = mean1
        self._mean2 = mean2
        num_of_pts = mean1.shape[0]
        cov11 = cov_all[:num_of_pts, :num_of_pts]
        self._cov_11 = cov11

        s, u = scp.linalg.eigh(cov11, lower=True, check_finite=True)

        eps = _eigvalsh_to_eps(s)
        s[s < eps] = 0
        self._eig_vec = u
        self._eig_val = s
        ###########################
        s, u = scp.linalg.eigh(cov_all, lower=True, check_finite=True)
        s_pinv = self._pinv_1d(s, eps)

        U = np.multiply(u, np.sqrt(s_pinv))

        prec = np.dot(U, U.transpose())[:num_of_pts, num_of_pts:]

        self._prec_12 = prec
        ################ =============================================
        if tol>eps:
            eps = tol
        self._eig_val[self._eig_val < eps] = 0
        # self._l_12 = np.dot(U,U.transpose())
        self._main_vecs = self._eig_vec[:, self._eig_val > 0]
        self._l_cov = np.dot(cov11, self._prec_12)

        self._dist = stat.multivariate_normal(mean=mean1, cov=cov11, allow_singular=True)

    def __call__(self, *args, **kwargs):
        x0 = args[0]
        x1 = args[1]

        self._dist.mean = self._mean1 - np.dot(self._l_cov, x1 - self._mean2)
        return self._dist.logpdf(x0)

    def decompose_coords_to_eigcords(self, X):
        res = scp.linalg.solve(self._eig_vec, X - self._mean1)
        res = res[self._eig_val > 0]
        res[:] = 0
        fc = lambda x: np.linalg.norm(self.vector_2_points(x) - X)

        bounds = self.generate_bounds(3)
        Xpt = res.copy()
        for bd in range(1, len(bounds) + 1):
            bound = bounds[-bd]

            curr = bound[0]
            arr = []
            dt = bound[1] * 2 / 100
            cnt = 0
            while (bound[0] + dt * cnt < bound[1]):
                Xpt[-bd] = bound[0] + dt * cnt
                arr.append(fc(Xpt))
                cnt = cnt + 1

            arr = np.array(arr)
            ind = np.where(arr == min(arr))[0][0]
            Xpt[-bd] = bound[0] + ind * dt

        res = Xpt
        mimise = opt.minimize(fc, res, method='TNC', bounds=self.generate_bounds(3))
        r_x = mimise.fun
        for i in range(10):
            mimise = opt.minimize(fc, mimise.x, method='TNC', bounds=self.generate_bounds(3), options={"disp": True})
            mimise = opt.minimize(fc, mimise.x, method='Powell', bounds=self.generate_bounds(3), options={"disp": True})

            if r_x - mimise.fun < 1:
                break
            r_x = mimise.fun
        return mimise.x

    def vector_2_points(self, X):

        return self._mean1 + np.dot(self._main_vecs, X)

    def generate_bounds(self, n):
        res = []
        r_vec = self._eig_val[self._eig_val > 0]
        for i in range(self._main_vecs.shape[1]):
            m = np.sqrt(r_vec[i])
            res.append([-n * m, n * m])
        return res


class NormalConditionalBayes():
    _mean1 = None
    _mean2 = None
    _m1_m2 = None

    _cov_12 = None
    _l_12 = None
    _l_cov = None
    _dist = None

    _main_vecs = None

    _eig_vec = None
    _eig_val = None

    _pdf_prior = None

    def __init__(self, mean_all, cov_all, num_of_pts,tol = -1):
        self._mean1 = mean_all[:num_of_pts]
        self._mean2 = mean_all[num_of_pts:]
        self._cov_11 = cov_all[:num_of_pts, :num_of_pts]

        self._pdf_prior = stat.multivariate_normal(mean=self._mean1, cov=self._cov_11, allow_singular=True)
        s, u = scp.linalg.eigh(cov_all, lower=True, check_finite=True)

        eps = _eigvalsh_to_eps(s)
        s[s < eps] = 0
        s_pinv = _pinv_1d(s, eps)

        U = np.multiply(u, np.sqrt(s_pinv))
        prec_all = np.dot(U, U.transpose())
        self._eig_vec = u
        self._eig_val = s
        if tol>eps:
            eps = tol
        self._eig_val[self._eig_val < eps] = 0

        self._main_vecs = self._eig_vec[:, self._eig_val > 0]
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

    def __init__(self, mean_s_1, mean_s_2, cov_s, mean_si1, mean_si2, cov_si):
        self._mean_s2 = mean_s_2
        self.dist_s1s2 = NormalConditional(mean_s_1, mean_s_2, cov_s ,tol=0.5)
        self.dist_I_s1 = NormalConditional(mean_si1, mean_si2, cov_si,tol=0.5)
        pass

    def set_S2(self, pts):
        self._mean_s2 = np.array(pts)

    def __call__(self, *args, **kwargs):
        shape = args[0]
        intenstities = args[1]

        return self.dist_I_s1(intenstities, shape) + self.dist_s1s2(shape, self._mean_s2)

    def vector_2_points(self, X):
        return self.dist_s1s2.vector_2_points(X)

    def get_num_eigenvecs(self):

        return  (self.dist_s1s2._eig_val > 0).sum()
    def generate_bounds(self,n):

        return self.dist_s1s2.generate_bounds(n)