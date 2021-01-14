
import numpy as np
import scipy.stats as stat
import scipy as scp
import scipy.optimize as opt


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

    # rows = samples
    # cols = variables
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

        self.distr = stat.multivariate_normal(mean=mean, cov=cov,allow_singular=True)

        self.mean_logN = self.distr.logpdf(self.distr.mean)


    def __call__(self, *args, **kwargs):

        x = args[0]
        return self.distr.logpdf(x) - self.mean_logN






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

    def _eigvalsh_to_eps(self,spectrum, cond=None, rcond=None):
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

    def __init__(self,mean1,mean2, cov11,prec12):

        self._mean1 = mean1
        self._mean2 = mean2
        self._prec_12 = prec12
        self._cov_11 = cov11

        s,u = scp.linalg.eigh(cov11, lower=True, check_finite=True)

        eps = self._eigvalsh_to_eps(s)
        s[s<eps] = 0
        #s_pinv = self._pinv_1d(s, eps)

        #U = np.multiply(u, np.sqrt(s_pinv))
        #self._l_12 = np.dot(U,U.transpose())
        self._eig_vec = u
        self._eig_val = s
        self._main_vecs = self._eig_vec[:,self._eig_val>0]
        self._l_cov = np.dot(cov11,self._prec_12)

        self._dist = stat.multivariate_normal(mean=mean1,cov=cov11,allow_singular=True)


    def __call__(self, *args, **kwargs):
        x0 = args[0]
        x1 = args[1]

        self._dist.mean = self._mean1 - np.dot( self._l_cov,x1 - self._mean2)
        return self._dist.logpdf(x0)


    def decompose_coords_to_eigcords(self,X):
        res = scp.linalg.solve(self._eig_vec,X - self._mean1)
        res = res[self._eig_val>0]
        res[:] =0
        fc = lambda x: np.linalg.norm(self.vector_2_points(x)- X)

        mimise = opt.minimize(fc,res,method='TNC',bounds=self.generate_bounds(3))
        r_x = mimise.fun
        for i in range(10):
            mimise = opt.minimize(fc, mimise.x, method='TNC', bounds=self.generate_bounds(3), options={"disp": True})
            mimise = opt.minimize(fc, mimise.x, method='Powell', bounds=self.generate_bounds(3), options={"disp": True})

            if r_x - mimise.fun < 1:
                break
            r_x = mimise.fun
        return mimise.x

    def vector_2_points(self,X):

        return self._mean1 + np.dot(self._main_vecs,X)

    def generate_bounds(self,n):
        res = []
        r_vec = self._eig_val[self._eig_val>0]
        for i in range(self._main_vecs.shape[1]):
            m = np.sqrt(r_vec[i])
            res.append([-n*m,n*m])
        return res