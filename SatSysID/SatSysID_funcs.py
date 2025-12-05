import numpy as np
import cvxpy as cp
from scipy.stats import halfnorm, goodness_of_fit, gaussian_kde
from scipy.optimize import linprog

# ==============================================================================

def solve_LP(Phi:np.ndarray, H:np.ndarray, verbose=False):
        """ Solve the quadratic programming problem with Phi and H """
        # # Convex optimization problem
        # theta = cp.Variable([4, 1])
        # objective = cp.Minimize(cp.sum(Phi@theta))
        # constraints = [Phi@theta >= H]
        # prob = cp.Problem(objective=objective, constraints=constraints)
        # prob.solve(solver="MOSEK", verbose=verbose)
        # return theta.value
        # ==============================================
        c = np.sum(np.asarray(Phi), axis=0)
        A_ub = -np.asarray(Phi)
        b_ub = -np.asarray(H).flatten()
        inf_bounds = [(-np.inf, np.inf) for _ in range(len(c))]
        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, method='highs', bounds=inf_bounds, options={'disp': verbose})
        return res

# ==============================================================================

def PhiSat_mat(T:np.ndarray, F:np.ndarray, u1:np.ndarray, T0:float, Tr:float)->np.ndarray:
        """Returns the regression matrix for the given series or T using Chebyshev polynomials
           eta[k+1] = [th1, th2 th3]^T
           T_0 = 1,
           T_1 = x,
           T_2 = 2x^2 -1
           T_3 = 4x^3 - 3x
           T_4 = 8x^4 - 8x^2 +1
        """
        N = len(T)
        PhiSat = np.zeros([N, 3])
        # Looping
        for i in range(N):
                x = (T[i] - T0)/Tr
                PhiSat[i, :] = (u1[i]/F[i]) * np.array([(2*x**2)-1, x, 1])
        #===
        print("Condition number of PhiSat matrix: ", np.linalg.cond(PhiSat))
        return PhiSat

# ==============================================================================

def scale2lambda(scale:float):
        """ Converts the scale parameter to lambda """
        return np.sqrt(np.pi/2)/(scale)

# =======================================================

def fit_dist(eps:np.ndarray, eps_max:float=8):
        """ Fits a half normal distribution to epsilon """
        eps_stat =  [eps[i] for i in range(len(eps)) if (eps[i] < eps_max and eps[i]>=0)]
        known_parms = {'loc' : 0}
        res = goodness_of_fit(halfnorm, eps_stat, known_params=known_parms, statistic='ks')
        return res

# ===============================================================================================

def Fisher_Information(lmbd:float, Phi:np.ndarray)->np.ndarray:
        """ Returns the Fisher Information Matrix for the given lambda and regression matrix Phi """
        N, _ = np.shape(Phi)
        I_theta = (2*lmbd**2/np.pi) * (Phi.T @ Phi)
        return I_theta

# ==============================================================================================


def W_kde(eta:np.ndarray, u2:np.ndarray, T:np.ndarray, F:np.ndarray)->np.ndarray:
        """ Returns the diagonal weight matrix square for uniform sampling in the given state/input range"""
        pdf = gaussian_kde([eta, u2, T, F])
        N = np.size(T)
        w = np.array([1/(pdf([eta[i], u2[i], T[i], F[i]])) for i in range(N)])
        w_norm = w/np.sum(w)
        return np.diag(w_norm.flatten())
