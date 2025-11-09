import numpy as np
import cvxpy as cp

# ==============================================================================

def solve_QP(Phi:np.ndarray, H:np.ndarray, verbose=False):
        """ Solve the quadratic programming problem with Phi and H """
        P = 2 * Phi.T @ Phi
        q = 2 * (H.T @ Phi).T
        h = np.vstack([-H, np.zeros([3, 1])])
        parm_signs = np.zeros([3,3])
        parm_signs[0, 0] = 1
        parm_signs[2, 2] = -1
        G = np.vstack([-Phi, parm_signs])
        # ===
        # Convex optimization problem
        theta = cp.Variable([3, 1])
        objective = cp.Minimize( (1/2)*cp.quad_form(theta, P) + q.T @ theta )
        constraints = [G @ theta <= h]
        prob = cp.Problem(objective=objective, constraints=constraints)
        prob.solve(solver='MOSEK', verbose=verbose)
        #===
        # Solution
        return theta.value

# ==============================================================================

def PhiSat_mat(T, F, u1):
        """Returns the regression matrix for the given series or T, F and u1
           eta[k+1] = (u1[k]/F[k]) * [T**2 T 1] * [th1, th2 th3]^T
        """
        N = len(T)
        PhiSat = np.zeros([N, 3])
        # Looping
        for i in range(N):
                PhiSat[i, :] = (u1[i]/F[i]) * np.array([T[i]**2, T[i], 1])
        #===
        return PhiSat

# ==============================================================================
