import numpy as np
import cvxpy as cp

# ==============================================================================
def solve_QP(Phi:np.ndarray, H:np.ndarray):
        """ Solve the quadratic programming problem with Phi and H """
        P = 2 * Phi.T @ Phi
        q = 2 * H.T @ Phi
        h = np.vstack([H, np.zeros([3, 1])])
        parm_signs = np.zeros([3,3])
        parm_signs[0, 0] = 1
        parm_signs[2, 2] = -1
        G = np.vstack([Phi, parm_signs])

        # Convex optimization problem
        theta = cp.Variable([3, 1])
        objective = cp.Minimize( (1/2)*cp.quad_form(theta, P) + q.T @ theta )
        constraints = [G @ theta <= h]
        prob = cp.Problem(objective=objective, constraints=constraints)
        prob.solve(solver='MOSEK', verbose=True)

        return theta.value

# ==============================================================================
