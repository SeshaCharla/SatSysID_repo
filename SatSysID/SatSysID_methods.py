import numpy as np
import SatSysID_funcs as sf

class SatSys_ssd:
        """ The class that holds the data and methods for ssd SatSysID """

        def __init__(self, ssd_data, name):
                """ Initiates the class with ssd data """
                self.ssd = ssd_data
                self.name = name
                # Calculating theta
                self.Phi = sf.PhiSat_mat(self.ssd['T'], self.ssd['F'], self.ssd['u1'])
                self.H = np.matrix(self.ssd['eta']).T
                self.W = sf.W_kde(eta=self.ssd['eta'], u2=self.ssd['u2'],
                                  T=self.ssd['T'], F=self.ssd['F'])
                self.theta = sf.solve_QP(self.Phi[0:-1,:], self.H[1:, :], W=self.W[:-1, :-1], verbose=True)
                # Calculating epsilon
                self.eta_hat = (self.Phi[0:-1, :] @ self.theta).flatten()
                self.eps_bimodal = (self.eta_hat - self.ssd['eta'][1:])
                self.indices = [i for i in range(len(self.eps_bimodal)) if self.eps_bimodal[i] <= 2 and self.eps_bimodal[i] >= 0]
                self.eps = self.eps_bimodal[self.indices]
                # Fitting distribution
                self.hfn_fit = sf.fit_dist( self.eps, eps_max=2)
                self.hfn_lambda = sf.scale2lambda(self.hfn_fit.fit_result.params[1])
                self.var_eps = self.hfn_fit.fit_result.params[1]**2 * (1 - 2/np.pi)
                self.exp_eps = 1/self.hfn_lambda
                self.I_theta = sf.Fisher_Information(self.hfn_lambda, self.Phi, self.indices)
                self.C_theta = np.linalg.inv(self.I_theta)
                self.sigma_eta = np.sqrt( np.array( [ (self.Phi[j,:]@ self.C_theta @ self.Phi[j,:].T)
                                                        for j in range(np.shape(self.Phi)[0]-1) ] ))
                # self.temp_var()

        #==============================================================================================

        def predict_eta_sat(self, ssd_ref):
                """ Predicts the response of this particular system to reference inputs ssd_ref"""
                self.ssd_ref = ssd_ref
                Phi_ref = sf.PhiSat_mat(self.ssd_ref['T'], self.ssd_ref['F'], self.ssd_ref['u1'])
                self.eta_pred = (Phi_ref[0:-1, :] @ self.theta).flatten()
                self.sigma_pred = np.sqrt( np.array( [ (Phi_ref[j,:] @ self.C_theta @ Phi_ref[j,:].T)
                                                        for j in range(np.shape(Phi_ref)[0]-1) ] ))

        # =================================================================================================

        def temp_var(self):
                """ Get the temperature variation of the Max. sigma """
                N = 1000
                self.T_lin = np.linspace(np.min(self.ssd['T']), np.max(self.ssd['T']), N)
                Phi = np.matrix([np.array([Ti**2, Ti, 1]) for Ti in self.T_lin])
                self.gamma_max = (Phi @ self.theta).flatten()
                self.sigma_gamma_max = np.sqrt( np.array( [ (Phi[j,:] @ self.C_theta @  Phi[j,:].T)[0,0]
                                                        for j in range(N) ] ))

        # ===================================================================================================

        def calc_Tw(self, theta_ref):
                """ Calculates the wald test's test-statistic given theta_ref """
                theta_diff = (self.theta - theta_ref)
                self.Tw = (theta_diff.T @ self.I_theta @ theta_diff)[0,0]
                return self.Tw
