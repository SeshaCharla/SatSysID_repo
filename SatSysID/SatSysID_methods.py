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
                print(self.Phi)
                self.H = np.matrix(self.ssd['eta']).T
                self.W = sf.W_kde(eta=self.ssd['eta'][1:], u1=self.ssd['u1'][0:-1],
                                  u2=self.ssd['u2'][0:-1], T=self.ssd['T'][0:-1],
                                  F=self.ssd['F'][0:-1])
                self.theta = sf.solve_QP(self.Phi[0:-1,:], self.H[1:, :], self.W, verbose=True)
                # Calculating epsilon
                self.eta_hat = (self.Phi[0:-1, :] @ self.theta).flatten()
                self.eps = (self.eta_hat - self.ssd['eta'][1:]) @ self.W
                # Fitting distribution
                self.hfn_fit = sf.fit_dist( self.eps, eps_max= np.max(self.W*self.H[1:,:]) )
                self.hfn_lambda = sf.scale2lambda(self.hfn_fit.fit_result.params[1])
                self.C_theta = np.linalg.inv( (2*self.hfn_lambda**2/np.pi)*
                                              (self.Phi[1: :].T@(self.W.T @ self.W)@self.Phi[1:, :])
                                             )
                self.sigma_eta = np.sqrt( np.array( [ (self.Phi[j,:]@ self.C_theta @ self.Phi[j,:].T)
                                                        for j in range(np.shape(self.Phi)[0]-1) ]
                                                   )
                                         )

        def predict_eta_sat(self, ssd_ref):
                """ Predicts the response of this particular system to reference inputs ssd_ref"""
                self.ssd_ref = ssd_ref
                Phi_ref = sf.PhiSat_mat(self.ssd_ref['T'], self.ssd_ref['F'], self.ssd_ref['u1'])
                self.eta_pred = (Phi_ref[0:-1, :] @ self.theta).flatten()
                self.sigma_pred = np.sqrt( np.array( [ (Phi_ref[j,:] @ self.C_theta @ Phi_ref[j,:].T)
                                                        for j in range(np.shape(Phi_ref)[0]-1) ]
                                                   )
                                         )
