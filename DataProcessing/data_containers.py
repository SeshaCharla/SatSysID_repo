class ssd:
        """ Takes the ssd dict and converts it into the data class """
        def __init__(self, ssd):
                self.t  = ssd['t']
                self.x1 = ssd['x1']
                self.u1 = ssd['u1']
                self.T  = ssd['T']
                self.F  = ssd['F']
                self.u2 = ssd['u2']

class iod:
        """ Takes the iod dict and converts it into the data class
                Also handles time discontinuities.
        """
        def __init__(self, iod):
                self.t  = iod['t']
                self.y1 = iod['y1']
                self.u1 = iod['u1']
                self.T  = iod['T']
                self.F  = iod['F']
