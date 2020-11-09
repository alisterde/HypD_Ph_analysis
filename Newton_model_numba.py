import math
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 24})
import os

import pints

from numba import int32, float32, float64# import the types
from numba.experimental import jitclass

spec = [
    ('R', float64),
    ('temp', float64),
    ('F', float64),
    ('s', float64),
    ('GAMMA', float64),
    ('v', float64),
    ('E0', float64),
    ('T0', float64),
    ('I0', float64),
    ('epsilon_start', float64),
    ('epsilon_reverse', float64),
    ('potentialRange', float64[:]),
    ('fullPotentialRange', float64[:]),
    ('deltaepislon', float64),
    ('mew', float64),
    ('freq', float64),
    ('omega', float64),
    ('epsilon', float64),
    ('epsilon_r', float64),
    ('row', float64),
    ('zeta', float64),
    ('epsilon0_1', float64),
    ('epsilon0_2', float64),
    ('kappa0_1', float64),
    ('kappa0_2', float64),
    ('alpha1', float64),
    ('alpha2', float64),
    ('startT', float64),
    ('revT', float64),
    ('dimlessRevT', float64),
    ('endT', float64),
    ('timeStepSize', float64),
    ('dimlessTimeStepSize', float64),
    ('theta_X', float64[:]),
    ('theta_Z', float64[:]),
    ('dtheta_X_dt', float64[:]),
    ('dtheta_Z_dt', float64[:]),
    ('i', float64[:]),
    ('theta_X_Initial', float64),
    ('theta_Z_Initial', float64),
    ('I_inital', float64),
    ('gamma', float64),
    ('gamma1', float64),
    ('gamma2', float64),
    ('gamma3', float64)
]

@jitclass(spec)
class newtonRaphsonFT():
    '''
    This is a class to solve the mathematical model outlined in [1] written in base python
    using an implimentation of the Newton-Raphson methond

    [1] Adamson, Hope, Martin Robinson, Paul S. Bond, Basem Soboh, Kathryn Gillow, Alexandr N. Simonov, Darrell M. Elton, et al. 2017. 
    ‘Analysis of HypD Disulfide Redox Chemistry via Optimization of Fourier Transformed Ac Voltammetric Data’.
     Analytical Chemistry 89 (3): 1565–73. https://doi.org/10.1021/acs.analchem.6b03589.
    '''

    def __init__(self, timeStepSize: float, inital_current: float = 6.620541e-07, freq: float = 8.95931721948, startPotential: float = -0.15, revPotential: float = -0.75, rateOfPotentialChange: float = -22.35e-3,
                 numberOfMeasurements: int= 1000000, deltaepislon: float = 150E-3, uncomp_resis: float = 27.160770551,
                 electrode_area: float = 0.03, electode_coverage: float = 6.5e-12):

        
        #defining constants
        self.R = 8.314 #J / mol·K the perfect gas constant
        self.temp = 25.0+273.15  # k temperature in kelvin
        self.F = 96485.3329 # A.S.mol−1 Faraday constant

        #parameters for non-dimensionalisation
        self.s = electrode_area # E-4 # m^2 geometric area of the electrode
        self.GAMMA = electode_coverage # *1.0e3  #mols per m the surface coverage per unit area of the electrode
        self.v = rateOfPotentialChange #Vs-1 the rate at which the potential is swept over at

         # parameters for dimension removal
        self.E0 = (self.R*self.temp)/self.F
        self.T0 = (self.E0/self.v)
        self.I0 = (self.F*self.s*self.GAMMA)/self.T0

        # electode potential variables for epsilon
        self.epsilon_start = startPotential/self.E0 
        self.epsilon_reverse = revPotential/self.E0 
        #self.potentialRange = np.concatenate((np.linspace(startPotential, revPotential, numberOfMeasurements), np.linspace(revPotential,startPotential, numberOfMeasurements)), axis=None)
        # FIXME: potentialRange needs fixing
        self.potentialRange = np.linspace(startPotential, revPotential, numberOfMeasurements)
        self.fullPotentialRange = np.hstack((self.potentialRange,self.potentialRange[1:]))

        self.deltaepislon = deltaepislon/self.E0 # V 
        self.mew  = 0.0 # phase set by solver
        self.freq = freq #Hz (0.11161564832 seconds per period insure data has even number of periods)
        self.omega = 0.0 # dimensionless omega  set by solver 2.0*math.pi*self.freq*self.T0
        self.epsilon = 0.0
        self.epsilon_r = 0.0
        self.row = uncomp_resis*(self.I0/self.E0)# dimensionless uncompensated resistance


        self.zeta = self.F*self.s*self.GAMMA/(self.T0*self.I0) 

        # electro-potential of the reaction
        self.epsilon0_1 = 0.0 # set by solver -0.437459534627/self.E0
        self.epsilon0_2 = 0.0 # set by solver -0.46045114238/self.E0

        #electron transfer rate constants
        self.kappa0_1 = 0.0 # set by solver kappa0_1 *self.T0
        self.kappa0_2 = 0.0 # set by solver kappa0_2*self.T0

        #electron charge transfer coefficients
        self.alpha1 = 0.5
        self.alpha2 = 0.5


        #time interval
        
        self.startT = 0.0#specify in seconds
        self.revT =  abs((revPotential - startPotential)/(rateOfPotentialChange))#specify in seconds
        self.dimlessRevT = self.revT/self.T0#
        self.endT = self.revT*2.0
        self.timeStepSize = timeStepSize # in seconds
        self.dimlessTimeStepSize = timeStepSize/self.T0

        self.theta_X = np.zeros(numberOfMeasurements, dtype = np.float64)
        self.theta_Z = np.zeros(numberOfMeasurements, dtype = np.float64)
        self.dtheta_X_dt = np.zeros(numberOfMeasurements, dtype = np.float64)
        self.dtheta_Z_dt = np.zeros(numberOfMeasurements, dtype = np.float64)
        self.i = np.zeros(numberOfMeasurements, dtype = np.float64)

        self.theta_X_Initial = 1.0
        self.theta_Z_Initial = 0.0
        self.I_inital = inital_current


        # capacitance parameters
        self.gamma = 0.0
        self.gamma1 =  0.0
        self.gamma2 =   0.0
        self.gamma3 = 0.0


    def find_epsilon(self,time: float, index: int):
        ''' finding epsilon and epsilon_r
            as described in ref [1]
        '''
        if abs(self.T0) == self.T0:
            if time < self.dimlessRevT:
                # epsilon before dc current reversal
                self.epsilon = self.epsilon_start + time + (self.deltaepislon)*math.sin(self.omega*time + self.mew)
            elif time >= self.dimlessRevT:
            # epsilon after dc current reversal
                self.epsilon = self.epsilon_reverse - time + self.dimlessRevT + (self.deltaepislon)*math.sin(self.omega*time + self.mew)
        else:
            # taking into account changes is logic if T0 is negative 
            if time > self.dimlessRevT:
                # epsilon before dc current reversal
                self.epsilon = self.epsilon_start + time + (self.deltaepislon)*math.sin(self.omega*time + self.mew)
            elif time <= self.dimlessRevT:
                # epsilon after dc current reversal
                self.epsilon = self.epsilon_reverse - time + self.dimlessRevT + (self.deltaepislon)*math.sin(self.omega*time + self.mew)

        self.epsilon_r = self.epsilon  - self.row*self.i[int(index-1)]

    
    def find_dtheta_X_dt(self, index: int):
        '''
            finding  dtheta_X/dt as described in ref [1]
            dtheta_X/dt = k10*((1.0 - theta_X - theta_Z)*exp((1.0 - alpha1)*(epsilon_r - epsilon0_1))
                - theta_X*exp(-alpha*(epsilon_r - epsilon0_1)))
        '''
        self.dtheta_X_dt[index] = self.kappa0_1*((1.0 - self.theta_X[index] - self.theta_Z[index])*math.exp((1.0 - self.alpha1)*(self.epsilon_r - self.epsilon0_1))
                - self.theta_X[index]*math.exp(-self.alpha1*(self.epsilon_r - self.epsilon0_1)))

    def find_dtheta_Z_dt(self, index: int):
        '''
            finding  dtheta_Z/dt as described in ref [1]
            dtheta_Z/dt = k20*((1.0 - theta_X - theta_Z)*exp(alpha2*(epsilon_r - epsilon0_2)
                - theta_Z*exp((1.0 - alpha2)*(epsilon_r - epsilon0_2)))
        '''
        self.dtheta_Z_dt[index] = self.kappa0_2*((1.0 - self.theta_X[index] - self.theta_Z[index])*math.exp(-self.alpha2*(self.epsilon_r - self.epsilon0_2))
               - self.theta_Z[index]*math.exp((1.0 - self.alpha2)*(self.epsilon_r - self.epsilon0_2)))


    
    def current_function(self, i_n, t, i_n1, index: int):
        ''' 
        solving the current function described in ref [1] rearraged to equal zero 
        note the backwards euler is used for di/dT
        '''
        gamma = self.gamma
        gamma1 =  self.gamma1
        gamma2 = self.gamma2
        gamma3 = self.gamma3

        dtheta_X_dt = self.dtheta_X_dt[index]
        dtheta_Z_dt = self.dtheta_Z_dt[index]

        if abs(self.T0) == self.T0:
            if t < self.dimlessRevT:
                # epsilon before dc current reversal
                depsilon_rdt = 1.0 + self.omega*(self.deltaepislon)*math.cos(self.omega*t + self.mew) -(self.row*(i_n1 - i_n ))/self.dimlessTimeStepSize
            elif t >= self.dimlessRevT:
            # epsilon after dc current reversal
                depsilon_rdt = -1.0 + self.omega*(self.deltaepislon)*math.cos(self.omega*t + self.mew) -(self.row*(i_n1 - i_n ))/self.dimlessTimeStepSize
        else:
            # taking into account changes in logic if T0 is negative 
            if t > self.dimlessRevT:
                # epsilon before dc current reversal
                depsilon_rdt = 1.0 + self.omega*(self.deltaepislon)*math.cos(self.omega*t + self.mew) -(self.row*(i_n1 - i_n ))/self.dimlessTimeStepSize
            elif t <= self.dimlessRevT:
                # epsilon after dc current reversal
                depsilon_rdt = -1.0 + self.omega*(self.deltaepislon)*math.cos(self.omega*t + self.mew) -(self.row*(i_n1 - i_n ))/self.dimlessTimeStepSize
        

        return(-i_n1 + (gamma + gamma1*self.epsilon_r + gamma2*math.pow(self.epsilon_r, 2.0 ) + gamma3*math.pow(self.epsilon_r, 3.0 ))*depsilon_rdt 
                + self.zeta*(dtheta_X_dt - dtheta_Z_dt))

    
    def deriv_current_function(self,i_n, t, i_n1):
        ''' solving the differential WRT i current function described in ref [1] rearraged to equal zero 
            note the backwards euler is used for di/dT
        '''
        gamma = self.gamma
        gamma1 = self.gamma1
        gamma2 = self.gamma2
        gamma3 = self.gamma3

        if abs(self.T0) == self.T0:
            if t < self.dimlessRevT:
                # epsilon before dc current reversal
                depsilon_rdt = 1.0 + self.omega*(self.deltaepislon)*math.cos(self.omega*t + self.mew) -(self.row*(i_n1 - i_n ))/self.dimlessTimeStepSize
            elif t >= self.dimlessRevT:
            # epsilon after dc current reversal
                depsilon_rdt = -1.0 + self.omega*(self.deltaepislon)*math.cos(self.omega*t + self.mew) -(self.row*(i_n1 - i_n ))/self.dimlessTimeStepSize
        else:
            # taking into account changes in logic if T0 is negative 
            if t > self.dimlessRevT:
                # epsilon before dc current reversal
                depsilon_rdt = 1.0 + self.omega*(self.deltaepislon)*math.cos(self.omega*t + self.mew) -(self.row*(i_n1 - i_n ))/self.dimlessTimeStepSize
            elif t <= self.dimlessRevT:
                # epsilon after dc current reversal
                depsilon_rdt = -1.0 + self.omega*(self.deltaepislon)*math.cos(self.omega*t + self.mew) -(self.row*(i_n1 - i_n ))/self.dimlessTimeStepSize
        
        d2epsilon_rdidt = -self.row/self.dimlessTimeStepSize

        return(-1.0 + (-gamma1*self.row - 2.0*gamma2*self.row*(self.epsilon - self.row*i_n1) - 3.0*gamma3*self.row*math.pow((self.epsilon - self.row*i_n1),2.0))*depsilon_rdt 
                + (gamma + gamma1*self.epsilon_r + gamma2*math.pow(self.epsilon_r, 2.0 ) + gamma3*math.pow(self.epsilon_r, 3.0 ))*d2epsilon_rdidt)

    
    def newton_raphson(self, time, index: int):
        '''implementation of the newton-raphson method to solve for the current i at the next time step
        '''

        x0 = self.i[int(index-1)]
        x1 = self.i[int(index-1)]

        if time == 0.0 or time == -0.0:
            print('inital didT: ', self.deriv_current_function(x0, time, x1))

        h = self.current_function(x0, time, x1, index)/self.deriv_current_function(x0, time, x1)

        while abs(h) >= 0.00001:

            h = self.current_function(x0, time, x1, index)/self.deriv_current_function(x0, time, x1)

            # x(i+1) = x(i) - f(x) / f'(x) 

            x1 = x1 - h

        self.i[index] = x1

    
    def backwards_euler(self, index: int):
        '''
        applies the backwards euler method to theta_X and theta_Z
        as f(i(n),theta(n+1),time(n+1))
        i(n) is used to simplify the equation (rather than i(n+1))
        However, I believe this introduces error and is responsbile for the differences
        seen between this method and the pints implementation it is compared to.
        It helps explain why it scales with increasing row (uncompensated compensated)
        as the error will be greatest on the calculation of i(n+1) later
        '''
        A = math.exp((1.0 - self.alpha1)*(self.epsilon_r - self.epsilon0_1))
        B = math.exp(-self.alpha1*(self.epsilon_r - self.epsilon0_1))
        C = math.exp(-self.alpha2*(self.epsilon_r - self.epsilon0_2))
        D = math.exp((1.0 - self.alpha2)*(self.epsilon_r - self.epsilon0_2))

        left = np.array([[1.0/self.dimlessTimeStepSize + self.kappa0_1*A + self.kappa0_1*B, self.kappa0_1*A],
                        [self.kappa0_2*C, 1.0/self.dimlessTimeStepSize + self.kappa0_2*C + self.kappa0_2*D]])

        right = np.array([[self.kappa0_1*A + self.theta_X[int(index-1)]/self.dimlessTimeStepSize],
                        [self.kappa0_2*C + self.theta_Z[int(index-1)]/self.dimlessTimeStepSize]])

        solution = np.linalg.solve(left,right)

        self.theta_X[index] = solution[0,0]
        self.theta_Z[index] = solution[1,0]


    def set_faradaic_parameters(self, parameters):

        # electro-potential of the reaction
        self.kappa0_1 = parameters[0]
        self.kappa0_2 = parameters[1]
        self.epsilon0_1 = parameters[2]
        self.epsilon0_2 = parameters[3]
        self.mew = parameters[4]
        self.zeta = parameters[5]

    def set_capacitance_params(self, cap_params = None):
        '''
        takes a list of capasiance parameters and sets these for the model
        :param: cap_params = [gamma0, gamma1, gamma2, gamma3, omega]
        defaults to[0.0001411712994, 0.0195931114228, 0.000639515427465, 6.94671729801e-06, 2.0*math.pi*self.freq*self.T0]
        '''
        # if cap_params == None:
        #     cap_params = self.suggested_capacitance_params()

        non_dimensiosation_constant = self.E0*self.s/(self.T0*self.I0)
        self.gamma = (cap_params[0]*non_dimensiosation_constant)
        self.gamma1 = (cap_params[1]*self.E0)*non_dimensiosation_constant
        self.gamma2 = (cap_params[2]*math.pow(self.E0,2.0))*non_dimensiosation_constant
        self.gamma3 = (cap_params[3]*math.pow(self.E0,3.0))*non_dimensiosation_constant
        self.omega = (cap_params[4])


    def solve(self, times: float64):
        '''Steps through and solves the system
        '''
        t = times[1:]
        # non dimensioanless times
        t=t/self.T0
        # specifying initial values of the following
        self.theta_X[0] = self.theta_X_Initial
        self.theta_Z[0] = self.theta_Z_Initial
        self.i[0] = self.I_inital/self.I0
        # calculating initial differentials
        # these aren't used they are just
        # calculated for completeness
        self.find_dtheta_X_dt(0)
        self.find_dtheta_Z_dt(0)
        index = 1
        for time in t:
            self.find_epsilon(time, index)
            # calculate both theta values at next time step (time)
            self.backwards_euler(index)
            # cacluate differentials
            self.find_dtheta_X_dt(index)
            self.find_dtheta_Z_dt(index)
            # finding current at next time step
            self.newton_raphson(time, index)
            index = index + 1

        return self.i

class wrappedNewton(pints.ForwardModel):
    def __init__(self, times: float, inital_current: float = 6.620541e-07, freq: float = 8.95931721948, startPotential: float = -0.15, revPotential: float = -0.75,
                 rateOfPotentialChange: float = -22.35e-3, deltaepislon: float = 150E-3, uncomp_resis: float = 27.160770551,
                 electrode_area: float = 0.03, electode_coverage: float = 6.5e-12,initaldiscard: float = 0.025, enddiscard: float = 0.875,
                 cap_params: tuple = (1.13465158675681913e-04, 1.71228672908262905e-06, -2.02632468231267758e-05, -6.41028656277626023e-05, -6.47083954113886932e+01)):

        self.inital_current = inital_current
        self.freq = freq
        self.startPotential = startPotential
        self.revPotential = revPotential
        self.rateOfPotentialChange = rateOfPotentialChange
        length = times.shape
        self.numberOfMeasurements = length[0]
        self.half_of_measuremnts = int(self.numberOfMeasurements/2)
        self.deltaepislon = deltaepislon
        self.uncomp_resis = uncomp_resis
        self.electrode_area = electrode_area
        self.electode_coverage = electode_coverage
        self.initaldiscard = int(initaldiscard*self.half_of_measuremnts)
        self.enddiscard = int(enddiscard*self.half_of_measuremnts)

        # as the first time is at 0.0s we take one of the numberOfMeasurements
        # to split total time evenly and get the most accurate timeStepSize
        self.timeStepSize = times[-1]/(self.numberOfMeasurements - 1)

        # capactiance parameters
        self.gamma0 = cap_params[0]
        self.gamma1 = cap_params[1]
        self.gamma2 = cap_params[2]
        self.gamma3 = cap_params[3]
        self.omega = cap_params[4]
        
    def n_outputs(self):
        """ 
        See :meth:`pints.ForwardModel.n_outputs()`.
        number of outputs of the model
        """
        # current I
        return 1
    
    def n_parameters(self):
        """ See :meth:`pints.ForwardModel.n_parameters()`. 
        :return: dimensions of parameter vector
        """
        # kappa0_1, kappa0_2, epsilon0_1, epsilon0_2, mew, zeta
        # gamma0, gamma1, gamma2, gamma3, omega
        # need to change to 11 when sensitivities have been adjusted
        return 6
    
    def _simulate(self, parameters, times, FT):
        """
        Private helper function that uses ``scipy.integrate.odeint`` to
        simulate a model (with or without sensitivities).
        """
               
        # ensuring time and parameters are numpy array
        # times = np.asarray(times)
        parameters = np.asarray(parameters)

        # creating instance of newtonRaphsonFT

        solver = newtonRaphsonFT(timeStepSize=self.timeStepSize, inital_current=self.inital_current, freq=self.freq, startPotential=self.startPotential,
                                 revPotential=self.revPotential, rateOfPotentialChange=self.rateOfPotentialChange,
                                 numberOfMeasurements=self.numberOfMeasurements, deltaepislon=self.deltaepislon, uncomp_resis=self.uncomp_resis,
                                 electrode_area=self.electrode_area, electode_coverage=self.electode_coverage)

        # nondimensionalsing parameters
        params= []
        params.append(parameters[0]*solver.T0) #k0_1
        params.append(parameters[1]*solver.T0) #K0_2
        params.append(parameters[2]/solver.E0) #E0_1
        params.append(parameters[3]/solver.E0) # E0_2
        params.append(parameters[4]) # phase is demnsionless
        params.append(parameters[5]*(solver.F*solver.s*solver.GAMMA/(solver.T0*solver.I0))) # zeta
        # nondimensionalsing parameter for capacitance
        # self.gamma0 = (parameters[6]*self.E0/(self.T0*self.I0))
        # self.gamma1 = parameters[7]*self.E0
        # self.gamma2 = parameters[8]*math.pow(self.E0,2.0)
        # self.gamma3 = parameters[9]*math.pow(self.E0,3.0)
        # self.omega = 2.0*math.pi*self.freq*self.T0 # dimensionless omega
        params = np.asarray(params)

        solver.set_faradaic_parameters(params)
        capacitance = self.get_capacitance_params()
        solver.set_capacitance_params(capacitance)

        # solving up to potential reversal
        # nondimensionalsing time
        #dimlessTimes = times/solver.T0

        # solving using newtonRaphsonFT
        i = solver.solve(times)

        if FT == True:
            return self.FT_and_reduce_to_harmonics_4_to_12(i)
        else:
            return i


    def simulate(self, parameters, times):
        """ See :meth:`pints.ForwardModel.simulate()`. """
        i = self._simulate(parameters, times, False)
        return i

    def simulate_raw_current(self, parameters, times):
                
        i = self._simulate(parameters, times, False)
        #only returns Fourier transformed observable parameters which is current
        #i = np.append(i, i[0])
        # I = np.asarray(i)
        return i

    def simulate_reduced_FT_current(self, parameters, times):
                
        FT = self._simulate(parameters, times, True)
        #only returns Fourier transformed observable parameters which is current
        #i = np.append(i, i[0])
        # I = np.asarray(i)
        return FT

    def suggested_parameter(self):
        """Returns a list with suggestsed parameters for the model with dimensions
        kappa0_1 and kappa0_2 have dims s^(-1)
        epsilon0_1 and epsilon0_2 have dims V
        mew is in radians
        zeta is dimensionless
        return: [kappa0_1, kappa0_2, epsilon0_1, epsilon0_2, mew, zeta]
        """
        # mew = -8.82407598543352156e-02 # by my fitting
        mew = -0.031244092599793216 # by paper
        return [3400.0, 3400.0, -0.437459534627, -0.46045114238, mew, 1.0]

    def get_capacitance_params(self):
        """Returns a list with suggestsed capacitance parameters for the model with dimension
        return: [gamma0, gamma1, gamma2, gamma3, omega]
        """
        return [self.gamma0, self.gamma1, self.gamma2, self.gamma3, self.omega]

    def get_non_dimensionality_constants(self):
        """ Helper function to obtain the non dimensionality constants from the base Python class

        Returns:
            list: contains the non dimenstality constants [E0, T0, I0]
        """

        solver = newtonRaphsonFT(timeStepSize=self.timeStepSize, inital_current=self.inital_current, freq=self.freq, startPotential=self.startPotential,
                                 revPotential=self.revPotential, rateOfPotentialChange=self.rateOfPotentialChange, numberOfMeasurements=self.numberOfMeasurements,
                                 deltaepislon=self.deltaepislon, uncomp_resis=self.uncomp_resis, electrode_area=self.electrode_area,
                                 electode_coverage=self.electode_coverage)

        return [solver.E0, solver.T0, solver.I0]

    def FT_and_reduce_to_harmonics_4_to_12(self, Data):
        """Fourier transforms given data and reduces it to harmonics 3 to 12

        param: Data data to Fourier transform and reduce
        return: numpy array contain fourier transformed data for harmonics 3 -12
        """
        sp = np.fft.fft(Data)
        #sp = sp/(self.numberOfMeasurements*2) # as FFT scales by number of measurements to make inverse easy
        #sp = np.abs(sp) # combining real and imaginary parts
        #sp = 2*sp # doubling amplitudes as it is split between -ve and +ve frequencies and we are going to discard negative frequencies
        sp = sp[:self.half_of_measuremnts] #discarding -ve frequencies
        output = sp[self.initaldiscard:self.half_of_measuremnts - self.enddiscard] # reducing to harmonics 4 - 12
        output = np.asarray(output)
        return output

    def frequencies_for_harmonics_4_to_12(self, times):
        """Fourier transforms given data and reduces it to harmonics 3 to 12

        param: Data data to Fourier transform and reduce
        return: numpy array contain fourier transformed data for harmonics 3 -12
        """
        freq_org = np.fft.fftfreq(times.shape[0], d= self.timeStepSize)
        freq=freq_org[:self.half_of_measuremnts]
        freq = freq[self.initaldiscard:self.half_of_measuremnts - self.enddiscard] # reducing to harmonics 4 - 12
        return freq
    
    def harmonic_spacing(self, experimental_data, exp_times, adjustment: int = -1):
        """caculates the spacing between individual harmonics of the FT current

        Args:
            experimental_data (numpy array , float): experimental currents 
            exp_times (numpy array , float): experimental times corresponding to current measurements

        Returns:
            int: index of the highest magnitude peak of Fourier transformation i.e the peak of the first harmonic
        """

        full_sim = np.fft.fft(experimental_data)
        half_full_sim = full_sim[:self.half_of_measuremnts]

        freq_org = np.fft.fftfreq(exp_times.shape[0], d= self.timeStepSize)
        freq_org=freq_org[:self.half_of_measuremnts]

        x = np.where(freq_org < self.freq)
        print('x[0][-1]: ', x[0][-1])
        spacing = x[0][-1]

        y = np.where(freq_org > self.freq)
        print('y[0][0]: ', y[0][0])

        z = np.where(freq_org == self.freq)
        print('z[0]: ', z[0])

        low = spacing - 80
        upper = spacing + 81

        if exp_times is not None:
            xaxislabel = "frequency/Hz" # "potential/V"

            plt.figure(figsize=(18,10))
            plt.title("experimental FT")
            plt.ylabel("amplituide")
            plt.xlabel(xaxislabel)
            plt.plot(freq_org, np.log10(half_full_sim),'b', label='experimental_data')
            plt.plot(freq_org[low:upper], np.log10(half_full_sim[low:upper]),'r', label='experimental_harmonic 1')
            plt.legend(loc='best')
            plt.show()

            spacing_x = x[0][-1] +adjustment
            spacing_y = y[0][0] +adjustment

            plt.figure(figsize=(18,10))
            plt.title("experimental FT harmonic 1 and mid point (max)")
            plt.ylabel("amplituide")
            plt.xlabel(xaxislabel)
            plt.plot(freq_org[low:upper], np.log10(half_full_sim[low:upper]),'r', label='experimental_harmonic 1')
            plt.plot(freq_org[spacing_x], np.log10(half_full_sim[spacing_x]),'kX', label='spacing_x')
            plt.plot(freq_org[spacing_y], np.log10(half_full_sim[spacing_y]),'yX', label='spacing_y')
            if z[0] is not None:
                spacing_z =z[0]
                plt.plot(freq_org[spacing_z], np.log10(half_full_sim[spacing_z]),'cX', label='spacing_z')
            plt.legend(loc='best')
            plt.show()
        # change as approriate
        # was orginally -1
        return int(spacing_x)

    def index_distance_covering(self, Hz_interval, exp_times):
        """number of indexs needed to span the frequency interval Hz_interal

        Args:
            Hz_interval (float): Hz interval desried to cut of harmonics around
            exp_times (numpy array , float): experimental times corresponding to current measurements

        Returns:
            float: number of frequency steps (indexs) spaning the Hz interval 
        """

        freq_org = np.fft.fftfreq(exp_times.shape[0], d= exp_times[1])

        return Hz_interval/freq_org[1]


    def ploting_harmonic(self, experimental_data, times, parameter_for_sim, print_these_harmonics = None, Hz_interval = 1.5, print_harmonics = True,
                         check_FT_harmonic_locations = False, print_all_harmonics = True, print_simulated_harmonics_alone = False):
        """ploting harmonics of data against simulated harmonics

        Args:
            experimental_data (numpy array): experimental data
            times (numpy array): times for simulation
            parameter_for_sim (numpy array): [kappa0_1, kappa0_2, epsilon0_1, epsilon0_2, mew, zeta]
        """

        FT_reduced_exp = self.FT_and_reduce_to_harmonics_4_to_12(experimental_data)
        Ft_reduced_sim = self.simulate_reduced_FT_current(parameters= parameter_for_sim, times = times)

        freq = self.frequencies_for_harmonics_4_to_12(times =times)

        # 4th harmonic centered at 303
        # should be seprated by ~ 480 measurements
        print('*'*10+'cacluating harmonic spacing'+'*'*10)
        spacing = self.harmonic_spacing(experimental_data, times)
        print('Spacing between harmonics: ', spacing)

        # FIXME: issue finding location of 4th harmonic mid point
        print('\n'+'*'*10+'cacluating location of 4th harmonic'+'*'*10)
        x = np.where(freq < self.freq*4)
        mid_point_index = x[0][-1] - 4
        print('mid point index of 4th harmonic: ', mid_point_index)
        print('\n'+'*'*10+'index distance of ' + str(Hz_interval) + 'Hz'+'*'*10)
        index_window = self.index_distance_covering(Hz_interval, times)
        print('index window covering ' + str(Hz_interval) + 'Hz: ', index_window)
        index_window = np.round(index_window)
        print('int index window covering ' + str(Hz_interval) + 'Hz: ', index_window)


        if check_FT_harmonic_locations is True:
            self._ploting_FT_haromics(mid_point_index, index_window, spacing, freq, Ft_reduced_sim, FT_reduced_exp,
                                      print_simulated_harmonics_alone, print_all_harmonics, print_these_harmonics)
        
        if print_harmonics is True:
            dims = freq.shape
            self._ploting_ifft_haromics(mid_point_index, index_window, spacing, dims, Ft_reduced_sim, FT_reduced_exp,
                                        print_simulated_harmonics_alone, print_all_harmonics, print_these_harmonics)



    def _ploting_ifft_haromics(self, mid_point_index, index_window, spacing, dims, Ft_reduced_sim, FT_reduced_exp,
                               print_simulated_harmonics_alone, print_all_harmonics, print_these_harmonics):

            harmonic = 4
            low = int(mid_point_index - index_window)
            high = int(mid_point_index+ index_window + 1)
            mid = int(mid_point_index)

            temp = self.get_non_dimensionality_constants()

            I0 = temp[2]
            
            while high <= dims[0]:
                sim_plot = Ft_reduced_sim[low:high]
                mid_upper_sim_plot = Ft_reduced_sim[mid:high]
                lower_sim_plot = Ft_reduced_sim[low:mid]
                kaiser_window = np.kaiser(sim_plot.shape[0], 2)
                
                array_for_iFFT = np.hstack((mid_upper_sim_plot, lower_sim_plot))
                array_for_iFFT = np.multiply(kaiser_window,array_for_iFFT)
                sim_harmonic = np.fft.ifft(array_for_iFFT)
                sim_harmonic = sim_harmonic*I0

                sim_plot = FT_reduced_exp[low:high]
                mid_upper_sim_plot = FT_reduced_exp[mid:high]
                lower_sim_plot = FT_reduced_exp[low:mid]
                
                array_for_iFFT = np.hstack((mid_upper_sim_plot, lower_sim_plot))
                array_for_iFFT = np.multiply(kaiser_window,array_for_iFFT)
                exp_harmonic = np.fft.ifft(array_for_iFFT)
                exp_harmonic = exp_harmonic*I0

                if print_simulated_harmonics_alone is True:
                
                    if print_all_harmonics is True or harmonic in print_these_harmonics:
                        plt.figure(figsize=(18,10))
                        plt.title("Harmonic "+ str(harmonic))
                        plt.ylabel("current/Amps")
                        plt.plot(sim_harmonic.real,'r', label='Real Simulated Harmonic '+str(harmonic))
                        plt.plot(sim_harmonic.imag,'r', linestyle='dashed', label='Imaginary Simulated Harmonic '+str(harmonic))
                        plt.legend(loc='best')
                        plt.show()

                if print_all_harmonics is True or harmonic in print_these_harmonics:

                    plt.figure(figsize=(18,10))
                    plt.title("Simulated on Experimental Harmonic "+ str(harmonic))
                    plt.ylabel("current/Amps")
                    plt.plot(exp_harmonic.real,'b', label='Real Experimental Harmonic '+str(harmonic))
                    plt.plot(sim_harmonic.real,'r', label='Real Simulated Harmonic '+str(harmonic))
                    plt.plot(exp_harmonic.imag,'b', linestyle='dashed', label='Imaginary Experimental Harmonic '+str(harmonic))
                    plt.plot(sim_harmonic.imag,'r', linestyle='dashed', label='Imaginary Simulated Harmonic '+str(harmonic))
                    plt.legend(loc='best')
                    plt.show()

                    plt.figure(figsize=(18,10))
                    plt.title("Simulated on Experimental Harmonic "+ str(harmonic))
                    plt.ylabel("current/Amps")
                    plt.plot(np.absolute(exp_harmonic),'b', label='Absolute Experimental Harmonic '+str(harmonic))
                    plt.plot(np.absolute(sim_harmonic),'r', label='Absolute Simulated Harmonic '+str(harmonic))
                    plt.legend(loc='best')
                    plt.show()


                high = high+spacing
                low = low+spacing
                mid = mid+spacing
                harmonic = harmonic +1
    

    def _ploting_FT_haromics(self, mid_point_index, index_window, spacing, freq, Ft_reduced_sim, FT_reduced_exp,
                               print_simulated_harmonics_alone, print_all_harmonics, print_these_harmonics):
        
        dims = freq.shape
    
        harmonic = 4
        low = int(mid_point_index - index_window)
        high = int(mid_point_index+ index_window + 1)
        mid = int(mid_point_index)

        while high <= dims[0]:
            sim_plot = Ft_reduced_sim[low:high]
            mid_upper_sim_plot = Ft_reduced_sim[mid:high]
            lower_sim_plot = Ft_reduced_sim[low:mid]
            print('sim_plot.shape:', sim_plot.shape)
            print('mid_upper_sim_plot.shape:', mid_upper_sim_plot.shape)
            print('lower_sim_plot.shape:', lower_sim_plot.shape)
            xaxis = freq[low:high] #model.potentialRange
            xaxis_mid_upper = freq[mid:high]
            xaxis_lower= freq[low:mid]
            xaxislabel = "freq/Hz" # "potential/V"

            exp_plot = FT_reduced_exp[low:high]
            mid_upper_exp_plot = FT_reduced_exp[mid:high]
            lower_exp_plot = FT_reduced_exp[low:mid]

            if print_all_harmonics is True or harmonic in print_these_harmonics:
                plt.figure(figsize=(18,10))
                plt.title("simulation FT")
                plt.ylabel("amplituide")
                plt.xlabel(xaxislabel)
                plt.plot(xaxis, np.log10(sim_plot),'r', label='simulated_harmonic_'+str(harmonic))
                plt.plot(xaxis_mid_upper[1:], np.log10(mid_upper_sim_plot[1:]),'k', label='simulated_harmonic_'+str(harmonic)+'_upper_1/2')
                plt.plot(xaxis_lower, np.log10(lower_sim_plot),'m', label='simulated_harmonic_'+str(harmonic)+'_lower_1/2')
                plt.plot(freq[mid], np.log10(Ft_reduced_sim[mid]),'cX', label='harmonic_center')
                plt.legend(loc='best')
                plt.show()

                plt.figure(figsize=(18,10))
                plt.title("experimental FT")
                plt.ylabel("amplituide")
                plt.xlabel(xaxislabel)
                plt.plot(xaxis, np.log10(exp_plot),'r', label='experimental_harmonic_'+str(harmonic))
                plt.plot(xaxis_mid_upper[1:], np.log10(mid_upper_exp_plot[1:]),'k', label='experimental_harmonic_'+str(harmonic)+'_upper_1/2')
                plt.plot(xaxis_lower, np.log10(lower_exp_plot),'m', label='experimental_harmonic_'+str(harmonic)+'_lower_1/2')
                plt.plot(freq[mid], np.log10(FT_reduced_exp[mid]),'cX', label='harmonic_center')
                plt.legend(loc='best')
                plt.show()

                plt.figure(figsize=(18,10))
                plt.title("simulation and experimental FT")
                plt.ylabel("amplituide")
                plt.xlabel(xaxislabel)
                plt.plot(freq, np.log10(FT_reduced_exp),'b', label='experimental_data')
                plt.plot(xaxis, np.log10(sim_plot),'r', label='simulated_harmonic_'+str(harmonic))
                plt.legend(loc='best')
                plt.show()


            high = high+spacing
            low = low+spacing
            mid = mid+spacing
            harmonic = harmonic +1