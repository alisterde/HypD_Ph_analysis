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
    ('tau', float64),
    ('v', float64),
    ('E0', float64),
    ('T0', float64),
    ('I0', float64),
    ('epsilon_start', float64),
    ('epsilon_reverse', float64),
    ('deltaepislon', float64),
    ('mew', float64),
    ('freq', float64),
    ('omega', float64),
    ('omega0', float64),
    ('epsilon', float64),
    ('epsilon_r', float64),
    ('row', float64),
    ('startT', float64),
    ('revT', float64),
    ('dimlessRevT', float64),
    ('endT', float64),
    ('timeStepSize', float64),
    ('dimlessTimeStepSize', float64),
    ('i', float64[:]),
    ('I_inital', float64),
    ('gamma0', float64),
    ('gamma1', float64),
    ('gamma2', float64),
    ('gamma3', float64),
    ('gamma4', float64),
    ('gamma5', float64),
    ('gamma6', float64),
    ('gamma7', float64)
]

@jitclass(spec)
class newtonRaphsonCap():
    '''
    This is a class to solve the mathematical model outlined in [1] written in base python
    using an implimentation of the Newton-Raphson methond
    [1] Adamson, Hope, Martin Robinson, Paul S. Bond, Basem Soboh, Kathryn Gillow, Alexandr N. Simonov, Darrell M. Elton, et al. 2017. 
    ‘Analysis of HypD Disulfide Redox Chemistry via Optimization of Fourier Transformed Ac Voltammetric Data’.
     Analytical Chemistry 89 (3): 1565–73. https://doi.org/10.1021/acs.analchem.6b03589.
    '''

    def __init__(self, timeStepSize: float, numberOfMeasurements: int, startPotential: float = -0.15, revPotential: float = -0.75,
                 rateOfPotentialChange: float = -22.35e-3, inital_current: float = 6.620541e-07, freq: float = 8.95931721948,
                 deltaepislon: float = 150E-3, electrode_area: float = 0.03, electode_coverage: float = 6.5e-12):
        
        #defining constants
        self.R = 8.314 #J / mol·K the perfect gas constant
        self.temp = 25.0+273.15  # k temperature in kelvin
        self.F = 96485.3329 # A.S.mol−1 Faraday constant

        #parameters for non-dimensionalisation
        self.s = electrode_area#E-4 # m^2 geometric area of the electrode
        self.tau = electode_coverage#*1.0e3  #mols per m the surface coverage per unit area of the electrode
        self.v = rateOfPotentialChange #Vs-1 the rate at which the potential is swept over at

         # parameters for dimension removal
        self.E0 = (self.R*self.temp)/self.F
        self.T0 = (self.E0/self.v)
        self.I0 = (self.F*self.s*self.tau)/self.T0

        # electode potential variables for epsilon
        self.epsilon_start = startPotential/self.E0 
        self.epsilon_reverse = revPotential/self.E0 

        self.deltaepislon = deltaepislon/self.E0 # V 
        self.mew  = 0.0 #-0.031244092599793216 #phase
        self.freq = freq #Hz (0.11161564832 seconds per period insure data has even number of periods)
        self.omega = 2.0*math.pi*self.freq # *self.T0 # dimensionless omega
        self.omega0 = 2.0*math.pi*self.freq# *self.T0 # dimensionless omega 
        self.epsilon = 0.0
        self.epsilon_r = 0.0
        self.row = 0.0 # 27.160770551*(self.I0/self.E0)# dimensionless uncompensated resistance

        #time interval
        
        self.startT = 0.0#specify in seconds
        self.revT =  abs((revPotential - startPotential)/(rateOfPotentialChange))#specify in seconds
        self.dimlessRevT = self.revT/self.T0#
        self.endT = self.revT*2.0
        self.timeStepSize = timeStepSize #self.revT/numberOfMeasurements # in seconds
        self.dimlessTimeStepSize = (self.timeStepSize)/self.T0

        self.i = np.zeros(numberOfMeasurements, dtype = np.float64)

        self.I_inital = inital_current

        # capacitance parameters for forward sweep
        self.gamma0 = 0.0
        self.gamma1 =  0.0
        self.gamma2 =   0.0
        self.gamma3 = 0.0

        # capacitance parameters for reverse sweep
        self.gamma4 = 0.0
        self.gamma5 =  0.0
        self.gamma6 =   0.0
        self.gamma7 = 0.0


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

    def current_function(self, i_n, t, i_n1):
        ''' 
        solving the current function described in ref [1] rearraged to equal zero 
        note the backwards euler is used for di/dT
        '''

        if abs(self.T0) == self.T0:
            if t < self.dimlessRevT:
                # capacitance polynomial before dc current reversal
                gamma0 = self.gamma0
                gamma1 =  self.gamma1
                gamma2 = self.gamma2
                gamma3 = self.gamma3
                # epsilon before dc current reversal
                depsilon_rdt = 1.0 + self.omega*(self.deltaepislon)*math.cos(self.omega*t + self.mew) -(self.row*(i_n1 - i_n ))/self.dimlessTimeStepSize
            elif t >= self.dimlessRevT:
                # capacitance polynomial after dc current reversal
                gamma0 = self.gamma4
                gamma1 =  self.gamma5
                gamma2 = self.gamma6
                gamma3 = self.gamma7
                # epsilon after dc current reversal
                depsilon_rdt = -1.0 + self.omega*(self.deltaepislon)*math.cos(self.omega*t + self.mew) -(self.row*(i_n1 - i_n ))/self.dimlessTimeStepSize
        else:
            # taking into account changes in logic if T0 is negative 
            if t > self.dimlessRevT:
                # capacitance polynomial before dc current reversal
                gamma0 = self.gamma0
                gamma1 =  self.gamma1
                gamma2 = self.gamma2
                gamma3 = self.gamma3
                # epsilon before dc current reversal
                depsilon_rdt = 1.0 + self.omega*(self.deltaepislon)*math.cos(self.omega*t + self.mew) -(self.row*(i_n1 - i_n ))/self.dimlessTimeStepSize
            elif t <= self.dimlessRevT:
                # capacitance polynomial after dc current reversal
                gamma0 = self.gamma4
                gamma1 =  self.gamma5
                gamma2 = self.gamma6
                gamma3 = self.gamma7
                # epsilon after dc current reversal
                depsilon_rdt = -1.0 + self.omega*(self.deltaepislon)*math.cos(self.omega*t + self.mew) -(self.row*(i_n1 - i_n ))/self.dimlessTimeStepSize
        

        return(-i_n1 + (gamma0 + gamma1*self.epsilon_r + gamma2*math.pow(self.epsilon_r, 2.0 ) + gamma3*math.pow(self.epsilon_r, 3.0 ))*depsilon_rdt)

    
    def deriv_current_function(self,i_n, t, i_n1):
        ''' solving the differential WRT i current function described in ref [1] rearraged to equal zero 
            note the backwards euler is used for di/dT
        '''

        if abs(self.T0) == self.T0:
            if t < self.dimlessRevT:
                # capacitance polynomial before dc current reversal
                gamma0 = self.gamma0
                gamma1 =  self.gamma1
                gamma2 = self.gamma2
                gamma3 = self.gamma3
                # epsilon before dc current reversal
                depsilon_rdt = 1.0 + self.omega*(self.deltaepislon)*math.cos(self.omega*t + self.mew) -(self.row*(i_n1 - i_n ))/self.dimlessTimeStepSize
            elif t >= self.dimlessRevT:
                # capacitance polynomial after dc current reversal
                gamma0 = self.gamma4
                gamma1 =  self.gamma5
                gamma2 = self.gamma6
                gamma3 = self.gamma7
                # epsilon after dc current reversal
                depsilon_rdt = -1.0 + self.omega*(self.deltaepislon)*math.cos(self.omega*t + self.mew) -(self.row*(i_n1 - i_n ))/self.dimlessTimeStepSize
        else:
            # taking into account changes in logic if T0 is negative 
            if t > self.dimlessRevT:
                # capacitance polynomial before dc current reversal
                gamma0 = self.gamma0
                gamma1 =  self.gamma1
                gamma2 = self.gamma2
                gamma3 = self.gamma3
                # epsilon before dc current reversal
                depsilon_rdt = 1.0 + self.omega*(self.deltaepislon)*math.cos(self.omega*t + self.mew) -(self.row*(i_n1 - i_n ))/self.dimlessTimeStepSize
            elif t <= self.dimlessRevT:
                # capacitance polynomial after dc current reversal
                gamma0 = self.gamma4
                gamma1 =  self.gamma5
                gamma2 = self.gamma6
                gamma3 = self.gamma7
                # epsilon after dc current reversal
                depsilon_rdt = -1.0 + self.omega*(self.deltaepislon)*math.cos(self.omega*t + self.mew) -(self.row*(i_n1 - i_n ))/self.dimlessTimeStepSize
        
        d2epsilon_rdidt = -self.row/self.dimlessTimeStepSize

        return(-1.0 + (-gamma1*self.row - 2.0*gamma2*self.row*(self.epsilon - self.row*i_n1) - 3.0*gamma3*self.row*math.pow((self.epsilon - self.row*i_n1),2.0))*depsilon_rdt 
                + (gamma0 + gamma1*self.epsilon_r + gamma2*math.pow(self.epsilon_r, 2.0 ) + gamma3*math.pow(self.epsilon_r, 3.0 ))*d2epsilon_rdidt)

    
    def analytical_current_solution(self, time, index: int):
        '''implementation of the newton-raphson method to solve for the current i at the next time step
        '''

        x0 = self.i[int(index-1)]
        x1 = self.i[int(index-1)]

        if time == 0.0 or time == -0.0:
            print('inital didT: ', self.deriv_current_function(x0, time, x1))

        h = self.current_function(x0, time, x1)/self.deriv_current_function(x0, time, x1)

        iterations = 0
        max_iterations = 100
        while abs(h) >= 0.00001 and iterations <= max_iterations:

            h = self.current_function(x0, time, x1)/self.deriv_current_function(x0, time, x1)

            # x(i+1) = x(i) - f(x) / f'(x) 

            x1 = x1 - h
            iterations = iterations +1

        self.i[index] = x1

    def set_capacitance_params(self, cap_params):
        '''
        takes a list of capasiance parameters and sets these for the model
        :param: cap_params = [gamma0, gamma1, gamma2, gamma3, gamma4,
                              gamma5, gamma6, gamma7, omega, mew, row]
        '''

        non_dimensiosation_constant = self.E0*self.s/(self.T0*self.I0)
        self.gamma0 = (cap_params[0]*non_dimensiosation_constant)
        self.gamma1 = (cap_params[1]*self.E0)*non_dimensiosation_constant
        self.gamma2 = (cap_params[2]*math.pow(self.E0,2.0))*non_dimensiosation_constant
        self.gamma3 = (cap_params[3]*math.pow(self.E0,3.0))*non_dimensiosation_constant
        self.gamma4 = (cap_params[4]*non_dimensiosation_constant)
        self.gamma5 = (cap_params[5]*self.E0)*non_dimensiosation_constant
        self.gamma6 = (cap_params[6]*math.pow(self.E0,2.0))*non_dimensiosation_constant
        self.gamma7 = (cap_params[7]*math.pow(self.E0,3.0))*non_dimensiosation_constant
        self.omega = cap_params[8]*self.T0
        self.mew = cap_params[9]
        self.row = cap_params[10]*(self.I0/self.E0)


    def solve(self, times):
        '''Steps through and solves the system
        '''
        # print('In base python simulator')
        t = times[1:]
        # non dimensioanless times
        t=t/self.T0
        # specifying initial value of the current
        self.i[0] = self.I_inital/self.I0
        index = 1
        for time in t:
            # print('find_epsilon')
            self.find_epsilon(time, index)
            # finding current at next time step
            # print('analytical_current_solution')
            self.analytical_current_solution(time, index)
            index = index + 1
        
        return self.i


class wrappedNewtonCap(pints.ForwardModel):
    def __init__(self, times: float, removed_measures_to_account_for: int, startPotential: float = -0.15, revPotential: float = -0.75, rateOfPotentialChange: float = -22.35e-3,
                inital_current: float = 6.620541e-07, freq: float = 8.95931721948, deltaepislon: float = 150E-3,
                electrode_area: float = 0.03, electode_coverage: float = 6.5e-12, beingPureCapitanceto: float = 0.2,
                endPureCapatianceFor: float = 0.2):

        self.startPotential = startPotential
        self.revPotential = revPotential
        self.rateOfPotentialChange = rateOfPotentialChange
       
       
        length = times.shape
        self.numberOfMeasurements = length[0]
        # As the first time is at 0.0s we take one of the numberOfMeasurements
        # to split total time evenly and get the most accurate timeStepSize
        self.timeStepSize = times[-1]/(self.numberOfMeasurements - 1)

        self.endPureCapatianceFor = int(endPureCapatianceFor*int(self.numberOfMeasurements/2))
        self.beingPureCapitanceto = int(beingPureCapitanceto*int(self.numberOfMeasurements/2))
        print('self.endPureCapatianceFor: ',self.endPureCapatianceFor)
        print('self.beingPureCapitanceto: ',self.beingPureCapitanceto)


        self.endCap = self.numberOfMeasurements - self.endPureCapatianceFor + int(removed_measures_to_account_for)

        self.midCapLow = int(self.numberOfMeasurements/2)-self.endPureCapatianceFor
        self.midCaphigh = int(self.numberOfMeasurements/2)+self.beingPureCapitanceto

        # parameters to pass to main model
        self.inital_current=inital_current
        self.freq=freq
        self.deltaepislon=deltaepislon
        self.electrode_area=electrode_area
        self.electode_coverage=electode_coverage

        # capactiance parameters
        self.gamma0 = 0.0
        self.gamma1 = 0.0
        self.gamma2 = 0.0
        self.gamma3 = 0.0
        self.omega = 0.0
        self.mew = 0.0
        self.uncomp_resis = 27.160770551
        
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
        # [gamma0, gamma1, gamma2, gamma3, gamma4, gamma5, gamma6,
        #  gamma7, mew, omega, uncompensated_resistance]
        return 11
    
    def _simulate(self, parameters, times, reduce):
        """
        Private helper function that uses ``scipy.integrate.odeint`` to
        simulate a model (with or without sensitivities).
        """
               
        # ensuring time and parameters are numpy array
        # times = np.asarray(times)
        # parameters = np.asarray(parameters)

        # creating instance of newtonRaphsonCap

        solver = newtonRaphsonCap(timeStepSize=self.timeStepSize, numberOfMeasurements=self.numberOfMeasurements,
                                  startPotential=self.startPotential, revPotential=self.revPotential,
                                  rateOfPotentialChange=self.rateOfPotentialChange, inital_current=self.inital_current,
                                  freq=self.freq, deltaepislon=self.deltaepislon, electrode_area =self.electrode_area,
                                  electode_coverage=self.electode_coverage)

        solver.set_capacitance_params(parameters)
        # solving using newtonRaphsonFT
        i = solver.solve(times)

        if reduce == True:
            output = self.reshape_to_cap_regions(i)
            return output
        else:
            return i


    def simulate(self, parameters, times):
        """ See :meth:`pints.ForwardModel.simulate()`. """

        i = self._simulate(parameters, times, False)
        I = np.asarray(i)

        return I

    def simulate_fitting_regions(self, parameters, times):
       
        i = self._simulate(parameters, times, True)
        I = np.asarray(i)
        return I

    def suggested_capacitance_params(self):
        """Returns a list with suggestsed capacitance parameters for the model with dimension
        return: [gamma0, gamma1, gamma2, gamma3, gamma4, gamma5, gamma6,
                 gamma7, omega, mew, uncompensated_resistance]
        """

        solver = newtonRaphsonCap(timeStepSize=self.timeStepSize, numberOfMeasurements=self.numberOfMeasurements,
                                  startPotential=self.startPotential, revPotential=self.revPotential,
                                  rateOfPotentialChange=self.rateOfPotentialChange, inital_current=self.inital_current,
                                  freq=self.freq, deltaepislon=self.deltaepislon, electrode_area =self.electrode_area,
                                  electode_coverage=self.electode_coverage)
        
        gamma0 = 0.0001411712994
        gamma1 = gamma0*0.0195931114228
        gamma2 = gamma0*0.000639515427465
        gamma3 = gamma0*6.94671729801e-06
        gamma4 = 0.0001411712994
        gamma5 = gamma4*0.0195931114228
        gamma6 = gamma4*0.000639515427465
        gamma7 = gamma4*6.94671729801e-06

        output = [gamma0, gamma1, gamma2, gamma3, gamma4, gamma5, gamma6,
                  gamma7, 2.0*math.pi*solver.freq*solver.T0, -0.031244092599793216, 27.160770551]

        return output

    def reshape_to_cap_regions(self, array):

        raw = np.asarray(array)
        a = raw[:self.beingPureCapitanceto]
        b = raw[self.midCapLow:self.midCaphigh]
        c = raw[self.endCap:]
        reshaped = np.hstack((a,b,c))
        return reshaped


    def get_omega_0(self):

        solver = newtonRaphsonCap(timeStepSize=self.timeStepSize, numberOfMeasurements=self.numberOfMeasurements,
                                  startPotential=self.startPotential, revPotential=self.revPotential,
                                  rateOfPotentialChange=self.rateOfPotentialChange, inital_current=self.inital_current,
                                  freq=self.freq, deltaepislon=self.deltaepislon, electrode_area =self.electrode_area,
                                  electode_coverage=self.electode_coverage)

        return solver.omega0

    def get_non_dimensionality_constants(self):

        solver = newtonRaphsonCap(timeStepSize=self.timeStepSize, numberOfMeasurements=self.numberOfMeasurements,
                                  startPotential=self.startPotential, revPotential=self.revPotential,
                                  rateOfPotentialChange=self.rateOfPotentialChange, inital_current=self.inital_current,
                                  freq=self.freq, deltaepislon=self.deltaepislon, electrode_area =self.electrode_area,
                                  electode_coverage=self.electode_coverage)

        return [solver.E0, solver.T0, solver.I0]

def change_cap_params_to_same_as_paper(parameter_list):
    """
    docstring
    """
    params = np.asarray(parameter_list)
    
    C_dl = params[:,0]
    params[:,1] = params[:,1]/C_dl
    params[:,2] = params[:,2]/C_dl
    params[:,3] = params[:,3]/C_dl

    return params