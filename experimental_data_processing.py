import numpy as np
import pints
import pandas as pd
import csv

class processingExperimentalData():

    def __init__(self, input_file_path: str, output_folder_path: str):

        self.input_file_path = input_file_path
        self.output_folder_path= output_folder_path
        self.data= pd.read_csv(input_file_path, sep='\t')
        print('columnsnames: ', self.data.columns)

    def reducing_data(self, data, skip_this_many):
        """downsamplingthe number of measurements in the given dataframe
        by skipping every 'skip_this_many' measurements

        Args:
            data (dataframe): Experimental data
            skip_this_many (int): factor to downsmaple by

        Returns:
            [type]: [description]
        """
        
        data_copy = data.copy()
        X = int(skip_this_many)
        reduced_data = data_copy.iloc[::X]
        return reduced_data

    def save_data(self, data):
        """save the given data to a file

        Args:
            data (dataframe float64): dataframe for saving
        """
        col = list(data.columns)

        data.to_csv(self.output_folder_path, header = col, index=None, sep='\t', mode='w')

    def how_many_periods(self, data, freq):
        """Calculates the number of periods of AC input spaned by the data

        Args:
            data (dataframe): Experimental data
            freq (float): frequency of the AC input used in the experiment

        Returns:
            float: The number of periods of AC input spaned by the data
        """

        length = data.shape
        index = length[0]-1
        times = data.time
        last_time = times[index]
        
        periods = last_time*freq

        return periods

    def suggested_cut_for_periods(self, data, freq):
        """Calculates the number of measurements from the data that is closest to 
            an integer number of periods

        Args:
            data (dataframe): Experimental data
            freq (float): frequency of the AC input used in the experiment

        Returns:
            float: The number of periods of AC input spaned by the data
        """
        untouched_periods = self.how_many_periods(data,freq)

        length = data.shape[0]
        
        measurements_per_period = length/untouched_periods

        print('measurements_per_period ~ ', measurements_per_period)

        int_periods = int(untouched_periods)

        updated_periods = untouched_periods.copy()
        updated = data.copy()
        N=0

        # FIXME: what if difference less than 10?
        n0 = int(((untouched_periods - int_periods)*measurements_per_period) -10.0)

        while updated_periods > int_periods:
            # Number of rows to drop 
            n = 1
            N = N+n
            
            if N == 1:
                N = N+n0
                # Dropping last n0 rows using drop 
                updated.drop(updated.tail(n0).index, inplace = True) 

            else:
                # Dropping last n rows using drop 
                updated.drop(updated.tail(n).index, inplace = True) 

            # calculating number of periods in reduced data size

            updated_periods = self.how_many_periods(updated,freq)
            print('rows dropped: '+ str(N) +' updated_periods: ', updated_periods)
        
        N = N - 1
        return N

    def cut_for_periods(self, data, drop_rows):
        """removes the given number of rows from the dataframe

        Args:
            data (dataframe): Experimental data
            drop_rows (int): Number of rows to drop

        Returns:
            data_frame: dataframe with dropped rows
        """

        data_copy = data.copy()
        data_copy.drop(data_copy.tail(drop_rows).index, inplace = True)

        return data_copy

    def suggested_measurements_reduction(self, data, freq):
        """caculates the factors pairs of the data and the 
           number of measurements corresponding to 200 per period

        Args:
            data (dataframe): Experimental data
            freq (float): frequency of the AC input used in the experiment
        """

        length = data.shape
        length = length[0]-1

        periods_in_data = self.how_many_periods(data,freq)
        factors = self.__paired_factors(length)

        print('\n'+'*'*20+'\npossible downsampling factors:\n', factors)

        print('\n total measuremnts for 200 measurements per period: ', periods_in_data*200,'\n' + '*'*20)

    def __paired_factors(self, value):
        """caculates pairs of factors for a given value

        Args:
            value (int): value to be factorised

        Returns:
            list: list containing tuple pairs of factors of the value
        """
        factors = []
        for i in range(1, int(value**0.5)+1):
            if value % i == 0:
                factors.append((i, value / i))
        return factors

