import numpy as np
import random


class EM:

    def __init__(self,file_path,file_name):
        self.file_name = file_name
        self.file_path = file_path
        self.data_matrix = None
        self.data_prob_matrix = None
        self.nbr_of_models = 2
        self.nbr_of_data = 0
        self.dim_of_data = 2
        self.sigma_for_models = list()
        self.covariance_for_models = list()
        self.coefficient_for_models = list()
        self.random_initialization()

    def random_initialization(self):

        temp_coefficient_for_models = list()
        for i in range(0,self.nbr_of_models):
            self.sigma_for_models[i] = random.randrange(2,6)
            self.covariance_for_models[i] = np.random.rand(2,2)
            self.coefficient_for_models[i] = random.random()

        self.coefficient_for_models = self.normalize(self.coefficient_for_models)

    def normalize(lst):
        s = sum(lst)
        return map(lambda x: float(x)/s, lst)

    def read_source_data(self):
        """ This function reads the source data files and creates the data matrix"""
        file_loc = self.file_path+'/'+self.file_name
        data_list = list()
        with open(file_loc,"r") as source_file:
            for line in source_file:
                if line != "":
                    self.nbr_of_data += 1
                    parts = line.replace("\n","").split(" ")
                    float_parts = list()
                    for value in parts:
                        float_parts.append(float(value))
                    data_list.append(float_parts)
        self.data_matrix = np.matrix(data_list)

    def print_data_matrix(self):
        """This function prints the data matrix"""
        for data in self.data_matrix:
            print(data)

    '''
    def expectation_step(self):

        for i in range(0,self.nbr_of_data):
            for m in range(0,self.nbr_of_models):
    '''





if __name__ == "__main__":
    print('Hello World !')
    file_path = '/Users/Darshan/Documents/MachineLearningAlgorithms/GenerativeModels/data'
    file_name = '2gaussian.txt'
    em = EM(file_path,file_name)
    em.read_source_data()
    em.print_data_matrix()