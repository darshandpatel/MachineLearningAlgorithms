import numpy as np
import random
import math


class EM:

    def __init__(self,file_path,file_name):
        self.file_name = file_name
        self.file_path = file_path
        self.data_matrix = None
        self.data_prob_matrix = None
        self.nbr_of_models = 2
        self.nbr_of_data = 0
        self.dim_of_data = 2
        self.mean_for_models = list()
        self.covariance_for_models = list()
        self.coefficient_for_models = list()
        self.random_initialization()

    def random_initialization(self):

        for i in range(0,self.nbr_of_models):
            self.mean_for_models.\
                append(np.matrix((random.randrange(2,6),random.randrange(2,6))))
            self.covariance_for_models.append(np.matrix(np.random.rand(2,2)))
            self.coefficient_for_models.append(random.random())
        self.self.data_prob_matrix = np.matrix(np.random.random(self.nbr_of_data,self.nbr_of_data))
        self.coefficient_for_models = [i/sum(self.coefficient_for_models)
                                       for i in self.coefficient_for_models]

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

    def print_algo_parameters(self):

        print("Model parameters")
        for i in range(0,self.nbr_of_models):
            print("Model number : "+ str((i+1)))
            print("Model mean   : "+ str(self.mean_for_models[i]))
            print("Model coefficient :"+ str(self.coefficient_for_models[i]))
            print("Model covariance : "+ str(self.covariance_for_models[i]))

    def expectation_step(self):

        for i in range(0,self.nbr_of_data):
            # Calculate the probability of data point by the gaussian distribution

            temp_data_prob = list()

            for m in range(0,self.nbr_of_models):
                prob = self.cal_data_prob_gaussian(m,i)
                temp_data_prob.append(prob*self.coefficient_for_models[m])

            self.data_prob_matrix[i] = [i/sum(temp_data_prob)
                                       for i in temp_data_prob]




    def cal_data_prob_gaussian(self,model_nbr,data_nbr):

        val = (-1/2) * (np.transpose(self.data_matrix[data_nbr] - self.mean_for_models[model_nbr]) * \
            np.linalg.inv(self.covariance_for_models) * \
            (self.data_matrix[data_nbr] - self.mean_for_models[model_nbr]))

        prob = math.pow((2*math.pi),-(self.nbr_of_models/2)) * \
            (math.pow(np.linalg.slogdet(self.covariance_for_models[model_nbr]),-1/2)) * \
            (math.pow(math.e,val))

        return prob






if __name__ == "__main__":
    print('Hello World !')
    file_path = '/Users/Darshan/Documents/MachineLearningAlgorithms/GenerativeModels/data'
    file_name = '2gaussian.txt'
    em = EM(file_path,file_name)
    em.read_source_data()
    #em.print_data_matrix()
    em.print_algo_parameters()
