import numpy as np
import random
import math
from sklearn.preprocessing import normalize


class EM:

    def __init__(self, file_path, file_name):
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
        self.converged = False
        self.read_source_data()
        self.random_initialization()

    def random_initialization(self):

        for i in range(0, self.nbr_of_models):
            self.mean_for_models.\
                append(np.matrix((random.random(), random.random())))
            self.covariance_for_models.append(np.matrix(np.random.rand(self.dim_of_data, self.dim_of_data)))
            self.coefficient_for_models.append(random.random())

        prob_matrix = np.matrix(np.random.rand(self.nbr_of_data, self.nbr_of_models))
        self.data_prob_matrix = normalize(prob_matrix, norm='l1', axis=1)
        self.coefficient_for_models = list(k/sum(self.coefficient_for_models)
                                           for k in self.coefficient_for_models)

    def read_source_data(self):
        """ This function reads the source data files and creates the data matrix"""
        file_loc = self.file_path+'/'+self.file_name
        data_list = list()
        with open(file_loc, "r") as source_file:
            for line in source_file:
                parts = line.replace("\n", "").split(" ")
                if len(parts) == self.dim_of_data:
                    self.nbr_of_data += 1
                    float_parts = []
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
        for i in range(0, self.nbr_of_models):
            print("Model number : " + str((i+1)))
            print("Model mean   : " + str(self.mean_for_models[i]))
            print("Model coefficient :" + str(self.coefficient_for_models[i]))
            print("Model covariance : " + str(self.covariance_for_models[i]))

        print("Data Prob Matrix")
        print(self.data_prob_matrix)

    def expectation_step(self):

        for i in range(0, self.nbr_of_data):
            # Calculate the probability of data point by the gaussian distribution
            temp_data_prob = []
            for j in range(0, self.nbr_of_models):

                prob = self.cal_data_prob_gaussian(i, j)
                temp_data_prob.append(prob*self.coefficient_for_models[j])
            temp_data_prob_matrix = np.matrix(temp_data_prob)
            self.data_prob_matrix.__setitem__(i, normalize(temp_data_prob_matrix, norm='l1', axis=1))

    def cal_data_prob_gaussian(self, data_nbr, model_nbr):

        diff_matrix = self.data_matrix.__getitem__(data_nbr) - self.mean_for_models[model_nbr]

        val = np.divide((diff_matrix *
                        np.linalg.pinv(self.covariance_for_models[model_nbr]) *
                        np.transpose(diff_matrix)), -2)
        determinant = np.linalg.det(self.covariance_for_models[model_nbr])
        prob = (math.pow(math.e, val.item((0, 0)))) / math.sqrt(abs(determinant))

        return prob

    def maximization_step(self):

        for i in range(0, self.nbr_of_models):
            prob_sum = 0.0
            mean_numerator_sum = 0.0
            covariance_numerator_sum = np.matrix(np.zeros((self.dim_of_data, self.dim_of_data)))

            for j in range(0, self.nbr_of_data):

                prob_val = self.data_prob_matrix.item((j, i))
                diff = self.data_matrix.__getitem__(j) - self.mean_for_models[i]
                prob_sum += prob_val
                mean_numerator_sum += np.multiply(self.data_matrix.__getitem__(j), prob_val)
                covariance_numerator_sum += np.multiply((np.transpose(diff) * diff), prob_val)

            self.covariance_for_models[i] = np.divide(covariance_numerator_sum, prob_sum)
            self.mean_for_models[i] = np.divide(mean_numerator_sum, prob_sum)
            self.coefficient_for_models[i] = prob_sum / self.nbr_of_data

    def run_em(self):
        count = 0

        while not self.converged:

            print("Cycle count : " + str(count))
            count += 1
            old_coefficient_for_models = self.coefficient_for_models
            old_covariance_for_models = self.covariance_for_models
            old_mean_for_models = self.mean_for_models
            self.expectation_step()
            self.maximization_step()
            self.print_algo_parameters()
            self.check_halting_cond(old_coefficient_for_models, old_covariance_for_models,
                                    old_mean_for_models, count)
        self.print_algo_parameters()

    def check_halting_cond(self, old_coefficient_for_models, old_covariance_for_models,
                           old_mean_for_models, count):

        total_coefficient_diff = 0.0
        total_covariance_diff = 0.0
        total_mean_diff = 0.0
        for i in range(0, self.nbr_of_models):
            total_coefficient_diff += abs(self.coefficient_for_models[i]-old_coefficient_for_models[i])
            total_covariance_diff += (self.covariance_for_models[i] - old_covariance_for_models[i]).max()
            total_mean_diff += (self.mean_for_models[i] - old_mean_for_models[i]).max()

        if count > 50:
            self.converged = True

if __name__ == "__main__":
    print('Hello World !')
    file_path = '/Users/Darshan/Documents/MachineLearningAlgorithms/GenerativeModels/data'
    file_name = '2gaussian.txt'
    em = EM(file_path, file_name)
    em.print_algo_parameters()
    em.run_em()
