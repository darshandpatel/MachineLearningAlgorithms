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
        self.mean_for_models = []
        self.covariance_for_models = []
        self.coefficient_for_models = []
        self.read_source_data()
        self.random_initialization()

    def random_initialization(self):

        for i in range(0,self.nbr_of_models):
            self.mean_for_models.\
                append(np.array((random.randrange(2,6),random.randrange(2,6))))
            self.covariance_for_models.append(np.array(np.random.rand(2,2)))
            self.coefficient_for_models.append(random.random())
        self.data_prob_matrix = np.random.rand(self.nbr_of_data,self.nbr_of_models)
        self.coefficient_for_models = np.asarray(list(k/sum(self.coefficient_for_models)
                                        for k in self.coefficient_for_models))

    def read_source_data(self):
        """ This function reads the source data files and creates the data matrix"""
        file_loc = self.file_path+'/'+self.file_name
        self.data_matrix = []
        with open(file_loc,"r") as source_file:
            for line in source_file:
                parts = line.replace("\n","").split(" ")
                if len(parts) == 2:
                    self.nbr_of_data += 1
                    float_parts = []
                    for value in parts:
                        float_parts.append(float(value))
                    self.data_matrix.append(float_parts)
        print("# of data points : "+ str(self.nbr_of_data))


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

        for i in range(0,self.nbr_of_data-1):
            # Calculate the probability of data point by the gaussian distribution

            temp_data_prob = []

            print("Data # " + str(i))
            for m in range(0,self.nbr_of_models):
                print("Model # " + str(m))
                prob = self.cal_data_prob_gaussian(m,i)
                temp_data_prob.append(prob*self.coefficient_for_models[m])

            sum_value = sum(temp_data_prob)
            for v in range(0,len(temp_data_prob)):
                self.data_prob_matrix[i][v] = temp_data_prob[v]/sum_value

    def cal_data_prob_gaussian(self,model_nbr,data_nbr):

        diff_matrix = self.data_matrix[data_nbr] - self.mean_for_models[model_nbr]

        val = (-1/2) * (diff_matrix * \
                        np.linalg.pinv(self.covariance_for_models[model_nbr]) * \
                        np.transpose(diff_matrix))
        determinant = np.linalg.det(self.covariance_for_models[model_nbr])
        prob = (math.pow(2*math.pi,(self.nbr_of_models/-2.0)) * (math.pow(math.e,val[0][0]))) \
               / math.sqrt(abs(determinant))

        return prob

    def maximization_step(self):

        for i in range(0,self.nbr_of_models):

            print("# of model : " + str(i))
            data_prob_sum = 0
            mean_numerator_sum = 0
            covariance_numerator_sum = 0

            for j in range(0,self.nbr_of_data):

                print("# of data : " + str(j))

                prob_val = self.data_prob_matrix[j][i]
                diff = (self.data_matrix[j] - self.mean_for_models[i])

                data_prob_sum += prob_val
                mean_numerator_sum += np.multiply(self.data_matrix[j],prob_val)
                covariance_numerator_sum += np.multiply((diff * np.transpose(diff)),prob_val)

            self.mean_for_models[i] = mean_numerator_sum/data_prob_sum
            self.data_prob_sum[i] = np.divide(self.nbr_of_data,data_prob_sum)
            self.covariance_numerator_sum[i] = covariance_numerator_sum/data_prob_sum

    def run_em(self):
        count = 0
        while(count < 100):
            print("Cycle count : " + str(count))
            count+=1
            self.expectation_step()
            self.maximization_step()
            em.print_algo_parameters()

if __name__ == "__main__":
    print('Hello World !')
    file_path = '/Users/Darshan/Documents/MachineLearningAlgorithms/GenerativeModels/data'
    file_name = '2gaussian.txt'
    em = EM(file_path,file_name)
    #em.print_algo_parameters()
    em.run_em()
