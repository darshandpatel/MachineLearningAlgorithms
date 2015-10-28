import numpy as np
import random
import math


class GDA:

    SPAM = "SPAM"
    NON_SPAM = "NON_SPAM"

    def __init__(self, file_path, file_name):
        self.file_path = file_path
        self.file_name = file_name
        self.attribute_matrix = None
        self.target_matrix = None
        self.train_attribute_matrix = None
        self.test_attribute_matrix = None
        self.train_target_matrix = None
        self.test_target_matrix = None
        self.nbr_of_fold = 10
        self.nbr_of_data = 0
        self.nbr_of_attributes = 57
        self.nbr_of_class = 2
        self.covariance_matrix = None
        self.read_source_data()
        self.cal_covariance()
        self.mean_matrix_by_class = dict()
        self.theta_by_class = dict()
        self.data_points_by_fold = dict()

    def read_source_data(self):
        """ This function reads the source data files and creates the data matrix"""
        file_loc = self.file_path+'/'+self.file_name
        attribute_list = list()
        target_list = list()
        with open(file_loc, "r") as source_file:
            for line in source_file:
                parts = line.split(",")
                if len(parts) > 2:
                    self.nbr_of_data += 1
                    float_parts = []
                    nbr_of_parts = len(parts)
                    for i in range(0,nbr_of_parts-1):
                        float_parts.append(float(parts[i]))
                    target_list.append([float(parts[nbr_of_parts-1])])
                    attribute_list.append(float_parts)
        self.attribute_matrix = np.matrix(attribute_list)
        self.target_matrix = np.matrix(target_list)

    def cal_covariance(self):
        self.covariance_matrix = np.cov(self.attribute_matrix, rowvar=0)

    def apply_gda(self):

        total_accuracy = 0.0
        self.form_diff_data_folds()
        for i in range(0, self.nbr_of_fold):

            # Calculate the require parameters for the GDA
            self.fetch_train_test_data_by_fold(i)
            self.cal_mean_by_class()
            # Apply the models on Test Data set
            total_accuracy += self.evaluate_model_on_test_data()

        print("Total Accuracy ")
        print(float(total_accuracy)/self.nbr_of_fold)

    def form_diff_data_folds(self):

        index_list = range(0, self.nbr_of_data)
        random.shuffle(index_list)
        nbr_of_dp_per_fold = int(self.nbr_of_data / self.nbr_of_fold)

        for i in range(0, self.nbr_of_fold):

            if i != (self.nbr_of_fold - 1):
                start_index = nbr_of_dp_per_fold * i
                self.data_points_by_fold[i] = index_list[start_index:(start_index+nbr_of_dp_per_fold)]
            else:
                start_index = nbr_of_dp_per_fold * i
                self.data_points_by_fold[i] = index_list[start_index:self.nbr_of_data]

    def fetch_train_test_data_by_fold(self,current_fold):

        self.train_attribute_matrix = None
        self.train_target_matrix = None
        self.test_attribute_matrix = None
        self.test_target_matrix = None

        index_list = range(0, self.nbr_of_data)
        random.shuffle(index_list)
        nbr_of_dp_per_fold = int(self.nbr_of_data / self.nbr_of_fold)

        for i in range(0, self.nbr_of_fold):

            if i != current_fold:
                index = self.data_points_by_fold[i]
                if self.train_attribute_matrix is None:
                    self.train_attribute_matrix = self.attribute_matrix.__getitem__(index)
                    self.train_target_matrix = self.target_matrix.__getitem__(index)
                else:
                    self.train_attribute_matrix = np.concatenate((self.train_attribute_matrix,
                                                                  self.attribute_matrix.__getitem__(index)))
                    self.train_target_matrix = np.concatenate((self.train_target_matrix,
                                                               self.target_matrix.__getitem__(index)))
            else:
                index = self.data_points_by_fold[i]
                if self.test_attribute_matrix is None:
                    self.test_attribute_matrix = self.attribute_matrix.__getitem__(index)
                    self.test_target_matrix = self.target_matrix.__getitem__(index)
                else:
                    self.test_attribute_matrix = np.concatenate((self.test_attribute_matrix,
                                                                 self.attribute_matrix.__getitem__(index)))
                    self.test_target_matrix = np.concatenate((self.test_target_matrix,
                                                              self.target_matrix.__getitem__(index)))

    def cal_mean_by_class(self):

        spam_attribute_matrix = None
        non_spam_attribute_matrix = None

        for index in range(0, self.train_target_matrix.shape[0]):
            target = self.train_target_matrix.__getitem__(index)
            if target.item((0, 0)) == 1:
                if spam_attribute_matrix is None:
                    spam_attribute_matrix = self.train_attribute_matrix.__getitem__(index)
                else:
                    spam_attribute_matrix = np.concatenate((spam_attribute_matrix,
                                                            self.train_attribute_matrix.__getitem__(index)))
            else:
                if non_spam_attribute_matrix is None:
                    non_spam_attribute_matrix = self.train_attribute_matrix.__getitem__(index)
                else:
                    non_spam_attribute_matrix = np.concatenate((non_spam_attribute_matrix,
                                                                self.train_attribute_matrix.__getitem__(index)))

        if spam_attribute_matrix is not None:
            self.mean_matrix_by_class[self.SPAM] = spam_attribute_matrix.mean(axis=0)
        else:
            self.mean_matrix_by_class[self.SPAM] = np.matrix(np.zeros((1, self.nbr_of_attributes)))

        if non_spam_attribute_matrix is not None:
            self.mean_matrix_by_class[self.NON_SPAM] = non_spam_attribute_matrix.mean(axis=0)
        else:
            self.mean_matrix_by_class[self.NON_SPAM] = np.matrix(np.zeros((1, self.nbr_of_attributes)))

        # Laplace Smoothing
        self.theta_by_class[self.SPAM] = float(spam_attribute_matrix.shape[0] + 1) / \
                                         (self.train_attribute_matrix.shape[0] + self.nbr_of_class)
        self.theta_by_class[self.NON_SPAM] = float(non_spam_attribute_matrix.shape[0] + 1) / \
                                             (self.train_attribute_matrix.shape[0] + self.nbr_of_class)

    def evaluate_model_on_test_data(self):

        nbr_of_test_records = self.test_target_matrix.shape[0]

        covariance_inverse = np.linalg.pinv(self.covariance_matrix)
        covariance_determinant = np.linalg.det(self.covariance_matrix)

        nbr_of_accurate_predict = 0

        for index in range(0,nbr_of_test_records):

            predicted_target = None
            test_attribute = self.test_attribute_matrix.__getitem__(index)
            test_target = self.test_target_matrix.__getitem__(index)

            # Check the likelihood probability of the current test attribute for each of the classification class
            spam_likelihood_prob = self.cal_likelihood_of_data(self.SPAM, test_attribute,
                                                               covariance_inverse, covariance_determinant)

            non_spam_likelihood_prob = self.cal_likelihood_of_data(self.NON_SPAM, test_attribute,
                                                                   covariance_inverse, covariance_determinant)

            if float(self.theta_by_class[self.NON_SPAM] * non_spam_likelihood_prob) > \
                    float(self.theta_by_class[self.SPAM] * spam_likelihood_prob):
                predicted_target = 0
            else:
                predicted_target = 1

            if int(test_target.item((0, 0))) == predicted_target:
                nbr_of_accurate_predict += 1

        print("Accuracy : ")
        print(float(nbr_of_accurate_predict)/nbr_of_test_records)

        return float(nbr_of_accurate_predict)/nbr_of_test_records

    def cal_likelihood_of_data(self, class_name, test_attribute,
                               covariance_inverse, covariance_determinant):

        diff_matrix = test_attribute - self.mean_matrix_by_class[class_name]
        val = np.divide((diff_matrix * covariance_inverse * np.transpose(diff_matrix)),
                        -2)
        prob = math.pow(math.e, val.item((0, 0)))
        return prob


if __name__ == "__main__":
    file_path = "/Users/Darshan/Documents/MachineLearningAlgorithms/GenerativeModels/data"
    file_name = "spambase.data"
    gda = GDA(file_path, file_name)
    gda.apply_gda()
