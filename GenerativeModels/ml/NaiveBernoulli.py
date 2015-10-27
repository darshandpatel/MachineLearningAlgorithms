import random
import numpy as np
import math


class Naive:

    SPAM = "SPAM"
    NON_SPAM = "NON_SPAM"
    ABOVE_MEAN = "ABOVE_MEAN"
    BELOW_MEAN = "BELOW_MEAN"

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
        #self.covariance_matrix = None
        self.read_source_data()
        #self.cal_covariance()
        self.attribute_mean_matrix = self.attribute_matrix.mean(axis=0)
        #self.mean_matrix_by_class = dict()
        #self.cal_overall_mean_by_class()
        self.feature_likelilood_prob = dict()
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

    def cal_overall_mean_by_class(self):

        spam_attribute_matrix = None
        non_spam_attribute_matrix = None

        for index in range(0, self.nbr_of_data):

            target = self.train_target_matrix.__getitem__(index)
            if target.item((0, 0)) == 1:
                if spam_attribute_matrix is None:
                    spam_attribute_matrix = self.attribute_matrix.__getitem__(index)
                else:
                    spam_attribute_matrix = np.concatenate((spam_attribute_matrix,
                                                            self.attribute_matrix.__getitem__(index)))
            else:
                if non_spam_attribute_matrix is None:
                    non_spam_attribute_matrix = self.attribute_matrix.__getitem__(index)
                else:
                    non_spam_attribute_matrix = np.concatenate((non_spam_attribute_matrix,
                                                                self.attribute_matrix.__getitem__(index)))

        if spam_attribute_matrix is not None:
            self.mean_matrix_by_class[self.SPAM] = spam_attribute_matrix.mean(axis=0)
        else:
            self.mean_matrix_by_class[self.SPAM] = np.matrix(np.zeros((1, self.nbr_of_attributes)))

        if non_spam_attribute_matrix is not None:
            self.mean_matrix_by_class[self.NON_SPAM] = non_spam_attribute_matrix.mean(axis=0)
        else:
            self.mean_matrix_by_class[self.NON_SPAM] = np.matrix(np.zeros((1, self.nbr_of_attributes)))

    def cal_feature_likelihood_prob(self):

        spam_above_mean_attr_value_count = np.zeros((1, self.nbr_of_attributes))
        spam_below_mean_attr_value_count = np.zeros((1, self.nbr_of_attributes))

        non_spam_above_mean_attr_value_count = np.zeros((1, self.nbr_of_attributes))
        non_spam_below_mean_attr_value_count = np.zeros((1, self.nbr_of_attributes))

        total_spam_data = 0
        total_non_spam_data = 0

        for index in range(0, self.train_target_matrix.shape[0]):

            target = self.train_target_matrix.__getitem__(index)

            if target.item((0, 0)) == 1:

                total_spam_data += 1

                for attribute_index in range(0, self.nbr_of_attributes):
                    attr_val = self.train_attribute_matrix.__getitem__(index).\
                        item((0, attribute_index))

                    if attr_val > self.attribute_mean_matrix.item((0, attribute_index)):
                        spam_above_mean_attr_value_count[attribute_index] += 1
                    else:
                        spam_below_mean_attr_value_count[attribute_index] += 1

            else:

                total_non_spam_data += 1

                for attribute_index in range(0, self.nbr_of_attributes):
                    attr_val = self.train_attribute_matrix.__getitem__(index).\
                        item((0, attribute_index))

                    if attr_val > self.attribute_mean_matrix.item((0, attribute_index)):
                        non_spam_above_mean_attr_value_count[attribute_index] += 1
                    else:
                        non_spam_below_mean_attr_value_count[attribute_index] += 1

        # Laplace Smoothing

        for index in range(0, self.nbr_of_attributes):

            feature_prob = dict()

            spam_above_mean_prob = float(spam_above_mean_attr_value_count[index] + 1) /\
                                   (total_spam_data + 2)
            spam_below_mean_prob = float(non_spam_below_mean_attr_value_count[index] + 1) / \
                                   (total_spam_data + 2)
            non_spam_above_mean_prob = float(spam_above_mean_attr_value_count[index] + 1) / \
                                       (total_non_spam_data + 2)
            non_spam_below_mean_prob = float(non_spam_below_mean_attr_value_count[index] + 1) / \
                                       (total_non_spam_data + 2)

            feature_prob[self.ABOVE_MEAN] = {self.SPAM: spam_above_mean_prob,
                                             self.NON_SPAM: non_spam_above_mean_prob}
            feature_prob[self.BELOW_MEAN] = {self.SPAM: spam_below_mean_prob,
                                             self.NON_SPAM: non_spam_below_mean_prob}

            self.feature_likelilood_prob[index] = feature_prob

    def apply_naive_bernoulli(self):

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
