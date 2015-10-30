import random
import numpy as np
from ggplot import *
import pandas as pd
import math
import operator

class NaiveBayes:

    SPAM = "SPAM"
    NON_SPAM = "NON_SPAM"
    DEFAULT_COVARIANCE = 0.001

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
        self.attribute_mean_matrix_by_class = dict()
        self.attribute_covariance_matrix_by_class = dict()
        self.class_prior_prob = dict()
        self.feature_likelihood_prob = dict()
        self.data_points_by_fold = dict()
        self.read_source_data()
        self.log_odds = dict()

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

    def form_diff_data_folds(self):

        for i in range(1, self.nbr_of_fold+1):
            self.data_points_by_fold[i-1] = range(i-1, self.nbr_of_data, 10)

    def cal_prior_prob(self):

        nbr_of_spam = 0
        nbr_of_non_spam = 0
        for index in range(0, self.nbr_of_data):

            temp_target_matrix = self.target_matrix.__getitem__(index)

            if temp_target_matrix.item((0, 0)) == 1:
                nbr_of_spam += 1
            else:
                nbr_of_non_spam += 1

        self.class_prior_prob[self.NON_SPAM] = float(nbr_of_non_spam + 1) /\
                                                   (self.nbr_of_data + 2)
        self.class_prior_prob[self.SPAM] = float(nbr_of_spam + 1) / \
                                           (self.nbr_of_data + 2)

    def fetch_train_test_data_by_fold(self, current_fold):

        self.train_attribute_matrix = None
        self.train_target_matrix = None
        self.test_attribute_matrix = None
        self.test_target_matrix = None

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

    def cal_train_mean_variance_by_class(self):

        train_spam_attribute_matrix = None
        train_non_spam_attribute_matrix = None

        nbr_of_train_data = self.train_attribute_matrix.shape[0]
        for index in range(0, nbr_of_train_data):

            target = self.train_target_matrix.__getitem__(index)
            if target.item((0, 0)) == 1:
                if train_spam_attribute_matrix is None:
                    train_spam_attribute_matrix = self.train_attribute_matrix.__getitem__(index)
                else:
                    train_spam_attribute_matrix = np.concatenate((train_spam_attribute_matrix,
                                                                  self.train_attribute_matrix.__getitem__(index)))
            else:
                if train_non_spam_attribute_matrix is None:
                    train_non_spam_attribute_matrix = self.train_attribute_matrix.__getitem__(index)
                else:
                    train_non_spam_attribute_matrix = np.concatenate((train_non_spam_attribute_matrix,
                                                                      self.train_attribute_matrix.__getitem__(index)))

        if train_spam_attribute_matrix is not None:

            self.attribute_mean_matrix_by_class[self.SPAM] = train_spam_attribute_matrix.mean(axis=0)

            covariance_matrix = np.cov(train_spam_attribute_matrix, rowvar=0)
            row = covariance_matrix.shape[0]
            column = covariance_matrix.shape[1]
            for i in range(0, row):
                for j in range(0, column):
                    if covariance_matrix[i][j] == 0:
                        covariance_matrix[i][j] = self.DEFAULT_COVARIANCE

            self.attribute_covariance_matrix_by_class[self.SPAM] = np.matrix(covariance_matrix)

        else:
            # Smoothing
            self.attribute_mean_matrix_by_class[self.SPAM] = np.matrix(np.zeros((1, self.nbr_of_attributes)))
            self.attribute_covariance_matrix_by_class[self.SPAM] = np.matrix(np.zeros((self.nbr_of_attributes,
                                                                                       self.nbr_of_attributes)).
                                                                             fill(self.DEFAULT_COVARIANCE))

        if train_non_spam_attribute_matrix is not None:

            self.attribute_mean_matrix_by_class[self.NON_SPAM] = train_non_spam_attribute_matrix.mean(axis=0)

            covariance_matrix = np.cov(train_non_spam_attribute_matrix, rowvar=0)
            row = covariance_matrix.shape[0]
            column = covariance_matrix.shape[1]
            for i in range(0, row):
                for j in range(0, column):
                    if covariance_matrix[i][j] == 0:
                        covariance_matrix[i][j] = self.DEFAULT_COVARIANCE

            self.attribute_covariance_matrix_by_class[self.NON_SPAM] = np.matrix(covariance_matrix)
        else:
            # Smoothing
            self.attribute_mean_matrix_by_class[self.NON_SPAM] = np.matrix(np.zeros((1, self.nbr_of_attributes)))
            self.attribute_covariance_matrix_by_class[self.NON_SPAM] = np.matrix(np.zeros((self.nbr_of_attributes,
                                                                                       self.nbr_of_attributes)).
                                                                             fill(self.DEFAULT_COVARIANCE))

    def cal_data_prob_gaussian(self, class_name, test_attribute):

        diff_matrix = test_attribute - self.attribute_mean_matrix_by_class[class_name]

        val = np.divide((diff_matrix *
                        np.linalg.pinv(self.attribute_covariance_matrix_by_class[class_name]) *
                        np.transpose(diff_matrix)), -2)

        determinant = np.linalg.det(self.attribute_covariance_matrix_by_class[class_name])

        prob = (math.pow(math.e, val.item((0, 0)))) / math.sqrt(abs(determinant))

        return prob

    def evaluate_model_on_test_data(self, current_fold):

        nbr_of_test_records = self.test_target_matrix.shape[0]

        nbr_of_false_positive = 0
        nbr_of_false_negative = 0
        nbr_of_true_positive = 0
        nbr_of_true_negative = 0

        for index in range(0, nbr_of_test_records):

            test_attribute = self.test_attribute_matrix.__getitem__(index)
            test_target = self.test_target_matrix.__getitem__(index)

            # Check the likelihood probability of the current test attribute for each of the classification class
            data_spam_likelihood_prob = self.class_prior_prob[self.SPAM] * \
                                   self.cal_data_prob_gaussian(self.SPAM, test_attribute)

            data_non_spam_likelihood_prob = self.class_prior_prob[self.NON_SPAM] * \
                                            self.cal_data_prob_gaussian(self.NON_SPAM, test_attribute)

            if current_fold == 0:
                if data_spam_likelihood_prob != 0 and data_non_spam_likelihood_prob != 0:
                    self.log_odds[index] = math.log(data_spam_likelihood_prob / data_non_spam_likelihood_prob)
                else:
                    self.log_odds[index] = 0

            if data_spam_likelihood_prob > data_non_spam_likelihood_prob:
                predicted_target = 1
            else:
                predicted_target = 0

            if int(test_target.item((0, 0))) == 1:

                if predicted_target == 1:
                    nbr_of_true_positive += 1
                else:
                    nbr_of_false_positive += 1
            else:

                if predicted_target == 0:
                    nbr_of_true_negative += 1
                else:
                    nbr_of_false_negative += 1

        print("Results")
        print("True Positive " + str(nbr_of_true_positive))
        print("False Positive " + str(nbr_of_false_positive))
        print("True Negative " + str(nbr_of_true_negative))
        print("False Negative " + str(nbr_of_false_negative))

        if current_fold == 0:

            nbr_of_false_positive = 0
            nbr_of_false_negative = 0
            nbr_of_true_positive = 0
            nbr_of_true_negative = 0

            true_positive_rate = list()
            false_positive_rate = list()

            sorted_log_odds = sorted(self.log_odds.items(), key=operator.itemgetter(1))

            for i in range(0, nbr_of_test_records):

                test_data_index = sorted_log_odds[i][0]
                test_data_log_vale = sorted_log_odds[i][1]

                actual_target_vale = self.test_target_matrix.__getitem__(test_data_index).item((0, 0))

                if test_data_log_vale > 0:

                    if actual_target_vale == 1:
                        nbr_of_true_positive += 1
                    else:
                        nbr_of_false_positive += 1
                else:

                    if actual_target_vale == 0:
                        nbr_of_true_negative += 1
                    else:
                        nbr_of_false_negative += 1

                print("True Positive " + str(nbr_of_true_positive))
                print("False Positive " + str(nbr_of_false_positive))
                print("True Negative " + str(nbr_of_true_negative))
                print("False Negative " + str(nbr_of_false_negative))

                if (nbr_of_true_positive + nbr_of_false_negative) != 0:
                    true_positive_rate.append(float(nbr_of_true_positive) / \
                                            (nbr_of_true_positive + nbr_of_false_negative))
                else:
                    true_positive_rate.append(0.0)

                if (nbr_of_false_positive + nbr_of_true_negative) != 0.0:
                    false_positive_rate.append(float(nbr_of_false_positive) / \
                                             (nbr_of_false_positive + nbr_of_true_negative))
                else:
                    false_positive_rate.append(0.0)
            true_positive_rate.append(1.0)
            false_positive_rate.append(1.0)

            df = pd.DataFrame(dict(tpr=true_positive_rate, fpr=false_positive_rate))
            print(true_positive_rate)
            print(false_positive_rate)
            g = ggplot(df, aes(x='fpr', y='tpr')) +\
                    geom_line() +\
                    geom_abline(linetype='dashed') + xlim(0, 1) + ylim(0, 1)
            print(g)

        accuracy = float(nbr_of_true_positive+nbr_of_true_negative) / nbr_of_test_records
        print("Accuracy " + str(accuracy))
        return accuracy

    def apply_naive_bayes_gaussian(self):

        self.form_diff_data_folds()
        self.cal_prior_prob()
        total_accuracy = 0.0

        for i in range(0, self.nbr_of_fold):

            # Calculate the require parameters for the GDA
            self.fetch_train_test_data_by_fold(i)
            self.cal_train_mean_variance_by_class()
            # Apply the models on Test Data set
            total_accuracy += self.evaluate_model_on_test_data(i)
        print("Average Accuracy")
        print(total_accuracy/self.nbr_of_fold)

if __name__ == "__main__":
    file_path = "/Users/Darshan/Documents/MachineLearningAlgorithms/GenerativeModels/data"
    file_name = "spambase.data"
    naive = NaiveBayes(file_path, file_name)
    naive.apply_naive_bayes_gaussian()
