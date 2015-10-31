import random
import numpy as np
from ggplot import *
import pandas as pd
import math
import operator

class NaiveBayes:

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
        self.nbr_of_bins = 4
        self.nbr_of_class = 2
        self.attribute_mean_matrix_by_class = dict()
        self.class_prior_prob = dict()
        self.feature_likelihood_prob = dict()
        self.data_points_by_fold = dict()
        self.read_source_data()
        self.log_odds = dict()
        self.bin_values_per_feature = dict()

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

    def form_feature_bins(self):

        spam_attribute_matrix = None
        non_spam_attribute_matrix = None

        for index in range(0, self.nbr_of_data):

            target = self.target_matrix.__getitem__(index)
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
            self.attribute_mean_matrix_by_class[self.SPAM] = spam_attribute_matrix.mean(axis=0)
            self.class_prior_prob[self.SPAM] = float(spam_attribute_matrix.shape[0] + 1) /\
                                               (self.target_matrix.shape[0] + 2)
        else:
            self.attribute_mean_matrix_by_class[self.SPAM] = np.matrix(np.zeros((1, self.nbr_of_attributes)))
            self.class_prior_prob[self.SPAM] = 0.5

        if non_spam_attribute_matrix is not None:
            self.attribute_mean_matrix_by_class[self.NON_SPAM] = non_spam_attribute_matrix.mean(axis=0)
            self.class_prior_prob[self.NON_SPAM] = float(non_spam_attribute_matrix.shape[0] + 1) /\
                                                   (self.target_matrix.shape[0] + 2)
        else:
            self.attribute_mean_matrix_by_class[self.NON_SPAM] = np.matrix(np.zeros((1, self.nbr_of_attributes)))
            self.class_prior_prob[self.NON_SPAM] = 0.5

        overall_attribute_mean_matrix = self.attribute_matrix.mean(axis=0)
        overall_attribute_min_matrix = self.attribute_matrix.min(axis=0)
        overall_attribute_max_matrix = self.attribute_matrix.max(axis=0)

        for index in range(0, self.nbr_of_attributes):

            bin_value_list = list()
            bin_value_list.append(self.attribute_mean_matrix_by_class[self.SPAM].item((0, index)))
            bin_value_list.append(self.attribute_mean_matrix_by_class[self.NON_SPAM].item((0, index)))
            bin_value_list.append(overall_attribute_mean_matrix.item((0, index)))
            bin_value_list.append(overall_attribute_min_matrix.item((0, index)))
            bin_value_list.append(overall_attribute_max_matrix.item((0, index)))
            bin_value_list.sort()
            self.bin_values_per_feature[index] = bin_value_list

    def cal_feature_likelihood_prob(self):

        spam_attr_bin_value_count = np.zeros((self.nbr_of_attributes,
                                              self.nbr_of_bins))

        non_spam_attr_bin_value_count = np.zeros((self.nbr_of_attributes,
                                                  self.nbr_of_bins))

        total_spam_data = 0
        total_non_spam_data = 0
        nbr_of_training_data = self.train_target_matrix.shape[0]

        for index in range(0, nbr_of_training_data):

            target = self.train_target_matrix.__getitem__(index)
            attribute = self.train_attribute_matrix.__getitem__(index)

            for attribute_index in range(0, self.nbr_of_attributes):

                attr_val = attribute.item((0, attribute_index))
                bin_index = self.cal_bin_index(attribute_index, attr_val)
                if target.item((0, 0)) == 1:
                    total_spam_data += 1
                    spam_attr_bin_value_count[attribute_index][bin_index] += 1
                else:
                    total_non_spam_data += 1
                    non_spam_attr_bin_value_count[attribute_index][bin_index] += 1

        # Laplace Smoothing

        for index in range(0, self.nbr_of_attributes):

            feature_prob_per_bin = dict()
            spam_feature_prob_per_bin = dict()
            non_spam_feature_prob_per_bin = dict()

            for i in range(0, self.nbr_of_bins):

                spam_feature_prob_per_bin[i] = (spam_attr_bin_value_count[index][i] + 1) / \
                                               (total_spam_data + 2)
                non_spam_feature_prob_per_bin[i] = (non_spam_attr_bin_value_count[index][i] + 1) / \
                                                   (total_non_spam_data + 2)

            feature_prob_per_bin[self.SPAM] = spam_feature_prob_per_bin
            feature_prob_per_bin[self.NON_SPAM] = non_spam_feature_prob_per_bin

            self.feature_likelihood_prob[index] = feature_prob_per_bin

    def cal_bin_index(self, attribute_index, attr_val):
        bin_value_list = self.bin_values_per_feature[attribute_index]
        bin_index = None
        for i in range(0, self.nbr_of_bins):
            if i == 0:
                if (bin_value_list[i] <= attr_val) and (attr_val <= bin_value_list[i+1]):
                    bin_index = i
                    break
            else:
                if (bin_value_list[i] < attr_val) and (attr_val <= bin_value_list[i+1]):
                    bin_index = i
                    break
        return bin_index

    def cal_likelihood_of_data(self, class_name, attributes):

        likelihood_prob = 1.0

        for index in range(0, self.nbr_of_attributes):

            attr_val = attributes.item((0, index))
            bin_index = self.cal_bin_index(index, attr_val)
            likelihood_prob *= self.feature_likelihood_prob[index][class_name][bin_index]

        return likelihood_prob

    def evaluate_model_on_test_data(self, current_fold):

        nbr_of_test_records = self.test_target_matrix.shape[0]

        nbr_of_false_positive = 0
        nbr_of_false_negative = 0
        nbr_of_true_positive = 0
        nbr_of_true_negative = 0

        for index in range(0, nbr_of_test_records):

            predicted_target = None
            test_attribute = self.test_attribute_matrix.__getitem__(index)
            test_target = self.test_target_matrix.__getitem__(index)

            # Check the likelihood probability of the current test attribute for each of the classification class
            data_spam_likelihood_prob = self.class_prior_prob[self.SPAM] * \
                                   self.cal_likelihood_of_data(self.SPAM, test_attribute)

            data_non_spam_likelihood_prob = self.class_prior_prob[self.NON_SPAM] * \
                                            self.cal_likelihood_of_data(self.NON_SPAM, test_attribute)

            if current_fold == 1000:
                self.log_odds[index] = math.log(data_spam_likelihood_prob / data_non_spam_likelihood_prob)

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

        accuracy = float(nbr_of_true_positive+nbr_of_true_negative) / nbr_of_test_records
        print("Accuracy " + str(accuracy))
        return accuracy

    def draw_roc(self):

        nbr_of_false_positive = 0
        nbr_of_false_negative = 0
        nbr_of_true_positive = 0
        nbr_of_true_negative = 0

        true_positive_rate = list()
        false_positive_rate = list()

        sorted_log_odds = sorted(self.log_odds.items(), key=operator.itemgetter(1))
        nbr_of_test_records = self.test_target_matrix.shape[0]

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
            true_positive_rate.append(1)

            if (nbr_of_false_positive + nbr_of_true_negative) != 0.0:
                false_positive_rate.append(float(nbr_of_false_positive) / \
                                         (nbr_of_false_positive + nbr_of_true_negative))
            else:
                false_positive_rate.append(0.0)
            false_positive_rate.append(1)

        df = pd.DataFrame(dict(tpr=true_positive_rate, fpr=false_positive_rate))
        print(true_positive_rate)
        print(false_positive_rate)
        g = ggplot(df, aes(x='fpr', y='tpr')) +\
                geom_line() +\
                geom_abline(linetype='dashed') + xlim(0, 1) + ylim(0, 1)
        print(g)

    def apply_naive_bernoulli(self):

        self.form_diff_data_folds()
        self.form_feature_bins()
        total_accuracy = 0.0

        for i in range(0, self.nbr_of_fold):

            # Calculate the require parameters for the GDA
            self.fetch_train_test_data_by_fold(i)
            self.cal_feature_likelihood_prob()
            # Apply the models on Test Data set
            total_accuracy += self.evaluate_model_on_test_data(i)
        print("Average Accuracy")
        print(total_accuracy/self.nbr_of_fold)

if __name__ == "__main__":
    file_path = "/Users/Darshan/Documents/MachineLearningAlgorithms/GenerativeModels/data"
    file_name = "spambase.data"
    naive = NaiveBayes(file_path, file_name)
    naive.apply_naive_bernoulli()
