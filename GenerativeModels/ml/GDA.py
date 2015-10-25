import numpy as np


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

        for i in range(1,self.nbr_of_fold + 1):

            # Calculate the require parameters for the GDA
            self.fetch_train_test_data_by_fold(i)
            self.cal_mean_by_class()
            break
            # Apply the models on Test Data set

    def fetch_train_test_data_by_fold(self, current_fold):

        nbr_of_dp_per_fold = int(self.nbr_of_data / self.nbr_of_fold)

        train_extra_dp = (self.nbr_of_data % self.nbr_of_fold) if current_fold == self.nbr_of_fold else 0

        train_data_start_index = (current_fold - 1) * nbr_of_dp_per_fold
        train_data_end_index = train_data_start_index + nbr_of_dp_per_fold + train_extra_dp

        self.train_attribute_matrix = self.attribute_matrix.__getslice__(train_data_start_index, train_data_end_index)
        self.train_target_matrix = self.target_matrix.__getslice__(train_data_start_index, train_data_end_index)

        if current_fold != 1:

            one_half_test_attribute_matrix = self.attribute_matrix.__getslice__(0, train_data_start_index)
            second_half_test_attribute_matrix = self.attribute_matrix.__getslice__(train_data_end_index, self.nbr_of_data)
            self.test_attribute_matrix = np.concatenate((one_half_test_attribute_matrix, second_half_test_attribute_matrix))

            one_half_test_target_matrix = self.target_matrix.__getslice__(0, train_data_start_index)
            second_half_test_target_matrix = self.target_matrix.__getslice__(train_data_end_index, self.nbr_of_data)
            self.test_target_matrix = np.concatenate((one_half_test_target_matrix, second_half_test_target_matrix))

        else:

            self.test_attribute_matrix = self.attribute_matrix.__getslice__(train_data_end_index, self.nbr_of_data)
            self.test_target_matrix = self.target_matrix.__getslice__(train_data_end_index, self.nbr_of_data)

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
        self.theta_by_class[self.SPAM] = (spam_attribute_matrix.shape[0] + 1) / \
                                         (self.train_data_matrix.shape[0] + self.nbr_of_class)
        self.theta_by_class[self.NON_SPAM] = (non_spam_attribute_matrix.shape[0] + 1) / \
                                             (self.train_data_matrix.shape[0] + self.nbr_of_class)

if __name__ == "__main__":
    file_path = "/Users/Darshan/Documents/MachineLearningAlgorithms/GenerativeModels/data"
    file_name = "spambase.data"
    gda = GDA(file_path, file_name)
    gda.apply_gda()
