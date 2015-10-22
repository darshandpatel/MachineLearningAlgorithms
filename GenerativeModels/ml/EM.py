import numpy as np
class EM:

    def __init__(self,file_path,file_name):
        self.file_name = file_name
        self.file_path = file_path

    def read_source_data(self):
        file_loc = self.file_path+'\\'+self.file_name
        with open(file_loc,"r") as source_file:
            for line in source_file:
                parts = line.replace("\n","").split(" ")
                print parts

print('Hello World !')
file_path = 'C:\Users\dpatel\Documents\MachineLearningAlgorithms\GenerativeModels\data'
file_name = '2gaussian.txt'
em = EM(file_path,file_name)
em.read_source_data()