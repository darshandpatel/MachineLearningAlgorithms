package ml;

import java.util.ArrayList;
import java.util.Arrays;

public class Constant {
	
	public static ArrayList<String> features = new ArrayList<String>(
			Arrays.asList("CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"));
	
	public static String TRAIN = "TRAIN";
	public static String TEST = "TEST";
	public static String CATEGORICAL = "CATEGORICAL";
	public static String NUMERIC = "NUMERIC";
	public static String BINARY_NUM = "BINARY-NUM";
	public static String BINARY_CAT = "BINARY-CAT";
	
	public static String FILE_PATH = 
			"C:\\Users\\dpatel\\Documents\\MachineLearningAlgorithms\\DecisionTree\\src\\data\\";
	public static String TRAINDATA_FILE = "housing_train.txt";
	public static String TESTDATA_FILE = "housing_test.txt";
	
	public static ArrayList<String> featureTypes = new ArrayList<String>(
			Arrays.asList(CATEGORICAL,NUMERIC,BINARY_NUM,BINARY_CAT));
}
