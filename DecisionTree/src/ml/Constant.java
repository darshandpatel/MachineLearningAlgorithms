package ml;

import java.util.ArrayList;
import java.util.Arrays;

public class Constant {
	
	
	public static String TRAIN = "TRAIN";
	public static String TEST = "TEST";
	public static String CATEGORICAL = "CATEGORICAL";
	public static String NUMERIC = "NUMERIC";
	public static String BINARY_NUM = "BINARY-NUM";
	public static String BINARY_CAT = "BINARY-CAT";
	public static String STRING_REGEX = "\\s+";
	
	public static ArrayList<String> FEATURE_TYPES = new ArrayList<String>(
			Arrays.asList(CATEGORICAL,NUMERIC,BINARY_NUM,BINARY_CAT));
	
	// HOUSING DATA SET CONSTANT STARTS
	public static ArrayList<String> HOUSING_DATA_FEATURUES = new ArrayList<String>(
			Arrays.asList("CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"));
	
	public static Integer HOUSING_DATA_NO_OF_FEATURES = HOUSING_DATA_FEATURUES.size();
	
	public static String HOUSING_DATA_FILE_PATH = 
			"/Users/Darshan/Documents/MachineLearningAlgorithms/DecisionTree/src/data";
	public static String HOUSING_TRAINDATA_FILE = "housing_train.txt";
	//public static String TRAINDATA_FILE = "simple_data.txt";
	public static String HOUSING_TESTDATA_FILE = "housing_test.txt";
	
	public static Integer HOUSING_DATA_NUM_OF_FEATURES = 13;
	public static Integer HOUSING_DATA_NUM_OF_TRAINING_DP = 433;
	public static Integer HOUSING_DATA_NUM_OF_TESTING_DP = 74;
	public static Integer HOUSING_DATA_TARGET_VALUE_INDEX = 13;
	
	public static Double HOUSING_DATA_INFO_GAIN_THRESHOLD = 2500d;
	
	public static Integer HOUSING_DATA_DEPTH_LIMIT = 10;
	public static Double HOUSING_DATA_ERROR_THRESHOLD = 1000d;
	public static Integer HOUSING_DATA_NUM_OF_CHILD_THRESHOLD = 15;
	
	//HOUSING DATA SET CONSTANT ENDS
	
	// SPAM DATA SET CONSTANT STARTS

	public static Integer SPAMBASE_DATA_NUM_OF_DP = 4601;
	//4601
	public static Integer SPAMBASE_DATA_NUM_OF_FEATURES = 57;
	public static Double SPAMBASE_DATA_INFO_GAIN_THRESHOLD = 0.1d;
	public static String SPAMBASE_DATA_FILE_PATH = 
			"/Users/Darshan/Documents/MachineLearningAlgorithms/DecisionTree/src/data";
	public static String SPAMBASE_DATA_FILE_NAME = "spambase.data";
	
	public static Integer  SPAMBASE_DATA_TARGET_VALUE_INDEX = 57;
	public static Integer SPAMBASE_DATA_DEPTH_LIMIT = 10;
	public static String ENTROPY = "entropy";
	public static String SPAM_COUNT = "spam_count";
	public static String HAM_COUNT = "ham_count";
	// SPAM DATA SET CONSTANT ENDS	
}
