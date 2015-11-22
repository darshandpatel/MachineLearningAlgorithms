package ml;

import java.util.ArrayList;
import java.util.Arrays;

public class Constant {
	
	
	public static String TRAIN = "TRAIN";
	public static String TEST = "TEST";
	public static String TARGET = "TARGET";
	public static String ATTRIBUTES = "ATTRIBUTES";
	public static String TRAIN_DP = "TRAIN_DP";
	public static String TEST_DP = "TEST_DP";
	public static String CATEGORICAL = "CATEGORICAL";
	public static String NUMERIC = "NUMERIC";
	public static String BINARY_NUM = "BINARY-NUM";
	public static String BINARY_CAT = "BINARY-CAT";
	public static String STRING_REGEX = "\\s+";
	public static String COMMA_REGEX = ",";
	
	public static ArrayList<String> FEATURE_TYPES = new ArrayList<String>(
			Arrays.asList(CATEGORICAL,NUMERIC,BINARY_NUM,BINARY_CAT));
	
	// SPAM DATA SET CONSTANT STARTS
	public static Integer NBR_OF_FOLDS = 10;
	public static Integer NBR_OF_DP = 4601;
	//4601
	public static Integer NBR_OF_FEATURES = 57;
	public static Double SPAMBASE_DATA_INFO_GAIN_THRESHOLD = 0.1d;
	public static String SPAMBASE_DATA_FILE_PATH = 
			"/Users/Darshan/Documents/MachineLearningAlgorithms/Boosting/src/data";
	public static String SPAMBASE_DATA_FILE_NAME = "spambase.data";
	
	public static Integer  SPAMBASE_DATA_TARGET_VALUE_INDEX = 57;
	public static Integer SPAMBASE_DATA_DEPTH_LIMIT = 0;
	public static String ENTROPY = "entropy";
	public static String SPAM_COUNT = "spam_count";
	public static String NON_SPAM_COUNT = "ham_count";
	public static String SPAM_SUM_OF_DIST = "SPAM_SUM_OF_DIST";
	public static String NON_SPAM_SUM_OF_DIST = "NON_SPAM_SUM_OF_DIST";
	public static String ERROR_VALUE = "ERROR_VALUE";
	public static String MISCLASSIFIED_DP = "MISCLASSIFIED_DP";
	public static String CORRECTLY_CLASSIFIED_DP = "CORRECTLY_CLASSIFIED_DP";
	public static String ALPHA_VALUE = "ALPHA_VALUE";
	// SPAM DATA SET CONSTANT ENDS	
	
	// HOUSING DATA SET CONSTANT STARTS
	public static ArrayList<String> HOUSING_DATA_FEATURUES = new ArrayList<String>(
			Arrays.asList("CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"));
	
	public static Integer HOUSING_DATA_NO_OF_FEATURES = HOUSING_DATA_FEATURUES.size();
	
	public static String HOUSING_DATA_FILE_PATH = 
			"/Users/Darshan/Documents/MachineLearningAlgorithms/Boosting/src/data";
	public static String BOOSTING_FILE_PATH = 
			"/Users/Darshan/Documents/MachineLearningAlgorithms/Boosting/src/data";
	public static String MULTI_CLASS_FILE_NAME ="bal.data";
	public static String HOUSING_TRAINDATA_FILE = "housing_train.txt";
	//public static String TRAINDATA_FILE = "simple_data.txt";
	public static String HOUSING_TESTDATA_FILE = "housing_test.txt";
	
	public static Integer BAL_DATA_NUM_OF_FEATURES = 4;
	public static Integer BAL_DATA_NUM_OF_DP = 625;
	public static Integer BAL_NBR_OF_CLASS = 3;
	public static Integer BAL_DATA_TARGET_VALUE_INDEX = 4;
	
	public static Integer HOUSING_DATA_NUM_OF_FEATURES = 13;
	public static Integer HOUSING_DATA_NUM_OF_TRAINING_DP = 433;
	public static Integer HOUSING_DATA_NUM_OF_TESTING_DP = 74;
	public static Integer HOUSING_DATA_TARGET_VALUE_INDEX = 13;
	
	public static Double HOUSING_DATA_INFO_GAIN_THRESHOLD = 1d;
	public static Double HOUSING_DATA_VARIANCE_THRESHOLD = 1d;
	
	public static Integer HOUSING_DATA_DEPTH_LIMIT = 2;
	public static Double HOUSING_DATA_ERROR_THRESHOLD = 0.1d;
	
	// SPAM DATA SET CONSTANT STARTS

	public static Integer SPAMBASE_DATA_NUM_OF_DP = 4601;
	//4601
	public static Integer SPAMBASE_DATA_NUM_OF_FEATURES = 57;
	public static String HAM_COUNT = "ham_count";
	public static Integer HOUSING_DATA_NUM_OF_CHILD_THRESHOLD = 15;
	public static Double GRADIENT_BOOSTING_CONVERNGE_THRESHOLD = 10d;
	public static String ROOT_NODE = "ROOT_NODE";
	public static String ROOT_NODES = "ROOT_NODES";
	public static String PREDICTED_TARGET_MATRIX = "PREDICTED_TARGET_MATRIX";
	public static String ALPHAS = "ALPHAS";
	
}
