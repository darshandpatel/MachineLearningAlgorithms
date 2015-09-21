package ml;

import java.util.ArrayList;
import java.util.Arrays;

public class Constant {
	
	public static ArrayList<String> features = new ArrayList<String>(
			Arrays.asList("CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"));
	
	//public static ArrayList<String> features = new ArrayList<String>(Arrays.asList("I1","I2"));
	public static Integer NO_OF_FEATURES = features.size();
	
	public static String TRAIN = "TRAIN";
	public static String TEST = "TEST";
	public static String CATEGORICAL = "CATEGORICAL";
	public static String NUMERIC = "NUMERIC";
	public static String BINARY_NUM = "BINARY-NUM";
	public static String BINARY_CAT = "BINARY-CAT";
	
	public static String FILE_PATH = 
			"C:\\Users\\dpatel\\Documents\\MachineLearningAlgorithms\\DecisionTree\\src\\data";
	public static String TRAINDATA_FILE = "housing_train.txt";
	//public static String TRAINDATA_FILE = "simple_data.txt";
	public static String TESTDATA_FILE = "housing_test.txt";
	
	public static Integer NUM_OF_FEATURES = 13;
	//public static Integer NUM_OF_FEATURES = 2;
	public static Integer NUM_OF_TRAINING_DATAPOINTS = 433;
	public static Integer TARGET_VALUE_INDEX = 13;
	//public static Integer TARGET_VALUE_INDEX = 2;
	
	public static ArrayList<String> featureTypes = new ArrayList<String>(
			Arrays.asList(CATEGORICAL,NUMERIC,BINARY_NUM,BINARY_CAT));
	
	public static Double INFO_GAIN_THRESHOLD = 1d;
	public static Double VARIANCE_THRESHOLD = 1d;
	
	public static Integer DEPTH_LIMIT = 15;
	public static Double ERROR_THRESHOLD = 10d;
}
