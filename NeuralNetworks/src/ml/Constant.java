package ml;

import java.util.ArrayList;
import java.util.Arrays;

public class Constant {
		
	public static String TRAIN = "TRAIN";
	public static String TARGET = "TARGET";
	public static String ATTRIBUTES = "ATTRIBUTES";
	public static String TEST = "TEST";
	
	public static String CATEGORICAL = "CATEGORICAL";
	public static String NUMERIC = "NUMERIC";
	public static String BINARY_NUM = "BINARY-NUM";
	public static String BINARY_CAT = "BINARY-CAT";
	public static String STRING_REGEX = "\\s+";
	public static String COMMA_REGEX = ",";
	
	public static ArrayList<String> FEATURE_TYPES = new ArrayList<String>(
			Arrays.asList(CATEGORICAL,NUMERIC,BINARY_NUM,BINARY_CAT));
	
	public static Integer PERCEPTRON_NO_OF_FEATURES = 4;
	
	public static String PERCEPTRON_DATA_FILE_PATH = 
			"/Users/Darshan/Documents/MachineLearningAlgorithms/DecisionTree/src/data";
	public static String PERCEPTRON_DATA_FILE_NAME = "peceptronData.txt";
	
	public static Integer PERCEPTRON_DATA_NUM_OF_DP = 1000;
	public static Integer PERCEPTRON_DATA_TARGET_VALUE_INDEX = 4;
	
}
