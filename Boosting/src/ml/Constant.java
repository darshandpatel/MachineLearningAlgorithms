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
	public static String HAM_COUNT = "ham_count";
	public static String ERROR_VALUE = "ERROR_VALUE";
	public static String MISCLASSIFIED_DP = "MISCLASSIFIED_DP";
	public static String CORRECTLY_CLASSIFIED_DP = "CORRECTLY_CLASSIFIED_DP";
	public static String ALPHA_VALUE = "ALPHA_VALUE";
	// SPAM DATA SET CONSTANT ENDS	
}
