package ml;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map.Entry;
import java.util.TreeSet;
import java.util.stream.Stream;

/**
 * This class contains all the methods related to read/write operation of the DataSet.
 * @author Darshan
 *
 */
public class FileOperations {
	
	private static String filePath = 
		"/Users/Darshan/Documents/MachineLearningAlgorithms/DecisionTree/src/data";
	private static String trainFile = "housing_train.txt";
	private static String testFile = "housing_test.txt";
	
	HashMap<String,TreeSet<Float>> featureValues = new HashMap<String,TreeSet<Float>>();
	//ArrayList<Float> labels = new ArrayList<Float>();
	
	/**
	 * 
	 * @param filePath - The folder path in which source file resides.
	 * @param fileName - The file name, which needs to be read.
	 * @return The HashMap which has key as feature and value as ArrayList of all possible value, 
	 * a feature can have from the given DataSet/file.
	 */
	public HashMap<String,TreeSet<Float>> fetchFeaturePossCriValues(String filePath,String fileName){
		
		try {
			
			Path trainFilepath = Paths.get(filePath,fileName);
			try(Stream<String> lines = Files.lines(trainFilepath)){
				lines.forEach(s -> parseFeatureValues(s));
			}
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return null;
	}
	
	/**
	 * This method parse the given line into the feature values and add those into featureValues
	 * HashMap
	 * @param line : The line from source data set
	 */
	public void parseFeatureValues(String line){
		
		//System.out.println(line.trim());
		String parts[] = line.trim().split("\\s+");
		//System.out.println(parts.length);
		for(int i = 0; i < (parts.length - 1); i++){

			//System.out.println("value of i is : " + i+ " : "+parts[i]);
			String featureName = Constant.features.get(i);
			if(featureValues.containsKey(featureName)){
				
				featureValues.get(featureName).add(Float.parseFloat(parts[i]));
				
			}else{
				
				TreeSet<Float> values = new TreeSet<Float>();
				values.add(Float.parseFloat(parts[i]));
				featureValues.put(featureName, values);
				
			}
			
		}
		//labels.add(Float.parseFloat(parts[parts.length-1]));
		
	}
	
	/**
	 * This method print the feature name and its possible values
	 */
	public void printFeatureValues(){
		
		for(Entry<String, TreeSet<Float>> value : featureValues.entrySet()){
			
			System.out.println("Feature name : "+value.getKey());
			System.out.println("Number of different values :" + value.getValue().size());
			//System.out.println("Different feature values : "+value.getValue());
		}
		
		
	}
	
	public static void main(String args[]){
		
		FileOperations fileOperations = new FileOperations();
		fileOperations.fetchFeaturePossCriValues(filePath,trainFile);
		fileOperations.printFeatureValues();
		
	}

}
