package ml;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.stream.Stream;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

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
	
	//HashMap<String,ArrayList<Float>> featureValues = new HashMap<String,ArrayList<Float>>();
	//ArrayList<Float> labels = new ArrayList<Float>();
	
	/**
	 * 
	 * @param filePath - The folder path in which source file resides.
	 * @param fileName - The file name, which needs to be read.
	 * @return The HashMap which has key as feature and value as ArrayList of all possible value, 
	 * a feature can have from the given DataSet/file.
	 */
	public ArrayList<Feature> fetchFeaturePossCriValues(String filePath,String fileName){
		
		ArrayList<Feature> features = new ArrayList<Feature>();
		try {
			
			ArrayList<ArrayList<Float>> featurePosCriValues = 
					new ArrayList<ArrayList<Float>>(Constant.features.size());
			
			Path trainFilepath = Paths.get(filePath,fileName);
			
			try(Stream<String> lines = Files.lines(trainFilepath)){
				lines.forEach(s -> parseFeatureValues(s,featurePosCriValues));
			}
			
			for(int i = 0; i < Constant.features.size();i++){
				
				String featureName = Constant.features.get(i);
				String featureCtg = null;
				
				ArrayList<Float> calculatedFeaturePosCriValues = featurePosCriValues.get(i);
				Collections.sort(calculatedFeaturePosCriValues);
				
				if(featureName.equals("CHAS")){
					featureCtg = Constant.BINARY_NUM;
				}else{
					featureCtg = Constant.NUMERIC;
					calculatedFeaturePosCriValues = filterFeaturePosCriValues(calculatedFeaturePosCriValues);
				}
				
				Feature feature = new Feature(featureName,featureCtg,calculatedFeaturePosCriValues,i);
				features.add(feature);
			}
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return features;
	}
	
	/**
	 * This method parse the given line into the feature values and add those into featureValues
	 * HashMap
	 * @param line : The line from source data set
	 */
	public void parseFeatureValues(String line,ArrayList<ArrayList<Float>> featurePosCriValues){
		
		//System.out.println(line.trim());
		String parts[] = line.trim().split("\\s+");
		//System.out.println(parts.length);
		for(int i = 0; i < (parts.length - 1); i++){

			//System.out.println("value of i is : " + i+ " : "+parts[i]);
			//String featureName = Constant.features.get(i);
			if( (i < featurePosCriValues.size()) && (featurePosCriValues.get(i) != null)){
				
				ArrayList<Float> featureValues = featurePosCriValues.get(i);
				Float value = Float.parseFloat(parts[i]);
				if(!featureValues.contains(value))
					featureValues.add(value);
				
			}else{
				ArrayList<Float> values = new ArrayList<Float>();
				values.add(Float.parseFloat(parts[i]));
				featurePosCriValues.add(values);
			}
		}
		//labels.add(Float.parseFloat(parts[parts.length-1]));
		
	}
	
	/**
	 * This method modifies the given unique filter criteria values ArrayList.
	 * Basically it takes the average of two existing criteria value and makes that average value 
	 * a new criteria values
	 * @param calculatedFeaturePosCriValues
	 * @return
	 */
	public ArrayList<Float> filterFeaturePosCriValues 
	(ArrayList<Float> calculatedFeaturePosCriValues){
		
		ArrayList<Float> filterdPosCriValue= new ArrayList<Float>();
			
		int length = calculatedFeaturePosCriValues.size();
		
		for(int i=0 ; (i+1) < length; i++){
			Float filteredValue = ((calculatedFeaturePosCriValues.get(i) + 
					calculatedFeaturePosCriValues.get(i+1))/2);
			filterdPosCriValue.add(filteredValue);
		}
		
		calculatedFeaturePosCriValues.clear();
		return filterdPosCriValue;
	}
	
	/**
	 * This method print the feature name and its possible values
	 */
	public void printFeatureValues(ArrayList<Feature> features){
		
		for(Feature feature : features){
			
			System.out.println("Feature name : "+ feature.getName());
			System.out.println("Feature Type : "+feature.getType());
			System.out.println("Number of different values :" + 
									((ArrayList) feature.getValues()).size());
			System.out.println("Different feature values : "+feature.getValues());
		}
		
		
	}
	
	public Object calculateMSE(Feature feature,Object featureValue){
		
		
		Path trainFilepath = Paths.get(filePath,trainFile);
		ArrayList<ArrayList<Float>> leftSideLabelValPerFeatureCrt = 
				new ArrayList<ArrayList<Float>>(((ArrayList)feature.getValues()).size());
		ArrayList<ArrayList<Float>> rightSideLabelValPerFeatureCrt = 
				new ArrayList<ArrayList<Float>>(((ArrayList)feature.getValues()).size());
		
		ArrayList<ArrayList<Integer>> leftSideDataPoint = new ArrayList<ArrayList<Integer>>();
		ArrayList<ArrayList<Integer>> rightSideDataPoint = new ArrayList<ArrayList<Integer>>();
		int count = 0;
		try{
			try(Stream<String> lines = Files.lines(trainFilepath)){
			lines.forEach(s -> splitByFeature(s,feature,leftSideLabelValPerFeatureCrt,
					rightSideLabelValPerFeatureCrt,
					count,leftSideDataPoint,rightSideDataPoint));
		}
			
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return (float) 0;
		
	}
	
	public Integer findBestFeatureSplit(Feature feature,
			ArrayList<ArrayList<Float>> leftSideLabelValPerFeatureCrt, 
			ArrayList<ArrayList<Float>> rightSideLabelValPerFeatureCrt){
			
		ArrayList<Float> criteriaVariance = new ArrayList<Float>();
		
		for(int i =0 ; i < ((ArrayList)feature.getValues()).size() ; i++){
			
			ArrayList<Float> leftSideLabelValue = leftSideLabelValPerFeatureCrt.get(i);
			ArrayList<Float> rightSideLabelValue = rightSideLabelValPerFeatureCrt.get(i);
			
			// Calculate the variance of label values
			Float totalVariance = calculateVariance(leftSideLabelValue) + 
					calculateVariance(rightSideLabelValue);
			criteriaVariance.add(totalVariance);
			leftSideLabelValue.clear();
			rightSideLabelValue.clear();
		}
		
		return 0;
	}
	
	public Float calculateVariance(ArrayList<Float> values){
		
		DescriptiveStatistics descriptiveStatistics = new DescriptiveStatistics();
		values.forEach(value -> descriptiveStatistics.addValue(value));
		
		return (float)descriptiveStatistics.getVariance();
		
	}
	
	public void splitByFeature(String line,Feature feature,
			ArrayList<ArrayList<Float>> leftSideLabelValPerFeatureCrt,
			ArrayList<ArrayList<Float>> rightSideLabelValPerFeatureCrt,int count,
			ArrayList<ArrayList<Integer>> leftSideDataPoint,
			ArrayList<ArrayList<Integer>> rightSideDataPoint){
		
		String parts[] = line.trim().split("\\s+");
		//System.out.println(parts.length);
		Integer featureIndex = feature.getIndex();
		if(feature.getType().equals(Constant.NUMERIC)){
			
			ArrayList<Float> values = (ArrayList)feature.getValues();
			for(int i= 0 ; i< values.size();i++){
				Float trainDataValue = Float.parseFloat(parts[featureIndex]);
				if(trainDataValue < values.get(featureIndex)){
					if(leftSideLabelValPerFeatureCrt.get(featureIndex) != null){
						leftSideLabelValPerFeatureCrt.get(featureIndex).add(trainDataValue);
						leftSideDataPoint.get(featureIndex).add(count);
					}else{
						ArrayList<Float> trainDataValues = new ArrayList<Float>();
						trainDataValues.add(trainDataValue);
						leftSideLabelValPerFeatureCrt.set(featureIndex,trainDataValues);
						ArrayList<Integer> dataPoints = new ArrayList<Integer>();
						dataPoints.add(count);
						leftSideDataPoint.set(featureIndex, dataPoints);
					}
						
				}else{
					
					if(rightSideLabelValPerFeatureCrt.get(featureIndex) != null){
						rightSideLabelValPerFeatureCrt.get(featureIndex).add(trainDataValue);
						rightSideDataPoint.get(featureIndex).add(count);
					}else{
						ArrayList<Float> trainDataValues = new ArrayList<Float>();
						trainDataValues.add(trainDataValue);
						rightSideLabelValPerFeatureCrt.set(featureIndex,trainDataValues);
						ArrayList<Integer> dataPoints = new ArrayList<Integer>();
						dataPoints.add(count);
						rightSideDataPoint.set(featureIndex, dataPoints);
					}
					
				}
				
			}
		}
		count++;
	}
	
	
	public static void main(String args[]){
		
		FileOperations fileOperations = new FileOperations();
		ArrayList<Feature> features = fileOperations.fetchFeaturePossCriValues(filePath,trainFile);
		fileOperations.printFeatureValues(features);
		
	}

}
