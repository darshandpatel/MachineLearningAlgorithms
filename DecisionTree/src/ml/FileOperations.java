package ml;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.stream.Stream;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

/**
 * This class contains all the methods related to read/write operation of the DataSet.
 * @author Darshan
 *
 */
public class FileOperations {
	
	private static String filePath = 
		"C:\\Users\\dpatel\\Documents\\MachineLearningAlgorithms\\DecisionTree\\src\\data\\";
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
	
	/**
	 * This method will find the best fitted criteria value for the given feature.
	 * 
	 * The best fitted criteria will be decided based on the variance in the split data
	 * set.
	 * 
	 * @param feature
	 * @return
	 */
	public Object findBestSplitFeatureCriVal(Feature feature){
		
		
		Path trainFilepath = Paths.get(filePath,trainFile);
		
		ArrayList<ArrayList<Float>> leftSideLabelValPerFeatureCrt = 
				new ArrayList<ArrayList<Float>>(((ArrayList)feature.getValues()).size());
		ArrayList<ArrayList<Float>> rightSideLabelValPerFeatureCrt = 
				new ArrayList<ArrayList<Float>>(((ArrayList)feature.getValues()).size());
		
		ArrayList<ArrayList<Integer>> leftSideDataPoint = new ArrayList<ArrayList<Integer>>();
		ArrayList<ArrayList<Integer>> rightSideDataPoint = new ArrayList<ArrayList<Integer>>();
		
		Integer count = 0;
		try{
			try(Stream<String> lines = Files.lines(trainFilepath)){
			lines.forEach(s -> splitByFeature(s,feature,leftSideLabelValPerFeatureCrt,
					rightSideLabelValPerFeatureCrt,
					count,leftSideDataPoint,rightSideDataPoint));
			}
			
		
			Integer criteriaIndex = findBestFeatureSplit(feature, leftSideLabelValPerFeatureCrt, 
					rightSideLabelValPerFeatureCrt);
			
			System.out.println("Feature value : " + feature.getName());
			System.out.println("Feature Index : " + feature.getIndex());
			System.out.println("Feature criteria index : " + criteriaIndex);
			System.out.println("Feature criteria value : " + ((ArrayList)feature.getValues()).get(criteriaIndex));
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return (float) 0;
		
	}
	
	/**
	 * This method find the feature criteria value, which has the less variance for the 
	 * predicted dataset label values.
	 * @param feature
	 * @param leftSideLabelValPerFeatureCrt
	 * @param rightSideLabelValPerFeatureCrt
	 * @return
	 */
	public Integer findBestFeatureSplit(Feature feature,
			ArrayList<ArrayList<Float>> leftSideLabelValPerFeatureCrt, 
			ArrayList<ArrayList<Float>> rightSideLabelValPerFeatureCrt){
			
		ArrayList<Float> criteriaVariance = new ArrayList<Float>();
		
		
		//System.out.println(((ArrayList)feature.getValues()).size());
		
		for(int i =0 ; i < ((ArrayList)feature.getValues()).size() ; i++){
			
			// Calculate the variance of label values
			Float leftSideLabelVariance = (float) 0.0;
			if(i < leftSideLabelValPerFeatureCrt.size()){
				ArrayList<Float> leftSideLabelValue = leftSideLabelValPerFeatureCrt.get(i);
				leftSideLabelVariance = calculateVariance(leftSideLabelValue);
				leftSideLabelValue.clear();
			}
			
			Float rightSideLabelVariance = (float)0.0;
			if(i < rightSideLabelValPerFeatureCrt.size()){
				ArrayList<Float> rightSideLabelValue = rightSideLabelValPerFeatureCrt.get(i);
				calculateVariance(rightSideLabelValue);
				rightSideLabelValue.clear();
			}
			
			Float totalVariance = leftSideLabelVariance + rightSideLabelVariance;
			criteriaVariance.add(totalVariance);
			
		}
		
		// Return the index of element from criteriaVariance which has lowest variance.
		
		return criteriaVariance.indexOf(Collections.min(criteriaVariance));
	}
	
	/**
	 * This method calculate the variance of the given ArrayList
	 * @param values
	 * @return
	 */
	public Float calculateVariance(ArrayList<Float> values){
		
		DescriptiveStatistics descriptiveStatistics = new DescriptiveStatistics();
		values.forEach(value -> descriptiveStatistics.addValue(value));
		
		return (float)descriptiveStatistics.getVariance();
		
	}
	
	/**
	 * This method parse the datapoint (line) and compare the value of feature
	 * with various feature criteria points. Based upon the comparision it divides 
	 * the dataset into two parts.
	 * @param line
	 * @param feature
	 * @param leftSideLabelValPerFeatureCrt
	 * @param rightSideLabelValPerFeatureCrt
	 * @param count
	 * @param leftSideDataPoint
	 * @param rightSideDataPoint
	 */
	public void splitByFeature(String line,Feature feature,
			ArrayList<ArrayList<Float>> leftSideLabelValPerFeatureCrt,
			ArrayList<ArrayList<Float>> rightSideLabelValPerFeatureCrt,Integer count,
			ArrayList<ArrayList<Integer>> leftSideDataPoint,
			ArrayList<ArrayList<Integer>> rightSideDataPoint){
		
		String parts[] = line.trim().split("\\s+");
		//System.out.println(parts.length);
		Integer featureIndex = feature.getIndex();
		
		if(feature.getType().equals(Constant.NUMERIC)){
			
			ArrayList<Float> values = (ArrayList)feature.getValues();
			
			for(int i= 0 ; i < values.size();i++){
				
				Float trainDataValue = Float.parseFloat(parts[featureIndex]);
				if(trainDataValue < values.get(i)){
					if(i < leftSideLabelValPerFeatureCrt.size()){
						leftSideLabelValPerFeatureCrt.get(i).add(trainDataValue);
						leftSideDataPoint.get(i).add(1);
					}else{
						ArrayList<Float> trainDataValues = new ArrayList<Float>();
						trainDataValues.add(trainDataValue);
						leftSideLabelValPerFeatureCrt.add(trainDataValues);
						ArrayList<Integer> dataPoints = new ArrayList<Integer>();
						dataPoints.add(1);
						leftSideDataPoint.add(dataPoints);
					}
						
				}else{
					
					if(i < rightSideLabelValPerFeatureCrt.size()){
						rightSideLabelValPerFeatureCrt.get(i).add(trainDataValue);
						rightSideDataPoint.get(i).add(0);
					}else{
						ArrayList<Float> trainDataValues = new ArrayList<Float>();
						trainDataValues.add(trainDataValue);
						rightSideLabelValPerFeatureCrt.add(trainDataValues);
						ArrayList<Integer> dataPoints = new ArrayList<Integer>();
						dataPoints.add(0);
						rightSideDataPoint.add(dataPoints);
					}
				}
				
			}
		}else if(feature.getType().equals(Constant.BINARY_NUM)){
			
			ArrayList<Float> values = (ArrayList)feature.getValues();
			
			for(int i= 0 ; i< values.size();i++){
				
				Float trainDataValue = Float.parseFloat(parts[featureIndex]);
				if(trainDataValue == values.get(i)){
					if(i < leftSideLabelValPerFeatureCrt.size()){
						leftSideLabelValPerFeatureCrt.get(i).add(trainDataValue);
						leftSideDataPoint.get(i).add(count);
					}else{
						ArrayList<Float> trainDataValues = new ArrayList<Float>();
						trainDataValues.add(trainDataValue);
						leftSideLabelValPerFeatureCrt.add(trainDataValues);
						ArrayList<Integer> dataPoints = new ArrayList<Integer>();
						dataPoints.add(1);
						leftSideDataPoint.add(dataPoints);
					}
						
				}else{
					
					if(i < rightSideLabelValPerFeatureCrt.size()){
						rightSideLabelValPerFeatureCrt.get(i).add(trainDataValue);
						rightSideDataPoint.get(i).add(count);
					}else{
						ArrayList<Float> trainDataValues = new ArrayList<Float>();
						trainDataValues.add(trainDataValue);
						rightSideLabelValPerFeatureCrt.add(trainDataValues);
						ArrayList<Integer> dataPoints = new ArrayList<Integer>();
						dataPoints.add(0);
						rightSideDataPoint.add(dataPoints);
					}
				}
				
				
			}
			
		}
		count++;
	}
	
	/**
	 * This method will return the HashMap with key as line number and
	 * values as byte position of that line in the file.
	 * @return
	 */
	public HashMap<Integer,Double> getBytePosOfLine(String fileCategory){
		
		HashMap<Integer,Double> bytesByLine = new HashMap<Integer,Double>();
		Integer lineCount = 1;
		Double startByteCount = 0d;
		String targetFileName;
		
		if(fileCategory.equals(Constant.TRAIN)){
			targetFileName = trainFile;
		}else{
			targetFileName = testFile;
		}
		
		try{
			
			Path trainFilepath = Paths.get(filePath,targetFileName);
			try(Stream<String> lines = Files.lines(trainFilepath)){
				
				Iterator<String> lineIterator = lines.iterator();
				while(lineIterator.hasNext()){
					
					String line = lineIterator.next();
					bytesByLine.put(lineCount, startByteCount);
					System.out.println("Line number :"+lineCount+" Start byte is : "+startByteCount);
					startByteCount += (line.length()+2);
					lineCount += 1;
					
				}
			}
			
		}catch(IOException ex){
			System.out.println(ex.getMessage());
		}

		return bytesByLine;
		
	}
	
	/**
	 * 
	 * @return RandomAccessFile object of training dataset file.
	 */
	public RandomAccessFile getRandomAccessTrainFile(){
		
		try {
			RandomAccessFile raf = new RandomAccessFile(filePath+trainFile, "r");
			return raf;
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
		
	}
	
	
	public static void main(String args[]){
		
		/*
		FileOperations fileOperations = new FileOperations();
		ArrayList<Feature> features = fileOperations.fetchFeaturePossCriValues(filePath,trainFile);
		fileOperations.printFeatureValues(features);
		
		for(Feature feature : features){
			fileOperations.findBestSplitFeatureCriVal(feature);
		}
		*/
		FileOperations fileOperations = new FileOperations();
		fileOperations.getBytePosOfLine(Constant.TRAIN);
		
	}

}
