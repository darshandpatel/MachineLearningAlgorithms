package ml;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.stream.Stream;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import Jama.Matrix;

/**
 * This class contains all the methods related to read/write operation of the DataSet.
 * @author Darshan
 *
 */
public class FileOperations {

	
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
		
		
		Path trainFilepath = Paths.get(Constant.FILE_PATH,Constant.TRAINDATA_FILE);
		
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
	 * 
	 * @return The matrix which rows represent the DataPoints (line in the file)
	 *  and which columns represent the feature values. 
	 */
	public Matrix fetchDataPoints(){
		
		double dataPoints[][] = new double
				[Constant.NUM_OF_TRAINING_DATAPOINTS][Constant.NUM_OF_FEATURES+1];
		try{
			
			Path trainFilepath = Paths.get(Constant.FILE_PATH,Constant.TRAINDATA_FILE);
			Integer lineCounter = 0;
			try(Stream<String> lines = Files.lines(trainFilepath)){
				Iterator<String> lineIterator = lines.iterator();
				while(lineIterator.hasNext()){
					String line = lineIterator.next();
					String parts[] = line.trim().split("\\s+");
					for(int i=0;i<parts.length;i++){
						dataPoints[lineCounter][i] = Double.parseDouble(parts[i]);
					}
					lineCounter++;
				}
			}
			
		}catch(IOException e){
			System.out.println(e.getMessage());
		}
		return new Matrix(dataPoints);
	}
	
}
