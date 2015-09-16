package ml;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.Queue;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import Jama.Matrix;

public class BasicRegressionTree {
	
	Queue<Node> nodeQueue;
	
	ArrayList<Feature> features;
	LinkedList<Feature> selectedFeature = new LinkedList<Feature>();
	FileOperations fileOperations;
	Matrix dataMatrix;
	Integer depthLimit = 15;
	
	public BasicRegressionTree(){
		
		// Create node queue
		nodeQueue = new LinkedList<Node>();
		
		// Create the mapping of DataPoint and its start byte code from the file 
		fileOperations =  new FileOperations();
		dataMatrix =  fileOperations.fetchDataPoints();
		features = fetchFeaturePossCriValues(dataMatrix);
		
		// Create the root node for Regression Tree and add into Queue
		Node rootNode = new Node();
		rootNode.setDataPoints(dataMatrix.getRowDimension());
		rootNode.setVariance(calculateVariance(rootNode.getDataPoints()));
		nodeQueue.add(rootNode);
				
		
	}
	
	
	public void formRegressionTree(){
		
		
		Integer exploredNodeCount = 0;
		Integer exploredNodeLimit = (1 + (int)Math.pow(2, depthLimit));
		
		// Form the tree until Node queue is not empty and the # of explored node 
		// is less the # of explored node limit
		while( (nodeQueue.size() > 0) && (exploredNodeCount < exploredNodeLimit)){
			
			Node currentNode = nodeQueue.poll();
			
			
			
			
			
		}
		
	}
	
	
	/**
	 * 
	 * @param dataPoints : The Matrix which contains the training DataSet 
	 * @return ArrayList of Feature, Basically this method parse the dataPoints 
	 * matrix and find the possible criteria value for all the features.
	 */
	public ArrayList<Feature> fetchFeaturePossCriValues(Matrix dataPoints){
		
		int noOfColumns = dataPoints.getColumnDimension();
		int noOfRows = dataPoints.getRowDimension();
		ArrayList<Feature> features = new ArrayList<Feature>();
		ArrayList<ArrayList<Double>> featurePosCriValues = 
				new ArrayList<ArrayList<Double>>(Constant.features.size());
			
		for(int i=0;i<noOfRows;i++){
			
			for(int j=0; j< (noOfColumns - 1) ; j++){
				
				//System.out.println("value of i is : " + i+ " : "+parts[i]);
				//String featureName = Constant.features.get(i);
				if( (i < featurePosCriValues.size()) && (featurePosCriValues.get(i) != null)){
					Double value = dataPoints.get(i, j);
					ArrayList<Double> values = featurePosCriValues.get(i);
					if(!values.contains(value))
						values.add(value);
					
				}else{
					
					ArrayList<Double> values = new ArrayList<Double>();
					values.add(dataPoints.get(i, j));
					featurePosCriValues.add(values);
				}
			}
			
		}
		
		for(int i = 0; i < Constant.features.size();i++){
			
			String featureName = Constant.features.get(i);
			String featureCtg = null;
			
			ArrayList<Double> calculatedFeaturePosCriValues = featurePosCriValues.get(i);
			Collections.sort(calculatedFeaturePosCriValues);
			
			if(featureName.equals("CHAS")){
				featureCtg = Constant.BINARY_NUM;
			}else{
				featureCtg = Constant.NUMERIC;
				calculatedFeaturePosCriValues = filterFeaturePosCriValues
						(calculatedFeaturePosCriValues);
			}
			
			Feature feature = new Feature(featureName,featureCtg,calculatedFeaturePosCriValues,i);
			features.add(feature);
		}
		
		return features;
	}
	

	/**
	 * This method modifies the given unique filter criteria values(ArrayList).
	 * Basically it takes the average of two existing criteria value and makes that average value 
	 * a new criteria values
	 * @param calculatedFeaturePosCriValues : unique filter criteria values(ArrayList)
	 * @return the ArrayList of new criteria value for the given criteria value ArrayList
	 */
	public ArrayList<Double> filterFeaturePosCriValues 
				(ArrayList<Double> calculatedFeaturePosCriValues){
		
		ArrayList<Double> filterdPosCriValue= new ArrayList<Double>();
			
		int length = calculatedFeaturePosCriValues.size();
		
		for(int i=0 ; (i+1) < length; i++){
			Double filteredValue = ((calculatedFeaturePosCriValues.get(i) + 
					calculatedFeaturePosCriValues.get(i+1))/2);
			filterdPosCriValue.add(filteredValue);
		}
		calculatedFeaturePosCriValues.clear();
		return filterdPosCriValue;
	}
	
	
	
	public void splitNodeByFeatureCriteria(Node node){
		
		Double previousNodeVariance = node.getVariance();
		
		ArrayList<Double> infoGainPerBestFeature = new ArrayList<Double>();
		ArrayList<Integer> bestCriteriaIndexPerBestFeature = new ArrayList<Integer>();
		ArrayList<ArrayList<Integer>> leftSideDataPointsForBestFeaturesCriteriaValue = 
				new ArrayList<ArrayList<Integer>>();
		ArrayList<ArrayList<Integer>> rightSideDataPointsForBestFeatureCriteriaValue = 
				new ArrayList<ArrayList<Integer>>();
		
		
		for(Feature feature : features){
			
			ArrayList<ArrayList<Integer>> leftSideDataPoints = new ArrayList<ArrayList<Integer>>();
			ArrayList<ArrayList<Integer>> rightSideDataPoints = new ArrayList<ArrayList<Integer>>();
			
			Integer NoOfDataPoints = dataMatrix.getRowDimension();
			Integer featureIndex = feature.getIndex();
			
			for(int row = 0 ; row < NoOfDataPoints ; row++){
				
				if(feature.getType().equals(Constant.NUMERIC)){
					
					ArrayList<Double> values = feature.getValues();
					
					for(int i= 0 ; i < values.size();i++){
						
						Double trainLabelValue = dataMatrix.get(row, Constant.TARGET_VALUE_INDEX);
						
						if(trainLabelValue < values.get(i)){
							if(i < leftSideDataPoints.size()){
								leftSideDataPoints.get(i).add(row);
							}else{
								ArrayList<Integer> dataPoints = new ArrayList<Integer>();
								dataPoints.add(row);
								leftSideDataPoints.add(dataPoints);
							}
								
						}else{
							
							if(i < rightSideDataPoints.size()){
								rightSideDataPoints.get(i).add(row);
							}else{
								ArrayList<Integer> dataPoints = new ArrayList<Integer>();
								dataPoints.add(row);
								rightSideDataPoints.add(dataPoints);
							}
						}
						
					}
				}else if(feature.getType().equals(Constant.BINARY_NUM)){
					
					ArrayList<Double> values = feature.getValues();
					
					for(int i= 0 ; i< values.size();i++){
						
						Double trainLabelValue = dataMatrix.get(row, Constant.TARGET_VALUE_INDEX);
						
						if(trainLabelValue == values.get(i)){
							if(i < leftSideDataPoints.size()){
								leftSideDataPoints.get(i).add(row);
							}else{
								ArrayList<Integer> dataPoints = new ArrayList<Integer>();
								dataPoints.add(row);
								leftSideDataPoints.add(dataPoints);
							}
								
						}else{
							
							if(i < rightSideDataPoints.size()){
								rightSideDataPoints.get(i).add(row);
							}else{
								ArrayList<Integer> dataPoints = new ArrayList<Integer>();
								dataPoints.add(row);
								rightSideDataPoints.add(dataPoints);
							}
						}
					}
					
				}
			}
			
			ArrayList<Double> leftSideLabelVariances = calculateLabelValueVariance(feature, 
					leftSideDataPoints);
			ArrayList<Double> rightSideLabelVariances = calculateLabelValueVariance(feature, 
					rightSideDataPoints);
			ArrayList<Double> totalLabelVariances = 
			
			Double lowestLableVariance = Collections.min(labelVariances);
			Integer criteriaIndex = labelVariances.indexOf(lowestLableVariance);
			bestCriteriaIndexPerBestFeature.add(criteriaIndex);
			Double infoGain = previousNodeVariance - lowestLableVariance;
			infoGainPerBestFeature.add(infoGain);
			leftSideDataPointsForBestFeaturesCriteriaValue.add(leftSideDataPoints.get(criteriaIndex));
			rightSideDataPointsForBestFeatureCriteriaValue.add(rightSideDataPoints.get(criteriaIndex));
		}
		
		Double higherstInfoGain = Collections.max(infoGainPerBestFeature);
		Integer featureIndex = infoGainPerBestFeature.indexOf(higherstInfoGain);
		
		ArrayList<Integer> leftSideDataPointsForBestFeature = 
				leftSideDataPointsForBestFeaturesCriteriaValue.get(featureIndex);
		ArrayList<Integer> rightSideDataPointsForBestFeature = 
				leftSideDataPointsForBestFeaturesCriteriaValue.get(featureIndex);
		
		Node leftNode = new Node();
		leftNode.setDataPoints(leftSideDataPointsForBestFeature);
		leftNode.setParentNode(node);
		
		Node rightNode = new Node();
		rightNode.setDataPoints(rightSideDataPointsForBestFeature);
		rightNode.setParentNode(node);
		
		
	}
	
	
	public ArrayList<Double> calculateLabelValueVariance(Feature feature, 
			ArrayList<ArrayList<Integer>> dataPoints){
			
		ArrayList<Double> criteriaVariance = new ArrayList<Double>();
		
		//System.out.println(((ArrayList)feature.getValues()).size());
		int noOfValues = feature.getValues().size() ;
		for(int i =0 ; i < noOfValues; i++){
			
			// Calculate the variance of label values
			Double labelVariance = 0d;
			if(i < dataPoints.size()){
				labelVariance = calculateVariance(dataPoints.get(i));
			}
			
			criteriaVariance.add(labelVariance);
			
		}
		
		// Return the index of element from criteriaVariance which has lowest variance.
		return criteriaVariance;
	}
	
	/**
	 * This method calculate the variance of the target value for the given 
	 * dataPoints.
	 * @param values
	 * @return
	 */
	public Double calculateVariance(ArrayList<Integer> dataPoints){
		
		DescriptiveStatistics descriptiveStatistics = new DescriptiveStatistics();
		for(Integer row : dataPoints){
			descriptiveStatistics.addValue(dataMatrix.get(row, Constant.TARGET_VALUE_INDEX));
		}
		
		return descriptiveStatistics.getVariance();
		
	}
	
}
