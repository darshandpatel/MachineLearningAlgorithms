package ml;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.Queue;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import Jama.Matrix;

public class BasicRegressionTree {
	
	Queue<Node> nodeQueue;
	Node rootNode;
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
		features = fetchFeaturePossThreshold(dataMatrix);
		
		// Create the root node for Regression Tree and add into Queue
		Node rootNode = new Node();
		rootNode.setDataPoints(dataMatrix.getRowDimension());
		rootNode.setVariance(calculateVariance(rootNode.getDataPoints()));
		nodeQueue.add(rootNode);
		this.rootNode = rootNode;
		
	}
	
	
	public void formRegressionTree(){
		
		Integer exploredNodeCount = 0;
		Integer exploredNodeLimit = (1 + (int)Math.pow(2, depthLimit));
		
		// Form the tree until Node queue is not empty and the # of explored node 
		// is less the # of explored node limit
		while( (nodeQueue.size() > 0) && (exploredNodeCount < exploredNodeLimit)){
			
			Node currentNode = nodeQueue.poll();
			
			splitNodeByFeaturethreshold(currentNode);
			currentNode.getDataPoints().clear();
			
			if(currentNode.getLeftChildNode() != null && currentNode.getRightChildNode() != null){
				nodeQueue.add(currentNode.getLeftChildNode());
				nodeQueue.add(currentNode.getRightChildNode());
			}
			
		}
	}
	
	
	/**
	 * 
	 * @param dataPoints : The Matrix which contains the training DataSet 
	 * @return ArrayList of Feature, Basically this method parse the dataPoints 
	 * matrix and find the possible threshold value for all the features.
	 */
	public ArrayList<Feature> fetchFeaturePossThreshold(Matrix dataPoints){
		
		int noOfColumns = dataPoints.getColumnDimension();
		int noOfRows = dataPoints.getRowDimension();
		ArrayList<Feature> features = new ArrayList<Feature>();
		ArrayList<ArrayList<Double>> featurePosCriValues = 
				new ArrayList<ArrayList<Double>>(Constant.features.size());
			
		for(int i=0; i < (noOfColumns -1);i++){
			
			for(int j=0; j< noOfRows; j++){
				
				//System.out.println("value of i is : " + i+ " : "+parts[i]);
				//String featureName = Constant.features.get(i);
				if( (i < featurePosCriValues.size()) && (featurePosCriValues.get(i) != null)){
					Double value = dataPoints.get(j,i);
					ArrayList<Double> values = featurePosCriValues.get(i);
					if(!values.contains(value))
						values.add(value);
					
				}else{
					
					ArrayList<Double> values = new ArrayList<Double>();
					values.add(dataPoints.get(j,i));
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
				calculatedFeaturePosCriValues = filterFeaturePosThreshold
						(calculatedFeaturePosCriValues);
			}
			
			Feature feature = new Feature(featureName,featureCtg,calculatedFeaturePosCriValues,i);
			features.add(feature);
		}
		
		return features;
	}
	

	/**
	 * This method modifies the given unique filter threshold values(ArrayList).
	 * Basically it takes the average of two existing threshold value and makes that average value 
	 * a new threshold values
	 * @param calculatedFeaturePosCriValues : unique filter threshold values(ArrayList)
	 * @return the ArrayList of new threshold value for the given threshold value ArrayList
	 */
	public ArrayList<Double> filterFeaturePosThreshold 
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
	
	
	
	public void splitNodeByFeaturethreshold(Node node){
		
		Double parentNodeVariance = node.getVariance();
		
		ArrayList<Double> infoGainPerFeatureBestThreshold = new ArrayList<Double>();
		ArrayList<Integer> bestThresholdIndexPerFeature = new ArrayList<Integer>();
		
		ArrayList<ArrayList<Integer>> leftSideDPForFeaturesBestThreshold = 
				new ArrayList<ArrayList<Integer>>();
		ArrayList<ArrayList<Integer>> rightSideDPForFeatureBestThreshold = 
				new ArrayList<ArrayList<Integer>>();
		
		ArrayList<Double> leftSideVarianceForFeaturesBestThreshold = 
				new ArrayList<Double>();
		ArrayList<Double> rightSideVarianceForFeatureBestThreshold = 
				new ArrayList<Double>();
		
		
		for(Feature feature : features){
			
			ArrayList<ArrayList<Integer>> leftSideDPPerThreshold = new ArrayList<ArrayList<Integer>>();
			ArrayList<ArrayList<Integer>> rightSideDPPerThreshold = new ArrayList<ArrayList<Integer>>();
			
			Integer NoOfDataPoints = dataMatrix.getRowDimension();

			for(int row = 0 ; row < NoOfDataPoints ; row++){
				
				ArrayList<Double> values = feature.getValues();
				int noOfThresholdValues = values.size();
				
				Double trainFeatureValue = dataMatrix.get(row,feature.getIndex());
				
				if(feature.getType().equals(Constant.NUMERIC)){
					
					for(int i= 0 ; i < noOfThresholdValues;i++){
						
						if(trainFeatureValue < values.get(i)){
							if(i < leftSideDPPerThreshold.size()){
								//System.out.println("Index : "+i+" , size is : "+leftSideDPPerThreshold.get(i).size());
								leftSideDPPerThreshold.get(i).add(row);
							}else{
								ArrayList<Integer> dataPoints = new ArrayList<Integer>();
								dataPoints.add(row);
								leftSideDPPerThreshold.add(dataPoints);
							}
								
						}else{
							
							if(i < rightSideDPPerThreshold.size()){
								//System.out.println("Index : "+i+" , size is : "+rightSideDPPerThreshold.get(i).size());
								rightSideDPPerThreshold.get(i).add(row);
							}else{
								ArrayList<Integer> dataPoints = new ArrayList<Integer>();
								dataPoints.add(row);
								rightSideDPPerThreshold.add(dataPoints);
							}
						}
						
					}
				}else if(feature.getType().equals(Constant.BINARY_NUM)){
					
					//System.out.println("In side binary");
					//System.out.println("# of different value "+ noOfThresholdValues);
					for(int i= 0 ; i< noOfThresholdValues;i++){
						
						if(trainFeatureValue == values.get(i)){
							if(i < leftSideDPPerThreshold.size()){
								leftSideDPPerThreshold.get(i).add(row);
							}else{
								ArrayList<Integer> dataPoints = new ArrayList<Integer>();
								dataPoints.add(row);
								leftSideDPPerThreshold.add(dataPoints);
							}
								
						}else{
							
							if(i < rightSideDPPerThreshold.size()){
								rightSideDPPerThreshold.get(i).add(row);
							}else{
								ArrayList<Integer> dataPoints = new ArrayList<Integer>();
								dataPoints.add(row);
								rightSideDPPerThreshold.add(dataPoints);
							}
						}
					}
					
				}
			}
			
			
			System.out.println("Total right data points "+ rightSideDPPerThreshold.size());
			System.out.println("Total left data points "+ leftSideDPPerThreshold.size());
			
			ArrayList<Double> leftSideLabelVariances = calculateLabelValueVariance(feature, 
					leftSideDPPerThreshold);
			ArrayList<Double> rightSideLabelVariances = calculateLabelValueVariance(feature, 
					rightSideDPPerThreshold);
			
			ArrayList<Double> totalLabelVariance = new ArrayList<Double>();
			int length = leftSideLabelVariances.size();
			
			for(int i = 0 ; i< length ; i++){
				totalLabelVariance.add(leftSideLabelVariances.get(i) + rightSideLabelVariances.get(i));
			}
			
			Double lowestLabelVariance = Collections.min(totalLabelVariance);
			Integer bestThresholdIndex = totalLabelVariance.indexOf(lowestLabelVariance);
			
			
			bestThresholdIndexPerFeature.add(bestThresholdIndex);
			Double infoGain = parentNodeVariance - lowestLabelVariance;
			infoGainPerFeatureBestThreshold.add(infoGain);
			
			System.out.println("Information Gain		: " +infoGain);
			System.out.println("Feature index 			: " +feature.getIndex());
			System.out.println("Feature type 			: " +feature.getType());
			System.out.println("Threshold value index 	: " +bestThresholdIndex);
			
			leftSideDPForFeaturesBestThreshold.add(leftSideDPPerThreshold.get(bestThresholdIndex));
			rightSideDPForFeatureBestThreshold.add(leftSideDPPerThreshold.get(bestThresholdIndex));
			
			leftSideVarianceForFeaturesBestThreshold.add(leftSideLabelVariances.get(bestThresholdIndex));
			rightSideVarianceForFeatureBestThreshold.add(rightSideLabelVariances.get(bestThresholdIndex));
		}
		
		Double higherstInfoGain = Collections.max(infoGainPerFeatureBestThreshold);
		Integer bestFeatureIndex = infoGainPerFeatureBestThreshold.indexOf(higherstInfoGain);
		
		if(higherstInfoGain > Constant.INFO_GAIN_THRESHOLD){
			
			Node leftNode = new Node();
			leftNode.setDataPoints(leftSideDPForFeaturesBestThreshold.get(bestFeatureIndex));
			leftNode.setParentNode(node);
			leftNode.setVariance(leftSideVarianceForFeaturesBestThreshold.get(bestFeatureIndex));
			
			Node rightNode = new Node();
			rightNode.setDataPoints(rightSideDPForFeatureBestThreshold.get(bestFeatureIndex));
			rightNode.setParentNode(node);
			leftNode.setVariance(rightSideVarianceForFeatureBestThreshold.get(bestFeatureIndex));
			
			node.setLeftChildNode(leftNode);
			node.setRightChildNode(rightNode);
			
		}
		
		
	}
	
	
	public ArrayList<Double> calculateLabelValueVariance(Feature feature, 
			ArrayList<ArrayList<Integer>> dataPoints){
			
		ArrayList<Double> thresholdVariance = new ArrayList<Double>();
		
		//System.out.println(((ArrayList)feature.getValues()).size());
		int noOfValues = feature.getValues().size() ;
		for(int i =0 ; i < noOfValues; i++){
			
			// Calculate the variance of label values
			Double labelVariance = 0d;
			if(i < dataPoints.size()){
				labelVariance = calculateVariance(dataPoints.get(i));
			}
			
			thresholdVariance.add(labelVariance);
			
		}
		
		// Return the index of element from thresholdVariance which has lowest variance.
		return thresholdVariance;
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
	
	public void printRegressionTree(){
		
		System.out.println("Root node :");
		printRegressionTreeNode(this.rootNode);
		
	}
	
	public void printRegressionTreeNode(Node node){
		
		System.out.println("Feature name: 				" + node.getFeature().getName());
		System.out.println("Feature Threshold Value: 	" + node.getThresholdValue());
		
		if(node.getLeftChildNode() != null){
			System.out.println("Left child node :");
			printRegressionTreeNode(node.getLeftChildNode());
			System.out.println("Right child node :");
			printRegressionTreeNode(node.getRightChildNode());
		}
	}
	
}
