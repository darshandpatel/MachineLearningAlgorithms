package ml;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Map.Entry;
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
		rootNode.setVariance(calculateVariance(dataMatrix));
		System.out.println("Root node variance is : "+ rootNode.getVariance());
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
			
			if(currentNode.getLeftChildNode() != null){
				nodeQueue.add(currentNode.getLeftChildNode());
			}
			if(currentNode.getRightChildNode() != null){
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
		HashMap<Integer,ArrayList<Double>> featurePosCriValues = 
				new HashMap<Integer,ArrayList<Double>>(Constant.features.size());
			
		for(int i=0; i < (noOfColumns -1);i++){
			
			for(int j=0; j< noOfRows; j++){
				
				//System.out.println("value of i is : " + i+ " : "+parts[i]);
				//String featureName = Constant.features.get(i);
				if(featurePosCriValues.containsKey(i)){
					Double value = dataPoints.get(j,i);
					ArrayList<Double> values = featurePosCriValues.get(i);
					if(!values.contains(value))
						values.add(value);
					
				}else{
					
					ArrayList<Double> values = new ArrayList<Double>();
					values.add(dataPoints.get(j,i));
					featurePosCriValues.put(i,values);
				}
			}
		}
		
		for(int i = 0; i < Constant.NO_OF_FEATURES;i++){
			
			String featureName = Constant.features.get(i);
			String featureCtg = null;
			
			ArrayList<Double> calculatedFeaturePosCriValues = featurePosCriValues.get(i);
			Collections.sort(calculatedFeaturePosCriValues);
			
			if(featureName.equals("CHAS")){
				featureCtg = Constant.BINARY_NUM;
			}else{
				featureCtg = Constant.NUMERIC;
				//calculatedFeaturePosCriValues = filterFeaturePosThreshold
				//		(calculatedFeaturePosCriValues);
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
				(ArrayList<Double> calculatedFeaturePosThresholdValues){
		
		ArrayList<Double> filterdPosThresholdValue= new ArrayList<Double>();
			
		int length = calculatedFeaturePosThresholdValues.size();
		
		for(int i=0 ; (i+1) < length; i++){
			Double filteredValue = ((calculatedFeaturePosThresholdValues.get(i) + 
					calculatedFeaturePosThresholdValues.get(i+1))/2);
			filterdPosThresholdValue.add(filteredValue);
		}
		calculatedFeaturePosThresholdValues.clear();
		return filterdPosThresholdValue;
	}
	
	
	
	public void splitNodeByFeaturethreshold(Node node){
		
		Double parentNodeVariance = node.getVariance();
		
		HashMap<Integer,Double> infoGainPerFeatureBestThreshold = new HashMap<Integer,Double>();
		HashMap<Integer,Integer> bestThresholdIndexPerFeature = new HashMap<Integer,Integer>();
		
		HashMap<Integer,ArrayList<Integer>> leftSideDPForFeaturesBestThreshold = 
				new HashMap<Integer,ArrayList<Integer>>();
		HashMap<Integer,ArrayList<Integer>> rightSideDPForFeatureBestThreshold = 
				new HashMap<Integer,ArrayList<Integer>>();
		
		HashMap<Integer,Double> leftSideVarianceForFeaturesBestThreshold = 
				new HashMap<Integer,Double>();
		HashMap<Integer,Double> rightSideVarianceForFeatureBestThreshold = 
				new HashMap<Integer,Double>();
		
		
		for(Feature feature : features){
			
			System.out.println("*********************FEATURE STARTS********************");
			
			HashMap<Integer,ArrayList<Integer>> leftSideDPPerThreshold = 
					new HashMap<Integer,ArrayList<Integer>>();
			HashMap<Integer,ArrayList<Integer>> rightSideDPPerThreshold = 
					new HashMap<Integer,ArrayList<Integer>>();
			
			Integer NoOfDataPoints = dataMatrix.getRowDimension();
			int featureIndex = feature.getIndex();

			ArrayList<Double> values = feature.getValues();
			int noOfThresholdValues = values.size();
			
			for(int row = 0 ; row < NoOfDataPoints ; row++){
				
				Double trainFeatureValue = dataMatrix.get(row,feature.getIndex());
				
				if(feature.getType().equals(Constant.NUMERIC)){
					
					for(int i= 0 ; i < noOfThresholdValues;i++){
						
						if(trainFeatureValue < values.get(i)){
							
							if(leftSideDPPerThreshold.containsKey(i)){
								//System.out.println("Index : "+i+" , size is : "+leftSideDPPerThreshold.get(i).size());
								leftSideDPPerThreshold.get(i).add(row);
							}else{
								ArrayList<Integer> dataPoints = new ArrayList<Integer>();
								dataPoints.add(row);
								leftSideDPPerThreshold.put(i,dataPoints);
							}
								
						}else{
							
							if(rightSideDPPerThreshold.containsKey(i)){
								//System.out.println("Index : "+i+" , size is : "+rightSideDPPerThreshold.get(i).size());
								rightSideDPPerThreshold.get(i).add(row);
							}else{
								ArrayList<Integer> dataPoints = new ArrayList<Integer>();
								dataPoints.add(row);
								rightSideDPPerThreshold.put(i,dataPoints);
							}
						}
						
					}
				}else if(feature.getType().equals(Constant.BINARY_NUM)){
					
					//System.out.println("In side binary");
					//System.out.println("# of different value "+ noOfThresholdValues);
					for(int i= 0 ; i< noOfThresholdValues;i++){
						
						if(trainFeatureValue == values.get(i)){
							if(leftSideDPPerThreshold.containsKey(i)){
								leftSideDPPerThreshold.get(i).add(row);
							}else{
								ArrayList<Integer> dataPoints = new ArrayList<Integer>();
								dataPoints.add(row);
								leftSideDPPerThreshold.put(i,dataPoints);
							}
								
						}else{
							
							if(rightSideDPPerThreshold.containsKey(i)){
								rightSideDPPerThreshold.get(i).add(row);
							}else{
								ArrayList<Integer> dataPoints = new ArrayList<Integer>();
								dataPoints.add(row);
								rightSideDPPerThreshold.put(i,dataPoints);
							}
						}
					}
					
				}
			}
			
			
			HashMap<Integer,Double> leftSideLabelVariances = calculateLabelValueVariance(feature, 
					leftSideDPPerThreshold);
			HashMap<Integer,Double> rightSideLabelVariances = calculateLabelValueVariance(feature, 
					rightSideDPPerThreshold);
			
			ArrayList<Double> totalLabelVariance = new ArrayList<Double>();
			
			for(int i = 0 ; i< noOfThresholdValues ; i++){
				
				double leftSideLabelVariance = 0d;
				if(leftSideLabelVariances.containsKey(i)){
					leftSideLabelVariance = leftSideLabelVariances.get(i);
				}
				double rightSideLabelVariance = 0d;
				if(rightSideLabelVariances.containsKey(i)){
					rightSideLabelVariance = rightSideLabelVariances.get(i);
				}
				totalLabelVariance.add(leftSideLabelVariance + rightSideLabelVariance);
			}
			
			
			Double lowestLabelVariance = Collections.min(totalLabelVariance);
			
			Integer bestThresholdIndex = totalLabelVariance.indexOf(lowestLabelVariance);
			System.out.println("Lowest label variance     "+lowestLabelVariance);
			
			bestThresholdIndexPerFeature.put(featureIndex,bestThresholdIndex);
			
			Double infoGain = parentNodeVariance - lowestLabelVariance;
			infoGainPerFeatureBestThreshold.put(featureIndex,infoGain);
			
			System.out.println("# of threshold values 	: " + values.size());
			System.out.println("Information Gain		: " + infoGain);
			System.out.println("Feature index 			: " + featureIndex);
			System.out.println("Feature type 			: " + feature.getType());
			System.out.println("Threshold value index 	: " + bestThresholdIndex);
			System.out.println("Threshold value 		: " + values.get(bestThresholdIndex));
			
			if(leftSideDPPerThreshold.get(bestThresholdIndex) != null){
				System.out.println("# of datapoint on left side: "+leftSideDPPerThreshold.get(bestThresholdIndex).size());
				leftSideDPForFeaturesBestThreshold.put(featureIndex,leftSideDPPerThreshold.get(bestThresholdIndex));
			}
			if(rightSideDPPerThreshold.get(bestThresholdIndex) != null){
				System.out.println("# of datapoint on right side: "+rightSideDPPerThreshold.get(bestThresholdIndex).size());
				rightSideDPForFeatureBestThreshold.put(featureIndex,rightSideDPPerThreshold.get(bestThresholdIndex));
			}
			
			leftSideVarianceForFeaturesBestThreshold.put(featureIndex,leftSideLabelVariances.get(bestThresholdIndex));
			System.out.println("Left side variance				: "+leftSideLabelVariances.get(bestThresholdIndex));
			rightSideVarianceForFeatureBestThreshold.put(featureIndex,rightSideLabelVariances.get(bestThresholdIndex));
			System.out.println("Right side variance				: "+rightSideLabelVariances.get(bestThresholdIndex));
			
			System.out.println("Total variance is				: "+totalLabelVariance.get(bestThresholdIndex));
		}
		
		System.out.println("*********************ALL FEATURES SCANNED************");
		
		Double higherstInfoGain = Collections.max(infoGainPerFeatureBestThreshold);
		Integer bestFeatureIndex = infoGainPerFeatureBestThreshold.indexOf(higherstInfoGain);
		
		System.out.println("Best Feature index is	: "+ bestFeatureIndex);
		Integer bestThresholdValueIndex = bestThresholdIndexPerFeature.get(bestFeatureIndex);
		System.out.println("Left : " + leftSideDPForFeaturesBestThreshold.get(bestFeatureIndex).size());
		System.out.println("Right : "+rightSideDPForFeatureBestThreshold.get(bestFeatureIndex).size());
		node.setThresholdValue(features.get(bestFeatureIndex).getValues().get(bestThresholdValueIndex));
		
		if(higherstInfoGain > Constant.INFO_GAIN_THRESHOLD){
			
			System.out.println("More than threshold information gain");
			
			Feature bestFeature = features.get(bestFeatureIndex);
			
			System.out.println("Best Threshold value	:" + bestFeature.getValues().get(bestThresholdIndexPerFeature.get(bestFeatureIndex)));
			
			if(Constant.VARIANCE_THRESHOLD < leftSideVarianceForFeaturesBestThreshold.get(bestFeatureIndex)){
				Node leftNode = new Node();
				System.out.println("Left child # of data points "
								+leftSideDPForFeaturesBestThreshold.get(bestFeatureIndex).size());
				leftNode.setDataPoints(leftSideDPForFeaturesBestThreshold.get(bestFeatureIndex));
				leftNode.setParentNode(node);
				leftNode.setVariance(leftSideVarianceForFeaturesBestThreshold.get(bestFeatureIndex));
				System.out.println("Left child variance	 		"
						+leftSideVarianceForFeaturesBestThreshold.get(bestFeatureIndex));
				node.setLeftChildNode(leftNode);
			}
			
			
			if(Constant.VARIANCE_THRESHOLD < rightSideVarianceForFeatureBestThreshold.get(bestFeatureIndex)){
				
				Node rightNode = new Node();
				System.out.println("Right child # of data points "
								+rightSideDPForFeatureBestThreshold.get(bestFeatureIndex).size());
				rightNode.setDataPoints(rightSideDPForFeatureBestThreshold.get(bestFeatureIndex));
				rightNode.setParentNode(node);
				rightNode.setVariance(rightSideVarianceForFeatureBestThreshold.get(bestFeatureIndex));
				System.out.println("Right child variance	 		"
						+rightSideVarianceForFeatureBestThreshold.get(bestFeatureIndex));
				
				node.setRightChildNode(rightNode);
			}
			
		}
		
	}
	
	
	public HashMap<Integer,Double> calculateLabelValueVariance(Feature feature, 
			HashMap<Integer,ArrayList<Integer>> dataPoints){
			
		HashMap<Integer,Double> variancePerThreshold = new HashMap<Integer,Double>();
		
		//System.out.println(((ArrayList)feature.getValues()).size());
		Iterator<Entry<Integer, ArrayList<Integer>>> dataPointIterator = dataPoints.entrySet().iterator();
		Double labelVariance = 0d;
		
		while(dataPointIterator.hasNext()){
			
			Entry<Integer, ArrayList<Integer>> entry = dataPointIterator.next();
			labelVariance = calculateVariance(entry.getValue());
			variancePerThreshold.put(entry.getKey(), labelVariance);
			
		}

		return variancePerThreshold;
	}
	
	/**
	 * This method calculate the variance of the target value for the given 
	 * dataPoints.
	 * @param values
	 * @return
	 */
	public Double calculateVariance(ArrayList<Integer> dataPoints){
		
		if(dataPoints.size() > 0){
			DescriptiveStatistics descriptiveStatistics = new DescriptiveStatistics();
			for(Integer row : dataPoints){
				descriptiveStatistics.addValue(dataMatrix.get(row, Constant.TARGET_VALUE_INDEX));
			}
			return descriptiveStatistics.getVariance();
		}else{
			return 0d;
		}
		
	}
	
	/**
	 * This method calculate the variance of the target value for the given 
	 * dataPoints.
	 * @param values
	 * @return
	 */
	public Double calculateVariance(Matrix dataPoints){
		
		DescriptiveStatistics descriptiveStatistics = new DescriptiveStatistics();
		for(int i=0;i<dataPoints.getRowDimension();i++){
			descriptiveStatistics.addValue(dataMatrix.get(i,Constant.TARGET_VALUE_INDEX));
		}
		return descriptiveStatistics.getVariance();
	}
	
	public void printRegressionTree(){
		
		System.out.println("Root node :");
		printRegressionTreeNode(this.rootNode);
		
	}
	
	public void printRegressionTreeNode(Node node){
		
		//System.out.println("Feature name: 				" + node.getFeature);
		System.out.println("Feature Threshold Value: 	" + node.getThresholdValue());
		
		if(node.getLeftChildNode() != null){
			System.out.println("Left child node :");
			printRegressionTreeNode(node.getLeftChildNode());
			System.out.println("Right child node :");
			printRegressionTreeNode(node.getRightChildNode());
		}
	}
	
}
