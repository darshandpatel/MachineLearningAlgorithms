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
	
	/**
	 * 
	 * @return The Root node of Regression Tree
	 */
	Node getRootNode(){
		return rootNode;
	}
	
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
		rootNode.setMSE(calculateVariance(dataMatrix));
		
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
			
			System.out.println("Parent node variance is : "+ currentNode.getMSE());
			
			System.out.println("Parent node Data Points are:"+currentNode.getDataPoints().size());
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
		
		Double parentNodeMSE = node.getMSE();
		
		HashMap<Integer,Double> infoGainPerFeatureBestThreshold = new HashMap<Integer,Double>();
		HashMap<Integer,Integer> bestThresholdIndexPerFeature = new HashMap<Integer,Integer>();
		
		HashMap<Integer,ArrayList<Integer>> leftSideDPForFeaturesBestThreshold = 
				new HashMap<Integer,ArrayList<Integer>>();
		HashMap<Integer,ArrayList<Integer>> rightSideDPForFeatureBestThreshold = 
				new HashMap<Integer,ArrayList<Integer>>();
		
		HashMap<Integer,Double> leftSideVarianceForFeatureBestThreshold = 
				new HashMap<Integer,Double>();
		HashMap<Integer,Double> rightSideVarianceForFeatureBestThreshold = 
				new HashMap<Integer,Double>();
		
		ArrayList<Integer> currentNodeDataPoints = node.getDataPoints();
		
		for(Feature feature : features){
			
			System.out.println("*********************FEATURE STARTS********************");
			
			HashMap<Integer,ArrayList<Integer>> leftSideDPPerThreshold = 
					new HashMap<Integer,ArrayList<Integer>>();
			HashMap<Integer,ArrayList<Integer>> rightSideDPPerThreshold = 
					new HashMap<Integer,ArrayList<Integer>>();
			
			Integer NoOfCurrentNodeDataPoints = currentNodeDataPoints.size();
			int featureIndex = feature.getIndex();

			ArrayList<Double> values = feature.getValues();
			int noOfThresholdValues = values.size();
			
			if(feature.getType().equals(Constant.NUMERIC)){
				
				for(int i= 0 ; i < noOfThresholdValues;i++){
					
					for(int k = 0 ; k < NoOfCurrentNodeDataPoints ; k++){
					
						Double trainFeatureValue = dataMatrix.get(k,featureIndex);
						
						if(trainFeatureValue < values.get(i)){
							
							if(leftSideDPPerThreshold.containsKey(i)){
								//System.out.println("Index : "+i+" , size is : "+leftSideDPPerThreshold.get(i).size());
								leftSideDPPerThreshold.get(i).add(k);
							}else{
								ArrayList<Integer> dataPoints = new ArrayList<Integer>();
								dataPoints.add(k);
								leftSideDPPerThreshold.put(i,dataPoints);
							}
								
						}else{
							
							if(rightSideDPPerThreshold.containsKey(i)){
								//System.out.println("Index : "+i+" , size is : "+rightSideDPPerThreshold.get(i).size());
								rightSideDPPerThreshold.get(i).add(k);
							}else{
								ArrayList<Integer> dataPoints = new ArrayList<Integer>();
								dataPoints.add(k);
								rightSideDPPerThreshold.put(i,dataPoints);
							}
						}
						
					}
				}
			}else if(feature.getType().equals(Constant.BINARY_NUM)){
				
				for(int i= 0 ; i< noOfThresholdValues;i++){
					
					for(int k = 0 ; k < NoOfCurrentNodeDataPoints ; k++){
					
						Double trainFeatureValue = dataMatrix.get(k,featureIndex);
					
						//System.out.println("In side binary");
						//System.out.println("# of different value "+ noOfThresholdValues);
					
						if(trainFeatureValue == 0){
							if(leftSideDPPerThreshold.containsKey(i)){
								leftSideDPPerThreshold.get(i).add(k);
							}else{
								ArrayList<Integer> dataPoints = new ArrayList<Integer>();
								dataPoints.add(k);
								leftSideDPPerThreshold.put(i,dataPoints);
							}
								
						}else{
							
							if(rightSideDPPerThreshold.containsKey(i)){
								rightSideDPPerThreshold.get(i).add(k);
							}else{
								ArrayList<Integer> dataPoints = new ArrayList<Integer>();
								dataPoints.add(k);
								rightSideDPPerThreshold.put(i,dataPoints);
							}
						}
					}
					
				}
			}
			
			// Calculation of mean square error (MSE)
			HashMap<Integer,Double> labelMSEPerThresholdValue = calculateLabelValueMSE(feature,
					leftSideDPPerThreshold,rightSideDPPerThreshold);
			
			
			Iterator<Entry<Integer, Double>> labelMSEIterator = labelMSEPerThresholdValue.entrySet().iterator();
			Double minMSE = Double.POSITIVE_INFINITY;
			Double lowestLabelMSE = Double.POSITIVE_INFINITY;
			Integer bestThresholdIndex=0;
			while(labelMSEIterator.hasNext()){
				Entry<Integer, Double> entry = labelMSEIterator.next();
				if(entry.getValue() < minMSE){
					minMSE = entry.getValue();
					bestThresholdIndex = entry.getKey();
				}
			}
			
			if(minMSE != Double.POSITIVE_INFINITY)
				lowestLabelMSE = minMSE;
			
			/**
			System.out.println("##################");
			
			for(int i=0;i<noOfThresholdValues;i++){
				System.out.print(totalLabelVariance.get(i));
				if(leftSideDPPerThreshold.get(i) != null)
					System.out.print(" Left #: "+ leftSideDPPerThreshold.get(i).size());
				if(rightSideDPPerThreshold.get(i) != null)
					System.out.print(" Right #: "+rightSideDPPerThreshold.get(i).size());
				System.out.print("\n");
			}
			**/
			
			bestThresholdIndexPerFeature.put(featureIndex,bestThresholdIndex);
			
			Double infoGain = parentNodeMSE - lowestLabelMSE;
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
			
			//leftSideVarianceForFeatureBestThreshold.put(featureIndex,leftSideLabelMSE.get(bestThresholdIndex));
			//System.out.println("Left side variance				: "+leftSideLabelMSE.get(bestThresholdIndex));
			//rightSideVarianceForFeatureBestThreshold.put(featureIndex,rightSideLabelMSE.get(bestThresholdIndex));
			//System.out.println("Right side variance				: "+rightSideLabelMSE.get(bestThresholdIndex));
		}
		
		System.out.println("*********************ALL FEATURES SCANNED************");
		
		createChildNodes(node, rightSideVarianceForFeatureBestThreshold, leftSideVarianceForFeatureBestThreshold, 
				infoGainPerFeatureBestThreshold, bestThresholdIndexPerFeature, leftSideDPForFeaturesBestThreshold, 
				rightSideDPForFeatureBestThreshold);
	}
	
	
	public void createChildNodes(Node node, HashMap<Integer,Double> rightSideVarianceForFeatureBestThreshold,
			HashMap<Integer,Double> leftSideVarianceForFeatureBestThreshold,
			HashMap<Integer,Double> infoGainPerFeatureBestThreshold,
			HashMap<Integer,Integer> bestThresholdIndexPerFeature,
			HashMap<Integer,ArrayList<Integer>> leftSideDPForFeaturesBestThreshold,
			HashMap<Integer,ArrayList<Integer>> rightSideDPForFeatureBestThreshold){
		
		Double higherstInfoGain = 0d;
		Integer bestFeatureIndex = 0;
		Iterator<Entry<Integer, Double>> infoGainIterator = infoGainPerFeatureBestThreshold.entrySet().iterator();
		Double maxValue = Double.NEGATIVE_INFINITY;
		while(infoGainIterator.hasNext()){
			
			Entry<Integer, Double> entry = infoGainIterator.next();
			if(entry.getValue() > maxValue){
				maxValue = entry.getValue();
				bestFeatureIndex = entry.getKey();
			}
		}
		if(maxValue != Double.NEGATIVE_INFINITY)
			higherstInfoGain = maxValue;
		
		System.out.println("Best Feature index is	: "+ bestFeatureIndex);
		System.out.println("Information Gained is	: "+higherstInfoGain);
		Integer bestThresholdValueIndex = bestThresholdIndexPerFeature.get(bestFeatureIndex);
		System.out.println("Left : " + leftSideDPForFeaturesBestThreshold.get(bestFeatureIndex).size());
		System.out.println("Right : "+rightSideDPForFeatureBestThreshold.get(bestFeatureIndex).size());
		node.setThresholdValue(features.get(bestFeatureIndex).getValues().get(bestThresholdValueIndex));
		
		if(higherstInfoGain > Constant.INFO_GAIN_THRESHOLD){
			
			System.out.println("More than threshold information gain");
			
			Feature bestFeature = features.get(bestFeatureIndex);
			
			System.out.println("Best Threshold value	:" + bestFeature.getValues().get(bestThresholdIndexPerFeature.get(bestFeatureIndex)));
			
			if(Constant.VARIANCE_THRESHOLD < leftSideVarianceForFeatureBestThreshold.get(bestFeatureIndex)){
				
				Node leftNode = new Node();
				System.out.println("Left child # of data points "
								+leftSideDPForFeaturesBestThreshold.get(bestFeatureIndex).size());
				leftNode.setDataPoints(leftSideDPForFeaturesBestThreshold.get(bestFeatureIndex));
				leftNode.setParentNode(node);
				leftNode.setMSE(leftSideVarianceForFeatureBestThreshold.get(bestFeatureIndex));
				System.out.println("Left child variance	 		"
						+leftSideVarianceForFeatureBestThreshold.get(bestFeatureIndex));
				node.setLeftChildNode(leftNode);
				
			}
			
			
			if(Constant.VARIANCE_THRESHOLD < rightSideVarianceForFeatureBestThreshold.get(bestFeatureIndex)){
				
				Node rightNode = new Node();
				System.out.println("Right child # of data points "
								+rightSideDPForFeatureBestThreshold.get(bestFeatureIndex).size());
				rightNode.setDataPoints(rightSideDPForFeatureBestThreshold.get(bestFeatureIndex));
				rightNode.setParentNode(node);
				rightNode.setMSE(rightSideVarianceForFeatureBestThreshold.get(bestFeatureIndex));
				System.out.println("Right child variance	 		"
						+rightSideVarianceForFeatureBestThreshold.get(bestFeatureIndex));
				
				node.setRightChildNode(rightNode);
			}
			
		}
		
		node.setLabelValue(calculateAverage(node.getDataPoints()));
	
	}
	
	

	public HashMap<Integer,Double> calculateLabelValueMSE(
			Feature feature,
			HashMap<Integer,ArrayList<Integer>> leftDataPoints,
			HashMap<Integer,ArrayList<Integer>> rightDataPoints){
			
		HashMap<Integer,Double> msePerThreshold = new HashMap<Integer,Double>();
		ArrayList<Double> thresholdValues = feature.getValues();
		int featureIndex = feature.getIndex();
		
		int noOfThresholdValues = thresholdValues.size();
		//System.out.println(((ArrayList)feature.getValues()).size());
		
		for(int i =0 ;i<noOfThresholdValues; i++){
			
			int leftDataPointCount = 0;
			int rightDataPointCount = 0;
			Double leftLabelValue = 0d;
			if(leftDataPoints.containsKey(i)){
				Double leftLabelValueAvg = calculateAverage(leftDataPoints.get(i));
				for(Integer index : leftDataPoints.get(i)){
					leftLabelValue += (dataMatrix.get(index, featureIndex) - leftLabelValueAvg);
					leftDataPointCount++;
				}
			}
			
			Double rightLabelValue = 0d;
			if(rightDataPoints.containsKey(i)){
				Double rightLabelValueAvg = calculateAverage(rightDataPoints.get(i));
				for(Integer index : rightDataPoints.get(i)){
					rightLabelValue += (dataMatrix.get(index, featureIndex) - rightLabelValueAvg);
					rightDataPointCount++;
				}
				
			}
			
			Double MSE = ((leftLabelValue + rightLabelValue) / rightDataPointCount + leftDataPointCount);
			msePerThreshold.put(i, MSE);
			
		}
		return msePerThreshold;
	}
	
	
	/**
	 * This method calculate the variance of the target value for the given 
	 * dataPoints.
	 * @param values : Data Points label value ArrayList
	 * @return
	 */
	public Double calculateVariance(ArrayList<Integer> dataPoints){
		
		DescriptiveStatistics descriptiveStatistics = new DescriptiveStatistics();
		for(Integer row : dataPoints){
			descriptiveStatistics.addValue(dataMatrix.get(row, Constant.TARGET_VALUE_INDEX));
		}
		
		return descriptiveStatistics.getStandardDeviation();
		
	}
	
	/**
	 * 
	 * @param dataPoints : The ArrayList of all Data Point's Index (row/line number)
	 * @return The mean value of all Data Point's label value
	 */
	public Double calculateAverage(ArrayList<Integer> dataPoints){
		
		if(dataPoints.size() > 0){
			DescriptiveStatistics descriptiveStatistics = new DescriptiveStatistics();
			for(Integer row : dataPoints){
				descriptiveStatistics.addValue(dataMatrix.get(row, Constant.TARGET_VALUE_INDEX));
			}
			return descriptiveStatistics.getMean();
		}else{
			return 0d;
		}
		
	}
	
	/**
	 * 
	 * @param dataPoints : Matrix of Data Points
	 * @return The variance of the label value of the given 
	 * Data Points
	 */
	public Double calculateVariance(Matrix dataPoints){
		
		DescriptiveStatistics descriptiveStatistics = new DescriptiveStatistics();
		for(int i=0;i<dataPoints.getRowDimension();i++){
			descriptiveStatistics.addValue(dataMatrix.get(i,Constant.TARGET_VALUE_INDEX));
		}
		return descriptiveStatistics.getStandardDeviation();
	}
	

	/**
	 * This method print the important information of the given Regression Tree
	 * Node
	 * @param node : Regression Tree Node
	 */
	public void printRegressionTreeNode(Node node){
		
		//System.out.println("Feature name: 				" + node.getFeature);
		System.out.println("Feature Threshold Value: 	: " + node.getThresholdValue());
		System.out.println("Label Value					: "+node.getLabelValue());
		
		if(node.getLeftChildNode() != null){
			System.out.println("Left child node :");
			printRegressionTreeNode(node.getLeftChildNode());
			System.out.println("Right child node :");
			printRegressionTreeNode(node.getRightChildNode());
		}
	}
	
}
