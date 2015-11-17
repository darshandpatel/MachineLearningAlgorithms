package ml;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map.Entry;
import java.util.Queue;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import Jama.Matrix;

/**
 * 
 * @author Darshan
 *
 */
public class BasicRegressionTree {
	
	private FileOperations fileOperations;
	
	public BasicRegressionTree(){
		fileOperations = new FileOperations();
	}
	
	public HashMap<String, Object> formRegressionTree(Matrix attributeMatrix, Matrix targetMatrix){
		
		HashMap<String, Object> hashMap = new HashMap<String, Object>();
		
		Node rootNode = new Node();
		Queue<Node> nodeQueue = new LinkedList<Node>(); 
		rootNode.setDataPoints(attributeMatrix.getRowDimension());
		rootNode.setError(calculateSumOfSquareError(rootNode.getDataPoints(), targetMatrix));
		rootNode.setLabelValue(calculateMean(rootNode.getDataPoints(), targetMatrix));
		
		nodeQueue.add(rootNode);
		
		Integer exploredNodeCount = 0;
		Integer exploredNodeLimit = 1;
		for(int i = 0; i < Constant.HOUSING_DATA_DEPTH_LIMIT; i++){
			exploredNodeLimit += (int)Math.pow(2,i);
		}
		
		// Form the tree until Node queue is not empty and the # of explored node 
		// is less the # of explored node limit
		while((nodeQueue.size() > 0) && (exploredNodeCount < exploredNodeLimit)){
			
			Node currentNode = nodeQueue.poll();
			exploredNodeCount++;
			//System.out.println("##############################");
			//System.out.println("Parent node Data Points are	:" + 
			//currentNode.getDataPoints().size());
			//System.out.println("Parent node error is 		:"+currentNode.getError());
			
			splitNodeByBestFeaturethreshold(currentNode,attributeMatrix,targetMatrix);
			
			if(currentNode.getLeftChildNode() != null){
				
				Node leftNode = currentNode.getLeftChildNode();
				if( (leftNode.getError() > Constant.HOUSING_DATA_ERROR_THRESHOLD) &&
						leftNode.getDataPoints().size() > Constant.HOUSING_DATA_NUM_OF_CHILD_THRESHOLD)
					nodeQueue.add(leftNode);
			}
			if(currentNode.getRightChildNode() != null){
				
				Node rightNode = currentNode.getRightChildNode();
				if((rightNode.getError() > Constant.HOUSING_DATA_ERROR_THRESHOLD) &&
						rightNode.getDataPoints().size() > Constant.HOUSING_DATA_NUM_OF_CHILD_THRESHOLD)
					nodeQueue.add(rightNode);
			}
		}
		
		Matrix predictedTargetMatrix = calculatePredictedValue(rootNode, attributeMatrix);
		System.out.println("Predicted Matrix : ");
		printMatrix(predictedTargetMatrix);
		
		hashMap.put(Constant.ROOT_NODE, rootNode);
		hashMap.put(Constant.PREDICTED_TARGET_MATRIX, predictedTargetMatrix);
		
		return hashMap;
		
	}
	
	
	/**
	 * 
	 * @param dataMatrix : The Matrix which contains the training DataSet 
	 * @return ArrayList of Feature, Basically this method parse the dataPoints 
	 * matrix and find the possible threshold value for all the features.
	 */
	public ArrayList<Feature> fetchFeaturePosThreshold(Matrix attributeMatrix, 
			List<Integer> dataPoints){
		
		int noOfColumns = attributeMatrix.getColumnDimension();
		
		HashMap<Integer,ArrayList<Double>> featurePosThreshold = 
				new HashMap<Integer,ArrayList<Double>>(Constant.HOUSING_DATA_NO_OF_FEATURES);
			
		for(int i=0; i < noOfColumns;i++){
			
			String featureName = Constant.HOUSING_DATA_FEATURUES.get(i);
			
			if(featureName.equals("CHAS")){
				ArrayList<Double> values = new ArrayList<Double>();
				values.add(0.5d);
				featurePosThreshold.put(i, values);
			}else{
			
				for(Integer row : dataPoints){
					
					//System.out.println("value of i is : " + i+ " : "+parts[i]);
					//String featureName = Constant.features.get(i);
					if(featurePosThreshold.containsKey(i)){
						Double value = attributeMatrix.get(row,i);
						ArrayList<Double> values = featurePosThreshold.get(i);
						if(!values.contains(value))
							values.add(value);
						
					}else{
						
						ArrayList<Double> values = new ArrayList<Double>();
						values.add(attributeMatrix.get(row,i));
						featurePosThreshold.put(i,values);
					}
				}
			}
		}
		
		return createFeatures(featurePosThreshold);
		
		
	}
	
	
	private ArrayList<Feature> createFeatures(HashMap<Integer , 
			ArrayList<Double>> featurePosThreshold){
	
		ArrayList<Feature> features = new ArrayList<Feature>();
		for(int i = 0; i < Constant.HOUSING_DATA_NO_OF_FEATURES;i++){
			
			String featureName = Constant.HOUSING_DATA_FEATURUES.get(i);
			String featureCtg = null;
			
			ArrayList<Double> calculatedFeaturePosCriValues = featurePosThreshold.get(i);
			Collections.sort(calculatedFeaturePosCriValues);
			
			if(featureName.equals("CHAS")){
				featureCtg = Constant.BINARY_NUM;
			}else{
				featureCtg = Constant.NUMERIC;
				calculatedFeaturePosCriValues = filterFeaturePosThreshold
					(calculatedFeaturePosCriValues);
			}
			
			Feature feature = new Feature(featureName,featureCtg,calculatedFeaturePosCriValues,i);
			//System.out.println("Feature values" + calculatedFeaturePosCriValues.toString());
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
		
		
		int length = calculatedFeaturePosThresholdValues.size();
		if(length < 2){
			return calculatedFeaturePosThresholdValues;
		}else{
			ArrayList<Double> filterdPosThresholdValue= new ArrayList<Double>();
			for(int i=0 ; (i+1) < length; i++){
				Double filteredValue = ((calculatedFeaturePosThresholdValues.get(i) + 
						calculatedFeaturePosThresholdValues.get(i+1))/2);
				filterdPosThresholdValue.add(filteredValue);
			}
			calculatedFeaturePosThresholdValues.clear();
			return filterdPosThresholdValue;
		}
		
	}
	
	
	/**
	 * This method find the best feature and its best threshold value to
	 * split the given Regression tree node.
	 * If the split is possible then the method will create the child nodes
	 * of the given node.
	 * @param node : The node
	 */
	public void splitNodeByBestFeaturethreshold(Node node, Matrix attributeMatrix, 
			Matrix targetMatrix){
		
		Double parentNodeError = node.getError();
		List<Integer> currentNodeDataPoints = node.getDataPoints();
		Integer NoOfCurrentNodeDataPoints = currentNodeDataPoints.size();
		
		ArrayList<Feature> features = fetchFeaturePosThreshold(attributeMatrix,currentNodeDataPoints);
		
		HashMap<Integer,Double> infoGainPerFeatureBestThreshold = new HashMap<Integer,Double>();
		HashMap<Integer,Integer> bestThresholdIndexPerFeature = new HashMap<Integer,Integer>();
		
		HashMap<Integer,ArrayList<Integer>> leftSideDPForFeaturesBestThreshold = 
				new HashMap<Integer,ArrayList<Integer>>();
		HashMap<Integer,ArrayList<Integer>> rightSideDPForFeatureBestThreshold = 
				new HashMap<Integer,ArrayList<Integer>>();
		
		for(Feature feature : features){
			
			//System.out.println("*********************FEATURE STARTS********************");
			
			HashMap<Integer,ArrayList<Integer>> leftSideDPPerThreshold = 
					new HashMap<Integer,ArrayList<Integer>>();
			HashMap<Integer,ArrayList<Integer>> rightSideDPPerThreshold = 
					new HashMap<Integer,ArrayList<Integer>>();
			
			int featureIndex = feature.getIndex();

			List<Double> thresholdValues = feature.getThresholdValues();
			//System.out.println("Feature index"+feature.getIndex());
			//System.out.println("Feature Threshold" + thresholdValues.toString());
			int noOfThresholdValues = thresholdValues.size();
			
			if(feature.getType().equals(Constant.NUMERIC)){
				
				for(int i = 0 ; i < noOfThresholdValues;i++){
					
					for(int k = 0 ; k < NoOfCurrentNodeDataPoints ; k++){
						
						Integer dataPoint = currentNodeDataPoints.get(k);
						Double trainFeatureValue = attributeMatrix.get(dataPoint,featureIndex);
						
						if(trainFeatureValue < thresholdValues.get(i)){
							
							if(leftSideDPPerThreshold.containsKey(i)){
								//System.out.println("Left side node Threshold value Index: "+i+" 
								//, size is : "+leftSideDPPerThreshold.get(i).size());
								leftSideDPPerThreshold.get(i).add(dataPoint);
							}else{
								ArrayList<Integer> dataPoints = new ArrayList<Integer>();
								dataPoints.add(dataPoint);
								leftSideDPPerThreshold.put(i,dataPoints);
							}
						}else{
							if(rightSideDPPerThreshold.containsKey(i)){
								//System.out.println("Right side node Threshold value index : "+i+"
								// Threshold value "+trainFeatureValue+", size is : " + 
								//rightSideDPPerThreshold.get(i).size());
								rightSideDPPerThreshold.get(i).add(dataPoint);
							}else{
								ArrayList<Integer> dataPoints = new ArrayList<Integer>();
								dataPoints.add(dataPoint);
								rightSideDPPerThreshold.put(i,dataPoints);
							}
						}
					}
				}
			}else if(feature.getType().equals(Constant.BINARY_NUM)){
				
				for(int i= 0 ; i< noOfThresholdValues;i++){
					
					for(int k = 0 ; k < NoOfCurrentNodeDataPoints ; k++){
					
						Integer dataPoint = currentNodeDataPoints.get(k);
						Double trainFeatureValue = attributeMatrix.get(dataPoint,featureIndex);
					
						//System.out.println("In side binary");
						//System.out.println("# of different value "+ noOfThresholdValues);
					
						if(trainFeatureValue == 0){
							if(leftSideDPPerThreshold.containsKey(i)){
								leftSideDPPerThreshold.get(i).add(dataPoint);
							}else{
								ArrayList<Integer> dataPoints = new ArrayList<Integer>();
								dataPoints.add(dataPoint);
								leftSideDPPerThreshold.put(i,dataPoints);
							}
						}else{
							if(rightSideDPPerThreshold.containsKey(i)){
								rightSideDPPerThreshold.get(i).add(dataPoint);
							}else{
								ArrayList<Integer> dataPoints = new ArrayList<Integer>();
								dataPoints.add(dataPoint);
								rightSideDPPerThreshold.put(i,dataPoints);
							}
						}
					}
				}
			}
			
			//TODO Only single method can return the best feature index
			// Calculation of square error (SE)
			HashMap<Integer,Double> labelSEPerThresholdValue = calculateLabelValueSE(feature,
					leftSideDPPerThreshold,rightSideDPPerThreshold, attributeMatrix, targetMatrix);
			
			
			Iterator<Entry<Integer, Double>> labelSEIterator = labelSEPerThresholdValue.entrySet().iterator();
			Double lowestLabelSE = Double.POSITIVE_INFINITY;
			Integer bestThresholdIndex=0;
			while(labelSEIterator.hasNext()){
				Entry<Integer, Double> entry = labelSEIterator.next();
				if(entry.getValue() < lowestLabelSE){
					lowestLabelSE = entry.getValue();
					bestThresholdIndex = entry.getKey();
				}
			}
			
			bestThresholdIndexPerFeature.put(featureIndex,bestThresholdIndex);
			
			Double infoGain = parentNodeError - lowestLabelSE;
			infoGainPerFeatureBestThreshold.put(featureIndex,infoGain);
			
			/*
			System.out.println("# of threshold values 	: " + thresholdValues.size());
			System.out.println("Information Gain		: " + infoGain);
			System.out.println("Feature index 			: " + featureIndex);
			System.out.println("Feature type 			: " + feature.getType());
			System.out.println("Threshold value index 	: " + bestThresholdIndex);
			System.out.println("Threshold value 		: " + 
			thresholdValues.get(bestThresholdIndex));
			*/
			
			if(leftSideDPPerThreshold.get(bestThresholdIndex) != null){
				//System.out.println("# of data point on left side: " + 
				//leftSideDPPerThreshold.get(bestThresholdIndex).size());
				leftSideDPForFeaturesBestThreshold.put(featureIndex,
						leftSideDPPerThreshold.get(bestThresholdIndex));
			}
			if(rightSideDPPerThreshold.get(bestThresholdIndex) != null){
				//System.out.println("# of data point on right side: " + 
				//	rightSideDPPerThreshold.get(bestThresholdIndex).size());
				rightSideDPForFeatureBestThreshold.put(featureIndex,
						rightSideDPPerThreshold.get(bestThresholdIndex));
			}
			
		}
		
		//System.out.println("*********************ALL FEATURES SCANNED************");
		
		createChildNodes(node,features,infoGainPerFeatureBestThreshold, bestThresholdIndexPerFeature, 
				leftSideDPForFeaturesBestThreshold,rightSideDPForFeatureBestThreshold, attributeMatrix,
				targetMatrix);
		
	}
	
	
	public void createChildNodes(Node node,ArrayList<Feature> features, 
			HashMap<Integer,Double> infoGainPerFeatureBestThreshold,
			HashMap<Integer,Integer> bestThresholdIndexPerFeature,
			HashMap<Integer,ArrayList<Integer>> leftSideDPForFeaturesBestThreshold,
			HashMap<Integer,ArrayList<Integer>> rightSideDPForFeatureBestThreshold,
			Matrix attributeMatrix, Matrix targetMatrix){
		
		Double higherstInfoGain = 0d;
		Integer bestFeatureIndex = 0;
		Iterator<Entry<Integer, Double>> infoGainIterator = 
				infoGainPerFeatureBestThreshold.entrySet().iterator();
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
		
		//System.out.println("Best Feature index is	: "+ bestFeatureIndex);
		//System.out.println("Information Gained is	: "+ higherstInfoGain);
		
		Feature bestFeature = features.get(bestFeatureIndex);
		
		if(higherstInfoGain > Constant.HOUSING_DATA_INFO_GAIN_THRESHOLD){
			
			Integer bestThresholdValueIndex = bestThresholdIndexPerFeature.get(bestFeatureIndex);
			node.setThresholdValue(bestFeature.getThresholdValues().get(bestThresholdValueIndex));
			node.setFeatureIndex(bestFeature.getIndex());
			//System.out.println("More than threshold information gain");
			
			//System.out.println("Best Threshold value	:" + bestFeature.getThresholdValues()
			//	.get(bestThresholdIndexPerFeature.get(bestFeatureIndex)));
			
			if(leftSideDPForFeaturesBestThreshold.containsKey(bestFeatureIndex)){
				
				Node leftNode = new Node();
				//System.out.println("Left child # of data points "
				//				+leftSideDPForFeaturesBestThreshold.get(bestFeatureIndex).size());
				leftNode.setDataPoints(leftSideDPForFeaturesBestThreshold.get(bestFeatureIndex));
				//System.out.println(leftNode.getDataPoints().toString());
				leftNode.setParentNode(node);
				
				leftNode.setError(calculateSumOfSquareError(leftNode.getDataPoints(),targetMatrix));
				leftNode.setLabelValue(calculateMean(leftNode.getDataPoints(),targetMatrix));
				//System.out.println("LEft label value :"+leftNode.getLabelValue());
				//System.out.println("LEft node error :"+leftNode.getError());
				node.setLeftChildNode(leftNode);
				
			}
			
			if(rightSideDPForFeatureBestThreshold.containsKey(bestFeatureIndex)){
				
				Node rightNode = new Node();
				//System.out.println("Right child # of data points "
				//				+rightSideDPForFeatureBestThreshold.get(bestFeatureIndex).size());
				rightNode.setDataPoints(rightSideDPForFeatureBestThreshold.get(bestFeatureIndex));
				//System.out.println(rightNode.getDataPoints().toString());
				rightNode.setParentNode(node);
				
				rightNode.setError(calculateSumOfSquareError(rightNode.getDataPoints(),targetMatrix));
				rightNode.setLabelValue(calculateMean(rightNode.getDataPoints(),targetMatrix));
				//System.out.println("Right label value :"+rightNode.getLabelValue());
				//System.out.println("Right node error :"+rightNode.getError());
				node.setRightChildNode(rightNode);
			}
			
		}
		
	}
	

	/**
	 * 
	 * @param feature 			: The feature
	 * @param leftDataPoints 	: The left side Data Points after 
	 * the split of the parent node by the given feature
	 * @param rightDataPoints	: The right side Data Points after the
	 * split of the parent node by the given feature 
	 * @return The Standard Error from the split of the Data Points
	 * for the given feature and its all threshold values.
	 */
	public HashMap<Integer,Double> calculateLabelValueSE(
			Feature feature,
			HashMap<Integer,ArrayList<Integer>> leftDataPoints,
			HashMap<Integer,ArrayList<Integer>> rightDataPoints,
			Matrix attributeMatrix, Matrix targetMatrix){
			
		HashMap<Integer,Double> errorPerThreshold = new HashMap<Integer,Double>();
		List<Double> thresholdValues = feature.getThresholdValues();
		
		int noOfThresholdValues = thresholdValues.size();
		
		for(int i =0 ; i < noOfThresholdValues; i++){
			
			Double leftLabelValueError = 0d;
			if(leftDataPoints.containsKey(i)){
				//System.out.println(leftDataPoints.get(i).toString());
				leftLabelValueError = calculateSumOfSquareError(leftDataPoints.get(i),targetMatrix);
			}
			//System.out.println("Left side error : "+leftLabelValueError);
			
			Double rightLabelValueError = 0d;
			if(rightDataPoints.containsKey(i)){
				//System.out.println(rightDataPoints.get(i).toString());
				rightLabelValueError = calculateSumOfSquareError(rightDataPoints.get(i),targetMatrix);
			}
			//System.out.println("Right side error : "+rightLabelValueError);
			
			Double error = leftLabelValueError + rightLabelValueError;
			//System.out.println("Threshold Index : "+ i +"\t Threshold value : "+thresholdValues.get(i) 
			//		+ "\t Error Value : "+error);
			errorPerThreshold.put(i, error);
			
		}
		return errorPerThreshold;
	}
	
	
	
	public Double calculateSumOfSquareError(List<Integer> list, Matrix targetMatrix){
		
		DescriptiveStatistics descriptiveStatistics = new DescriptiveStatistics();
		for(Integer row : list){
			descriptiveStatistics.addValue(targetMatrix.get(row, 0));
		}
		return (descriptiveStatistics.getVariance() * descriptiveStatistics.getN());
	}
	
	public Double calculateMean(List<Integer> list, Matrix targetMatrix){
		
		if(list.size() > 0){
			DescriptiveStatistics descriptiveStatistics = new DescriptiveStatistics();
			for(Integer row : list){
				descriptiveStatistics.addValue(
						targetMatrix.get(row, 0));
			}
			return descriptiveStatistics.getMean();
		}else{
			return 0d;
		}
		
	}
	
	
	public Double calculateMean(Matrix dataPoints, Matrix targetMatrix){
		
		DescriptiveStatistics descriptiveStatistics = new DescriptiveStatistics();
		for(int i=0;i<dataPoints.getRowDimension();i++){
			descriptiveStatistics.addValue(
					targetMatrix.get(i, 0));
		}
		return descriptiveStatistics.getMean();
	}

	/**
	 * This method print the important information of the given Regression Tree
	 * Node
	 * @param node : Regression Tree Root Node
	 */
	public void printRegressionTree(){
		
		Queue<Node> nodeQueue =  new LinkedList<Node>();
		
		Integer count = 1;
		while(nodeQueue.size() > 0){
			
			Node node = nodeQueue.poll();
			
			System.out.printf("%30s\t\t\t:\t%d\n","Node number",count);
			System.out.printf("%30s\t\t\t:\t%f\n","Node Error",node.getError());
			System.out.printf("%30s\t\t\t:\t%d\n","Feature index",node.getFeatureIndex());
			System.out.printf("%30s\t\t\t:\t%f\n","Feature Threshold Value",node.getThresholdValue());
			System.out.printf("%30s\t\t\t:\t%f\n","Label Value",node.getLabelValue());
			System.out.printf("%30s\t\t\t:\t%f\n","Parent Node number",Math.floor(count/2));
			System.out.printf("%30s\t\t\t:\t%d\n\n","Number of Data Points",node.getDataPoints().size());
			
			if(node.getLeftChildNode() != null){
				nodeQueue.add(node.getLeftChildNode());
			}
			if(node.getRightChildNode() != null){
				nodeQueue.add(node.getRightChildNode());
			}
			count++;
		}
	}
	
	/**
	 * This method evaluates the generated Regression Tree Model 
	 * based upon the Test Dataset.
	 * 
	 * It calculates the Mean Standard Error as the measurement of 
	 * accuracy. 
	 */
	public void evaluateTestDataSet(List<Node> roots){
		
		Matrix dataMatrix =  fileOperations.fetchDataPointsFromFile(
				Constant.HOUSING_DATA_FILE_PATH,Constant.HOUSING_TESTDATA_FILE,
				Constant.HOUSING_DATA_NUM_OF_TESTING_DP,Constant.HOUSING_DATA_NO_OF_FEATURES+1,
				Constant.STRING_REGEX);
		double dataArray[][] = dataMatrix.getArray();
		int nbrOfRoots = roots.size();
		
		Double standardError = 0d;
		for(int i=0;i< Constant.HOUSING_DATA_NUM_OF_TESTING_DP;i++){
			
			double predictedValue = 0d;
			for(int j = 0; j < nbrOfRoots; j++){
				predictedValue += predictValue(roots.get(j), dataArray[i]);
			}
			System.out.println("Predicted Value :" + predictedValue +" Actual Value : "+ 
			dataArray[i][Constant.HOUSING_DATA_TARGET_VALUE_INDEX]);
			standardError += Math.pow(dataArray[i][Constant.HOUSING_DATA_TARGET_VALUE_INDEX]
					-predictedValue,2);
		}
		
		System.out.println("Mean Standard Error :" + (standardError/
				Constant.HOUSING_DATA_NUM_OF_TESTING_DP));
		
	}
	
	public double predictValue(Node node, double attributeValue[]){
		
		while(true){

			if(node.getFeatureIndex() != null){
				
				Double featureValue = attributeValue[node.getFeatureIndex()];
				// Check whether flow should go to left or right
				if(featureValue < node.getThresholdValue()){
					node = node.getLeftChildNode();
				}else{
					node = node.getRightChildNode();
				}
				
			}else{
				Double predictedTargetValue = node.getLabelValue();
				return predictedTargetValue;
			}
		}
	}
	
	public Matrix calculatePredictedValue(Node rootNode,Matrix attributeMatrix){
		
		double dataValues[][] = attributeMatrix.getArray();
		int nbrOfRows = attributeMatrix.getRowDimension();
		double predictedValues[][] = new double[nbrOfRows][1];
		
		for(int i = 0; i < nbrOfRows ; i++){
			
			double attributeValue[] = dataValues[i];
			Node node = rootNode;
			while(true){
	
				if(node.getFeatureIndex() != null){
					
					Double featureValue = attributeValue[node.getFeatureIndex()];
					// Check whether flow should go to left or right
					if(featureValue < node.getThresholdValue()){
						node = node.getLeftChildNode();
					}else{
						node = node.getRightChildNode();
					}
					
				}else{
					Double predictedTargetValue = node.getLabelValue();
					predictedValues[i][0] = predictedTargetValue;
					break;
				}
			}
		}
		
		return new Matrix(predictedValues);
		
	}
	
	void printMatrix(Matrix matrix){
		
		int nbrOfRows = matrix.getRowDimension();
		int nbrOfColumns = matrix.getColumnDimension();
		
		for(int i=0; i<nbrOfRows; i++){
			
			for(int j=0; j<nbrOfColumns;j++){
				
				System.out.print(matrix.get(i, j));
				
			}
			System.out.println();
		}
		
	}
	
}
