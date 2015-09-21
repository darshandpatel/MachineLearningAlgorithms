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
	LinkedList<Feature> selectedFeature = new LinkedList<Feature>();
	FileOperations fileOperations;
	Matrix dataMatrix;
	
	
	/**
	 * 
	 * @return The Root node of Regression Tree
	 */
	Node getRootNode(){
		return rootNode;
	}
	
	/**
	 * Constructor : Which fetches the Data Matrix from the train Data set file 
	 * and create the root node of regression tree.
	 */
	public BasicRegressionTree(){
		
		// Create node queue
		nodeQueue = new LinkedList<Node>();
		
		// Create the mapping of DataPoint and its start byte code from the file 
		fileOperations =  new FileOperations();
		dataMatrix =  fileOperations.fetchDataPoints();
		
		// Create the root node for Regression Tree and add into Queue
		Node rootNode = new Node();
		rootNode.setDataPoints(dataMatrix.getRowDimension());
		rootNode.setError(calculateError(dataMatrix));
		
		rootNode.setError(calculateError(rootNode.getDataPoints()));
		rootNode.setLabelValue(calculateMean(rootNode.getDataPoints()));
		
		this.rootNode = rootNode;
		nodeQueue.add(rootNode);
		
	}
	
	/**
	 * This method forms the regression tree on the given train data set.
	 */
	public void formRegressionTree(){
		
		Integer exploredNodeCount = 0;
		Integer exploredNodeLimit = (1 + (int)Math.pow(2,Constant.DEPTH_LIMIT));
		
		// Form the tree until Node queue is not empty and the # of explored node 
		// is less the # of explored node limit
		while( (nodeQueue.size() > 0) && (exploredNodeCount < exploredNodeLimit)){
			
			Node currentNode = nodeQueue.poll();
			
			System.out.println("##############################");
			System.out.println("Parent node Data Points are	:"+currentNode.getDataPoints().size());
			System.out.println("Parent node error is 		:"+currentNode.getError());
			
			splitNodeByFeaturethreshold(currentNode);
			
			currentNode.getDataPoints().clear();
			
			if(currentNode.getLeftChildNode() != null){
				
				Node leftNode = currentNode.getLeftChildNode();
				if(leftNode.getError() > Constant.ERROR_THRESHOLD)
					nodeQueue.add(currentNode.getLeftChildNode());
				
			}
			if(currentNode.getRightChildNode() != null){
				
				Node rightNode = currentNode.getRightChildNode();
				if(rightNode.getError() > Constant.ERROR_THRESHOLD)
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
	public ArrayList<Feature> fetchFeaturePossThreshold(Matrix dataMatrix, ArrayList<Integer> dataPoints){
		
		int noOfColumns = dataMatrix.getColumnDimension();
		
		ArrayList<Feature> features = new ArrayList<Feature>();
		HashMap<Integer,ArrayList<Double>> featurePosCriValues = 
				new HashMap<Integer,ArrayList<Double>>(Constant.NO_OF_FEATURES);
			
		for(int i=0; i < (noOfColumns -1);i++){
			
			for(Integer row : dataPoints){
				
				//System.out.println("value of i is : " + i+ " : "+parts[i]);
				//String featureName = Constant.features.get(i);
				if(featurePosCriValues.containsKey(i)){
					Double value = dataMatrix.get(row,i);
					ArrayList<Double> values = featurePosCriValues.get(i);
					if(!values.contains(value))
						values.add(value);
					
				}else{
					
					ArrayList<Double> values = new ArrayList<Double>();
					values.add(dataMatrix.get(row,i));
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
				calculatedFeaturePosCriValues = filterFeaturePosThreshold
					(calculatedFeaturePosCriValues);
			}
			
			Feature feature = new Feature(featureName,featureCtg,calculatedFeaturePosCriValues,i);
			System.out.println("Feature values" + calculatedFeaturePosCriValues.toString());
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
		
		Double parentNodeError = node.getError();
		ArrayList<Integer> currentNodeDataPoints = node.getDataPoints();
		Integer NoOfCurrentNodeDataPoints = currentNodeDataPoints.size();
		
		ArrayList<Feature> features = fetchFeaturePossThreshold(dataMatrix,currentNodeDataPoints);
		
		HashMap<Integer,Double> infoGainPerFeatureBestThreshold = new HashMap<Integer,Double>();
		HashMap<Integer,Integer> bestThresholdIndexPerFeature = new HashMap<Integer,Integer>();
		
		HashMap<Integer,ArrayList<Integer>> leftSideDPForFeaturesBestThreshold = 
				new HashMap<Integer,ArrayList<Integer>>();
		HashMap<Integer,ArrayList<Integer>> rightSideDPForFeatureBestThreshold = 
				new HashMap<Integer,ArrayList<Integer>>();
		
		
		
		for(Feature feature : features){
			
			System.out.println("*********************FEATURE STARTS********************");
			
			HashMap<Integer,ArrayList<Integer>> leftSideDPPerThreshold = 
					new HashMap<Integer,ArrayList<Integer>>();
			HashMap<Integer,ArrayList<Integer>> rightSideDPPerThreshold = 
					new HashMap<Integer,ArrayList<Integer>>();
			
			
			int featureIndex = feature.getIndex();

			ArrayList<Double> thresholdValues = feature.getValues();
			
			System.out.println("Feature Threshold" + thresholdValues.toString());
			int noOfThresholdValues = thresholdValues.size();
			
			if(feature.getType().equals(Constant.NUMERIC)){
				
				for(int i = 0 ; i < noOfThresholdValues;i++){
					
					for(int k = 0 ; k < NoOfCurrentNodeDataPoints ; k++){
						
						Integer dataPoint = currentNodeDataPoints.get(k);
						Double trainFeatureValue = dataMatrix.get(dataPoint,featureIndex);
						
						if(trainFeatureValue < thresholdValues.get(i)){
							
							if(leftSideDPPerThreshold.containsKey(i)){
								//System.out.println("Index : "+i+" , size is : "+leftSideDPPerThreshold.get(i).size());
								leftSideDPPerThreshold.get(i).add(dataPoint);
							}else{
								ArrayList<Integer> dataPoints = new ArrayList<Integer>();
								dataPoints.add(dataPoint);
								leftSideDPPerThreshold.put(i,dataPoints);
							}
								
						}else{
							
							if(rightSideDPPerThreshold.containsKey(i)){
								//System.out.println("Index : "+i+" , size is : "+rightSideDPPerThreshold.get(i).size());
								rightSideDPPerThreshold.get(i).add(dataPoint);
							}else{
								ArrayList<Integer> dataPoints = new ArrayList<Integer>();
								dataPoints.add(dataPoint);
								rightSideDPPerThreshold.put(i,dataPoints);
							}
						}
						
					}
					
					
					/**
					// Display Data Point for each feature threshold value split
					if(leftSideDPPerThreshold.containsKey(i)){
						System.out.println("Left side data points :");
						for(Integer dataPoint : leftSideDPPerThreshold.get(i)){
							System.out.println(dataMatrix.get(dataPoint,Constant.TARGET_VALUE_INDEX));
						}
					}
					if(rightSideDPPerThreshold.containsKey(i)){
						System.out.println("Right side data points :");
						for(Integer dataPoint : rightSideDPPerThreshold.get(i)){
							System.out.println(dataMatrix.get(dataPoint,Constant.TARGET_VALUE_INDEX));
						}
						
					}
					**/
					
				}
			}else if(feature.getType().equals(Constant.BINARY_NUM)){
				
				for(int i= 0 ; i< noOfThresholdValues;i++){
					
					for(int k = 0 ; k < NoOfCurrentNodeDataPoints ; k++){
					
						Integer dataPoint = currentNodeDataPoints.get(k);
						Double trainFeatureValue = dataMatrix.get(dataPoint,featureIndex);
					
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
			
			// Calculation of square error (SE)
			HashMap<Integer,Double> labelSEPerThresholdValue = calculateLabelValueSE(feature,
					leftSideDPPerThreshold,rightSideDPPerThreshold);
			
			
			Iterator<Entry<Integer, Double>> labelSEIterator = labelSEPerThresholdValue.entrySet().iterator();
			Double minMSE = Double.POSITIVE_INFINITY;
			Double lowestLabelSE = Double.POSITIVE_INFINITY;
			Integer bestThresholdIndex=0;
			while(labelSEIterator.hasNext()){
				Entry<Integer, Double> entry = labelSEIterator.next();
				if(entry.getValue() < minMSE){
					minMSE = entry.getValue();
					bestThresholdIndex = entry.getKey();
				}
			}
			
			if(minMSE != Double.POSITIVE_INFINITY)
				lowestLabelSE = minMSE;
			
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
			
			Double infoGain = parentNodeError - lowestLabelSE;
			infoGainPerFeatureBestThreshold.put(featureIndex,infoGain);
			
			System.out.println("# of threshold values 	: " + thresholdValues.size());
			System.out.println("Information Gain		: " + infoGain);
			System.out.println("Feature index 			: " + featureIndex);
			System.out.println("Feature type 			: " + feature.getType());
			System.out.println("Threshold value index 	: " + bestThresholdIndex);
			System.out.println("Threshold value 		: " + thresholdValues.get(bestThresholdIndex));
			
			if(leftSideDPPerThreshold.get(bestThresholdIndex) != null){
				//System.out.println("# of datapoint on left side: "+leftSideDPPerThreshold.get(bestThresholdIndex).size());
				leftSideDPForFeaturesBestThreshold.put(featureIndex,leftSideDPPerThreshold.get(bestThresholdIndex));
			}
			if(rightSideDPPerThreshold.get(bestThresholdIndex) != null){
				//System.out.println("# of datapoint on right side: "+rightSideDPPerThreshold.get(bestThresholdIndex).size());
				rightSideDPForFeatureBestThreshold.put(featureIndex,rightSideDPPerThreshold.get(bestThresholdIndex));
			}
			
		}
		
		System.out.println("*********************ALL FEATURES SCANNED************");
		
		createChildNodes(node,features,infoGainPerFeatureBestThreshold, bestThresholdIndexPerFeature, 
				leftSideDPForFeaturesBestThreshold,rightSideDPForFeatureBestThreshold);
		
	}
	
	
	public void createChildNodes(Node node,ArrayList<Feature> features, 
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
		System.out.println("Information Gained is	: "+ higherstInfoGain);
		
		Feature bestFeature = features.get(bestFeatureIndex);
		
		if(higherstInfoGain > Constant.INFO_GAIN_THRESHOLD){
			
			Integer bestThresholdValueIndex = bestThresholdIndexPerFeature.get(bestFeatureIndex);
			node.setThresholdValue(bestFeature.getValues().get(bestThresholdValueIndex));
			
			System.out.println("More than threshold information gain");
			
			System.out.println("Best Threshold value	:" + bestFeature.getValues().get(bestThresholdIndexPerFeature.get(bestFeatureIndex)));
			
			rootNode.setError(calculateError(rootNode.getDataPoints()));
			rootNode.setLabelValue(calculateMean(rootNode.getDataPoints()));
			
			if(leftSideDPForFeaturesBestThreshold.containsKey(bestFeatureIndex)){
				
				Node leftNode = new Node();
				System.out.println("Left child # of data points "
								+leftSideDPForFeaturesBestThreshold.get(bestFeatureIndex).size());
				leftNode.setDataPoints(leftSideDPForFeaturesBestThreshold.get(bestFeatureIndex));
				leftNode.setParentNode(node);
				
				leftNode.setError(calculateError(leftNode.getDataPoints()));
				leftNode.setLabelValue(calculateMean(leftNode.getDataPoints()));
				
				System.out.println("LEft node error :"+leftNode.getError());
				node.setLeftChildNode(leftNode);
				
			}
			
		
			if(rightSideDPForFeatureBestThreshold.containsKey(bestFeatureIndex)){
				
				Node rightNode = new Node();
				System.out.println("Right child # of data points "
								+rightSideDPForFeatureBestThreshold.get(bestFeatureIndex).size());
				rightNode.setDataPoints(rightSideDPForFeatureBestThreshold.get(bestFeatureIndex));
				System.out.println(rightNode.getDataPoints().toString());
				rightNode.setParentNode(node);
				
				rightNode.setError(calculateError(rightNode.getDataPoints()));
				rightNode.setLabelValue(calculateMean(rightNode.getDataPoints()));
				
				node.setRightChildNode(rightNode);
			}
			
		}else
			System.out.println("Less than threshold value");
		
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
			HashMap<Integer,ArrayList<Integer>> rightDataPoints){
			
		HashMap<Integer,Double> errorPerThreshold = new HashMap<Integer,Double>();
		ArrayList<Double> thresholdValues = feature.getValues();
		
		int noOfThresholdValues = thresholdValues.size();
		//System.out.println(((ArrayList)feature.getValues()).size());
		
		for(int i =0 ; i < noOfThresholdValues; i++){
			
			Double leftLabelValueError = 0d;
			if(leftDataPoints.containsKey(i)){
				//System.out.println(leftDataPoints.get(i).toString());
				leftLabelValueError = calculateError(leftDataPoints.get(i)) * leftDataPoints.get(i).size();
			}
			//System.out.println("Left side error : "+leftLabelValueError);
			
			Double rightLabelValueError = 0d;
			if(rightDataPoints.containsKey(i)){
				//System.out.println(rightDataPoints.get(i).toString());
				rightLabelValueError = calculateError(rightDataPoints.get(i)) * rightDataPoints.get(i).size();
			}
			//System.out.println("Right side error : "+rightLabelValueError);
			
			Double error = leftLabelValueError + rightLabelValueError;
			//System.out.println("Threshold Index : "+ i +"\t Threshold value : "+thresholdValues.get(i) 
			//		+ "\t Error Value : "+error);
			errorPerThreshold.put(i, error);
			
		}
		return errorPerThreshold;
	}
	
	
	/**
	 * 
	 * @param dataPoints : The matrix of Data Points
	 * @return The Error (Variance * # of records) of the target value from
	 * the given Data Point matrix
	 */
	public Double calculateError(Matrix dataPoints){
		
		DescriptiveStatistics descriptiveStatistics = new DescriptiveStatistics();
		for(int i=0;i<dataPoints.getRowDimension();i++){
			descriptiveStatistics.addValue(dataMatrix.get(i,Constant.TARGET_VALUE_INDEX));
		}
		return descriptiveStatistics.getVariance() * descriptiveStatistics.getN();
		
	}
	
	/**
	 * 
	 * @param values : Data Points label value ArrayList
	 * @return The Error (Variance * # of records) of the target value from
	 * the given Data Point ArrayList
	 */
	public Double calculateError(ArrayList<Integer> dataPoints){
		
		DescriptiveStatistics descriptiveStatistics = new DescriptiveStatistics();
		for(Integer row : dataPoints){
			descriptiveStatistics.addValue(dataMatrix.get(row, Constant.TARGET_VALUE_INDEX));
		}
		
		return (descriptiveStatistics.getVariance() * descriptiveStatistics.getN());
		
	}
	
	/**
	 * 
	 * @param dataPoints : The ArrayList of all Data Point's Index (row/line number)
	 * @return The mean value of all Data Point's label value
	 */
	public Double calculateMean(ArrayList<Integer> dataPoints){
		
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
	 * @return The mean of the label value of the given 
	 * Data Points
	 */
	public Double calculateMean(Matrix dataPoints){
		
		DescriptiveStatistics descriptiveStatistics = new DescriptiveStatistics();
		for(int i=0;i<dataPoints.getRowDimension();i++){
			descriptiveStatistics.addValue(dataMatrix.get(i,Constant.TARGET_VALUE_INDEX));
		}
		return descriptiveStatistics.getMean();
	}

	/**
	 * This method print the important information of the given Regression Tree
	 * Node
	 * @param node : Regression Tree Node
	 */
	public void printRegressionTreeNode(Node node){
		
		
		System.out.println("Feature Threshold Value		:	" + node.getThresholdValue());
		System.out.println("Label Value					:	"+node.getLabelValue());
		
		if(node.getLeftChildNode() != null){
			System.out.println("Left child node :");
			printRegressionTreeNode(node.getLeftChildNode());
		}
		if(node.getRightChildNode() != null){
			System.out.println("Right child node :");
			printRegressionTreeNode(node.getRightChildNode());
		}
	}
	
}
