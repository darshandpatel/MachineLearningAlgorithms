package ml;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.Queue;

import Jama.Matrix;

public class BasicRegressionTree {
	
	Queue<Node> nodeQueue;
	
	ArrayList<Feature> features;
	LinkedList<Feature> selectedFeature = new LinkedList<Feature>();
	FileOperations fileOperations;
	Matrix dataPoints;
	Integer depthLimit = 15;
	
	public BasicRegressionTree(){
		
		// Create node queue
		nodeQueue = new LinkedList<Node>();
		
		// Create the mapping of DataPoint and its start byte code from the file 
		fileOperations =  new FileOperations();
		dataPoints =  fileOperations.fetchDataPoints();
		features = fetchFeaturePossCriValues(dataPoints);
		
		// Create the root node for Regression Tree and add into Queue
		Node rootNode = new Node();
		rootNode.setDataPoints(dataPoints.getRowDimension());
		rootNode.setVariance(0d);
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
	
}
