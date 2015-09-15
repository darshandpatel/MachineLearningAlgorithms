package ml;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.Queue;

public class BasicRegressionTree {
	
	Queue<Node> nodeQueue;
	
	ArrayList<Feature> features;
	LinkedList<Feature> selectedFeature = new LinkedList<Feature>();
	FileOperations fileOperations;
	HashMap<Integer,Double> bytesForTrainDataSetLines;
	Integer depthLimit = 2;
	
	public BasicRegressionTree(){
		
		// Create node queue
		nodeQueue = new LinkedList<Node>();
		
		// Create the mapping of DataPoint and its start byte code from the file 
		fileOperations =  new FileOperations();
		bytesForTrainDataSetLines = fileOperations.getBytePosOfLine(Constant.TRAIN);
		
		// Create the root node for Regression Tree and add into Queue
		Node rootNode = new Node();
		rootNode.setDataPoints(bytesForTrainDataSetLines.size());
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
	
}
