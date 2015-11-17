package ml;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import Jama.Matrix;

public class GradientBoosting {
	
	FileOperations fileOperations;
	BasicRegressionTree basicRegressionTree;
	
	public GradientBoosting(){
		
		fileOperations = new FileOperations();
		basicRegressionTree = new BasicRegressionTree();
		
	}
	
	public void performGradientBoosting(){
		
		HashMap<String,Matrix> trainMatrixHashMap = fileOperations.fetchAttributesTargetFromFile(
				Constant.HOUSING_DATA_FILE_PATH,Constant.HOUSING_TRAINDATA_FILE,
				Constant.HOUSING_DATA_NUM_OF_TRAINING_DP, 
				Constant.HOUSING_DATA_NO_OF_FEATURES, Constant.STRING_REGEX);
		
		Matrix attributeMatrix = trainMatrixHashMap.get(Constant.ATTRIBUTES);
		Matrix actualTargetMatrix = trainMatrixHashMap.get(Constant.TARGET);
		Matrix targetMatrix = (Matrix) actualTargetMatrix.clone();
		List<Node> rootNodes = new ArrayList<Node>();
		
		for(int i = 0; i < 10; i++){
			
			System.out.println("Iteration number :" + (i+1));
		
			HashMap<String, Object> hashMap = basicRegressionTree.formRegressionTree(attributeMatrix, targetMatrix);
			Matrix predictedTargetMatrix = (Matrix) hashMap.get(Constant.PREDICTED_TARGET_MATRIX);
			Node rootNode = (Node) hashMap.get(Constant.ROOT_NODE);
			targetMatrix = targetMatrix.minus(predictedTargetMatrix);
			rootNodes.add(rootNode);
			
		}
		
		basicRegressionTree.evaluateTestDataSet(rootNodes);
		
	}
	
	public boolean checkConvergence(Matrix targetMatrix){
		
		int nbrOfRows = targetMatrix.getRowDimension();
		double diff = 0d;
		
		for(int i = 0; i < nbrOfRows; i++){
			
			diff += Math.abs(targetMatrix.get(i, 0));
			
		}
		
		System.out.println("Difference is : " + diff);
		if(diff < Constant.GRADIENT_BOOSTING_CONVERNGE_THRESHOLD)
			return false;
		else
			return true;
		
	}
	
	
	public static void main(String args[]){
		
		GradientBoosting gd = new GradientBoosting();
		gd.performGradientBoosting();
		
	}

}
