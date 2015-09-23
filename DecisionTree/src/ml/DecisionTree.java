package ml;

import java.util.LinkedList;
import java.util.Queue;
import java.util.ArrayList;

import Jama.Matrix;

/**
 * 
 * @author Darshan
 *
 */
public class DecisionTree{

	
	private FileOperations fileOperations;
	Matrix dataMatrix;
	
	
	/**
	 * Constructor : Which fetches the Data Matrix from the train Data set file 
	 * and create the root node of Decision tree.
	 */
	public DecisionTree(){
		
		// Create the mapping of DataPoint and its start byte code from the file 
		fileOperations =  new FileOperations();
		dataMatrix =  fileOperations.fetchDataPointsFromFile(
				Constant.SPAMBASE_DATA_FILE_PATH,Constant.SPAMBASE_DATA_FILE_NAME,
				Constant.SPAMBASE_DATA_NUM_OF_DP,Constant.SPAMBASE_DATA_NUM_OF_FEATURES + 1
				,",");
		
	}
	
	public void formMultipleDecisionTrees(){
		
		int numOfFolds = 10;
		int currentFoldCount = 1;
		
		Queue<Node> nodeQueue;
		Node rootNode;
		
		
		while(currentFoldCount <= numOfFolds){
			
			
			System.out.println("Current Fold number" + currentFoldCount);
			// Form Decision Tree
			Matrix trainDataMatrix = formTrainDataMatrixByFold(numOfFolds,currentFoldCount);
			System.out.println("Train Data Matrix Dimenstions");
			System.out.printf("%s : %d\n%s : %d","# of rows",trainDataMatrix.getRowDimension(),
					"# od columns",trainDataMatrix.getColumnDimension());
			
			Matrix testDataMatrix = formTestDataMatrixByFold(numOfFolds,currentFoldCount);
			System.out.println("Train Data Matrix Dimenstions");
			System.out.printf("%s : %d\n%s : %d","# of rows",testDataMatrix.getRowDimension(),
					"# od columns",testDataMatrix.getColumnDimension());
			
			currentFoldCount++;
			// Evaluate Decision Tree
			
			// Accumulate Results
			
		}
		
		// Print the Results
	}
	
	
	public Matrix formTrainDataMatrixByFold(Integer numOfFolds,Integer currentFoldCount){
		
		int numOfDPPerFold = Constant.SPAMBASE_DATA_NUM_OF_DP / numOfFolds;
		int extraDP = (currentFoldCount != numOfFolds?Constant.SPAMBASE_DATA_NUM_OF_DP % numOfFolds:0);
		
		int numOfDPInTrain = ((numOfFolds - 1) * numOfDPPerFold) + extraDP;
		double trainData[][] = 
				new double[numOfDPInTrain][Constant.SPAMBASE_DATA_NUM_OF_FEATURES];
		int trainDPIndex = 0;
		
		for(int i = 1; i <= numOfFolds; i++){
			
			if(i != currentFoldCount){
				
				int startIndex = (i - 1) * numOfDPPerFold;
				int endIndex = startIndex + (numOfDPPerFold - 1) + (currentFoldCount == numOfFolds?extraDP:0);
				
				while(startIndex <= endIndex){
					for(int j = 0 ; j <Constant.SPAMBASE_DATA_NUM_OF_FEATURES;j++)
						trainData[trainDPIndex][j] = dataMatrix.get(startIndex, j);
					startIndex++;
					trainDPIndex++;
				}
			}
		}
		
		return new Matrix(trainData);
	}
	
	public Matrix formTestDataMatrixByFold(Integer numOfFolds,Integer currentFoldCount){
		
		int numOfDPPerFold = Constant.SPAMBASE_DATA_NUM_OF_DP / numOfFolds;
		int extraDP = (currentFoldCount == numOfFolds?Constant.SPAMBASE_DATA_NUM_OF_DP % numOfFolds:0);
		int numOfDPInTrain = numOfDPPerFold + extraDP;
		double testData[][] = 
				new double[numOfDPInTrain][Constant.SPAMBASE_DATA_NUM_OF_FEATURES];
				
		int startIndex = (currentFoldCount - 1) * numOfDPPerFold;
		int endIndex = startIndex + (numOfDPPerFold - 1) + extraDP;
		int testDPIndex = 0;
		
		while(startIndex <= endIndex){
			for(int j = 0 ; j <Constant.SPAMBASE_DATA_NUM_OF_FEATURES;j++)
				testData[testDPIndex][j] = dataMatrix.get(startIndex, j);
			startIndex++;
			testDPIndex++;
		}
		
		return new Matrix(testData);
	}
}
