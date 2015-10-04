package ml;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.function.Function;

import Jama.Matrix;

public class NeuralNetworks {
	
	HashMap<Integer,Matrix> weightMap;
	HashMap<Integer,Matrix> biasMap;
	HashMap<Integer,Matrix> errorMap;
	Matrix targetMatrix;
	Matrix inputMatrix; 
	HashMap<Integer,Integer> unitsByLayer;
	
	public NeuralNetworks(){
		
		double input[][] = {{1,0,0,0,0,0,0,0},
				{0,1,0,0,0,0,0,0},
				{0,0,1,0,0,0,0,0},
				{0,0,0,1,0,0,0,0},
				{0,0,0,0,1,0,0,0},
				{0,0,0,0,0,1,0,0},
				{0,0,0,0,0,0,1,0},
				{0,0,0,0,0,0,0,1}};
		
		inputMatrix = new Matrix(input);
		targetMatrix = new Matrix(input);
		
		unitsByLayer = new HashMap<Integer,Integer>();
		unitsByLayer.put(0, 8);
		unitsByLayer.put(1, 3);
		unitsByLayer.put(2, 8);
		
		errorMap = new HashMap<Integer,Matrix>();
		generateRandomWeights();
		generateBias();
		
	}
	
	
	public void generateRandomWeights(){
		
		weightMap = new HashMap<Integer,Matrix>();
		
		// Middle Hidden Layer Weights
		weightMap.put(1, Matrix.random(3,8));
		// Last Layer Weights
		weightMap.put(2, Matrix.random(8,3));
		
	}
	
	
	public void generateBias(){
		
		biasMap = new HashMap<Integer,Matrix>();
		
		// Middle Hidden Layer Weights
		biasMap.put(1, Matrix.random(1,3));
		// Last Layer Weights
		biasMap.put(2, Matrix.random(1,8));
		
	}
	
	
	public void formNeuralNetwork(){
		
		int noOfInputDataRow = inputMatrix.getRowDimension();
		int noOfInputDataColumn = inputMatrix.getColumnDimension();
		
		int loopCount = 1;
		
		while(true){
			
			System.out.println("Loop count :" + loopCount);
			HashMap<Integer,HashMap<String,Matrix>> inputOutputMatrixByLayer 
				= new HashMap<Integer,HashMap<String,Matrix>>();
			
			for(int i = 0; i < noOfInputDataRow; i++){
				
				// Calculate the Input and Output for all units 
				int row[] = {i};
				
				Matrix target = targetMatrix.getMatrix(row, 0, noOfInputDataColumn-1);
				Matrix input = inputMatrix.getMatrix(row, 0, noOfInputDataColumn-1);

				Matrix output = (Matrix) input.clone();
				
				HashMap<String,Matrix> inputOutputMap = new HashMap<String,Matrix>();
				inputOutputMap.put(Constant.INPUT, input);
				inputOutputMap.put(Constant.OUTPUT, output);
				
				inputOutputMatrixByLayer.put(0, inputOutputMap);
				
				for(int k = 1 ; k < Constant.NUM_OF_NN_LAYERS; k++){
					
					inputOutputMap = new HashMap<String,Matrix>();
					
					// Calculate the input values of a unit by multiplying weight factor with previous layer
					// output values.
					Matrix previousLayerOutputMatrix = 
							inputOutputMatrixByLayer.get(k-1).get(Constant.OUTPUT);
					Matrix calculatedInputMatrix = 
							previousLayerOutputMatrix.times(weightMap.get(k).transpose()).plus(biasMap.get(k));
					
					inputOutputMap.put(Constant.INPUT, calculatedInputMatrix);
					
					// Calculate the output of the current unit
					inputOutputMap.put(Constant.OUTPUT, applySigmoidMapFunction(calculatedInputMatrix));
					
					inputOutputMatrixByLayer.put(k, inputOutputMap);
					
				}
				
				errorMap.clear();
				Matrix outputMatrix;
				
				// Calculate the error for all units at last layer
				
				int maxUnits = unitsByLayer.get(Constant.NUM_OF_NN_LAYERS-1);
				Matrix errorMatrix = new Matrix(1, maxUnits);
				for(int p = 0 ; p < maxUnits;p++){
					
					outputMatrix = inputOutputMatrixByLayer.get(Constant.NUM_OF_NN_LAYERS-1).
							get(Constant.OUTPUT);
					Double calculatedOutputValue = outputMatrix.get(0,p);
					Double targetValue = target.get(0, p);
					errorMatrix.set(0, p, (calculatedOutputValue*
							(1-calculatedOutputValue)*(targetValue-calculatedOutputValue)));
					
				}
				errorMap.put(Constant.NUM_OF_NN_LAYERS-1, errorMatrix);
				
				// Calculate the error for all units except first and last layer units
				
				for(int k = (Constant.NUM_OF_NN_LAYERS-2) ; k > 0; k--){
					
					maxUnits = unitsByLayer.get(k);
					errorMatrix = new Matrix(1, maxUnits);
					
					Matrix previousLayerMatrix = errorMap.get(k+1).times(weightMap.get(k+1));
					for(int p = 0 ; p < maxUnits;p++){
						
						outputMatrix = inputOutputMatrixByLayer.get(k).
								get(Constant.OUTPUT);
						Double calculatedOutputValue = outputMatrix.get(0,p);
						errorMatrix.set(0, p, (calculatedOutputValue*(1-calculatedOutputValue)
								*previousLayerMatrix.get(0, p)));
						
					}
					errorMap.put(k, errorMatrix);
				}
				
				// Update the weights based on error
				Iterator<Map.Entry<Integer,Matrix>> iterator= weightMap.entrySet().iterator();
				while(iterator.hasNext()){
					
					Entry<Integer,Matrix> entry = iterator.next();
					Matrix weight = entry.getValue();
					int layer = entry.getKey();
					
					int weightRow = weight.getRowDimension();
					int weightColumn = weight.getColumnDimension();
					
					Matrix times = errorMap.get(layer).transpose().times(
							inputOutputMatrixByLayer.get(layer).get(Constant.OUTPUT));
							
					for(int x = 0; x < weightRow;x++){
						for(int y=0; y< weightColumn;y++){
							
							weight.set(x, y,(weight.get(x, y) + 
									(Constant.NN_LEARNING_RATE*times.get(0, 0))));
							
						}
					}
					
					Matrix bias = biasMap.get(layer);
					bias = bias.plus(errorMap.get(layer).times(Constant.NN_LEARNING_RATE));  
					
				}
			}
			
			loopCount++;
			
			}
	}
	
	public Matrix applySigmoidMapFunction(Matrix input){
		
		int noOfRows = input.getRowDimension();
		int noOfColumns = input.getColumnDimension();
		double[][] output = new double[noOfRows][noOfColumns];
		
		for (int i = 0; i < noOfRows; i++) {
			for (int j = 0; j < noOfColumns; j++) {
				
				output[i][j]  = (double)(1 /
						(1 + Math.pow(Math.E,(-1*input.get(i, j)))));
				
				}
			}
		
		return new Matrix(output);
	}
	
	/**
	 * This method print the given Matrix
	 * @param matrix : Data Matrix
	 */
	private void printMatrix(Matrix matrix){
		
		int noOfRows = matrix.getRowDimension();
		int noOfColumns = matrix.getColumnDimension();
		
		for(int i=0; i<noOfRows ; i++){
			for(int j=0;j<noOfColumns;j++){
				System.out.print(matrix.get(i, j)+" ");
			}
			System.out.print("\n");
		}
	}
	
}
