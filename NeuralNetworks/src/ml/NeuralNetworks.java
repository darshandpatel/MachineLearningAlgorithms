package ml;

import java.util.HashMap;

import Jama.Matrix;

public class NeuralNetworks {
	
	HashMap<Integer,Matrix> weightMap;
	HashMap<Integer,Matrix> biasMap;
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
		
		generateRandomWeights();
		generateBias();
		
	}
	
	
	public void generateRandomWeights(){
		
		weightMap = new HashMap<Integer,Matrix>();
		
		// Middle Hidden Layer Weights
		weightMap.put(0, Matrix.random(8,3));
		// Last Layer Weights
		weightMap.put(1, Matrix.random(3,8));
		
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
		
		printWeights();
		printBias();
		
		HashMap<Integer,HashMap<String,Matrix>> inputOutputMatrixByLayer = null;
		HashMap<Integer,Matrix> errorMatrixByLayer = null;
		Matrix finalLayerOutputMatrix = null;
		
		do{
			
			System.out.println("Loop count :" + loopCount);
			
			for(int i = 0; i < noOfInputDataRow; i++){
				
				// Calculate the Input and Output for all units 
				int row[] = {i};
				
				Matrix target = targetMatrix.getMatrix(row, 0, noOfInputDataColumn-1);
				Matrix input = inputMatrix.getMatrix(row, 0, noOfInputDataColumn-1);
				
				inputOutputMatrixByLayer = formInputOutputsByLayers(input);
				
				// Calculate the error for all units at last layer
				errorMatrixByLayer = calculateErrors(inputOutputMatrixByLayer,target);
				
				// Update the weights based on error
				updateWeightAndBias(inputOutputMatrixByLayer,errorMatrixByLayer);
				
			}
			
			finalLayerOutputMatrix = inputOutputMatrixByLayer.get(Constant.NUM_OF_NN_LAYERS-1).get(Constant.OUTPUT);
			loopCount++;
			
		}while(calculatePredictionError(finalLayerOutputMatrix) > Constant.ERROR_THRESHOLD);
		
		printNNResults();
		printBias();
		printWeights();
		
	}
	
	public Matrix applySigmoidMapFunction(Matrix input){
		
		int noOfRows = input.getRowDimension();
		int noOfColumns = input.getColumnDimension();
		double[][] output = new double[noOfRows][noOfColumns];
		
		for (int i = 0; i < noOfRows; i++) {
			for (int j = 0; j < noOfColumns; j++) {
				
				output[i][j]  = (double)(Math.pow(Math.E,input.get(i, j)) /
						(1 + Math.pow(Math.E,input.get(i, j))));
				
				}
			}
		
		return new Matrix(output);
	}
	
	public void printBias(){
		
		System.out.println("Bias values");
		for(int l = 1 ; l < Constant.NUM_OF_NN_LAYERS;l++){
			
			Matrix biasMatrix = biasMap.get(l);
			int r = biasMatrix.getRowDimension();
			int c = biasMatrix.getColumnDimension();
			
			for(int m=0;m<r;m++){
				for(int n=0;n<c;n++){
					
					System.out.printf("%5.3f",biasMatrix.get(m, n));
				}
				System.out.print("\n");
			}
		}
		
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
				System.out.printf("%5.3f ",matrix.get(i, j));
			}
			System.out.print("\n");
		}
	}
	
	private void printWeights(){
		
		System.out.println("Weight Matrix");
		for(int l = 0;l<=(Constant.NUM_OF_NN_LAYERS-2);l++){
			
			Matrix weight = weightMap.get(l);
			
			int r = weight.getRowDimension();
			int c = weight.getColumnDimension();
			
			for(int m=0;m<r;m++){
				for(int n=0;n<c;n++){
					
					System.out.printf("%5.3f",weight.get(m, n));
				}
				System.out.print("\n");
			}
			
		}
		
	}
	
	private void printNNResults(){
		
		int noOfInputDataRow = inputMatrix.getRowDimension();
		int noOfInputDataColumn = inputMatrix.getColumnDimension();
		
		
		for(int i = 0; i < noOfInputDataRow; i++){
			
			System.out.println("******************************");
			// Calculate the Input and Output for all units 
			int row[] = {i};
			
			System.out.println("Input Matrix");
			Matrix input = inputMatrix.getMatrix(row, 0, noOfInputDataColumn-1);
			printMatrix(input);
			
			Matrix hiddenLayerWeight = weightMap.get(0);
			
			Matrix middleLayerInput = input.times(hiddenLayerWeight);
			
			System.out.println("Middle Layer Input");
			printMatrix(middleLayerInput);
			
			Matrix middleLayerOutput = applySigmoidMapFunction(middleLayerInput);
			
			System.out.println("Middle Layer Output");
			printMatrix(middleLayerOutput);
			
			Matrix lastLayerWeight = weightMap.get(1);
			Matrix lastLayerInput = middleLayerOutput.times(lastLayerWeight);
			
			System.out.println("Last Layer Input");
			printMatrix(lastLayerInput);
			
			Matrix lastLayerOutput = applySigmoidMapFunction(lastLayerInput);
			
			System.out.println("Last Layer Output");
			printMatrix(lastLayerOutput);
		}
		
	}
	
	private HashMap<Integer,HashMap<String,Matrix>> formInputOutputsByLayers(Matrix input){
		
		HashMap<Integer,HashMap<String,Matrix>> inputOutputMatrixByLayer 
			= new HashMap<Integer,HashMap<String,Matrix>>();
		
		// First layer out matrix is same as the input matrix
		Matrix output = (Matrix) input.clone();
		
		// Create the HashMap for the first layer Input Output Matrix
		HashMap<String,Matrix> inputOutputMap = new HashMap<String,Matrix>();
		inputOutputMap.put(Constant.INPUT, input);
		inputOutputMap.put(Constant.OUTPUT, output);
		
		inputOutputMatrixByLayer.put(0, inputOutputMap);
		
		//Calculate the Input and Output matrix for the layers except the first layer.
		
		for(int k = 1 ; k < Constant.NUM_OF_NN_LAYERS; k++){
			
			inputOutputMap = new HashMap<String,Matrix>();
			
			// Calculate the input values of a unit by multiplying weight factor with previous layer
			// output values.
			Matrix previousLayerOutputMatrix = 
					inputOutputMatrixByLayer.get(k-1).get(Constant.OUTPUT);
			Matrix calculatedInputMatrix = 
					previousLayerOutputMatrix.times(weightMap.get(k-1)).plus(biasMap.get(k));
			
			inputOutputMap.put(Constant.INPUT, calculatedInputMatrix);
			
			// Calculate the output of the current unit
			inputOutputMap.put(Constant.OUTPUT, applySigmoidMapFunction(calculatedInputMatrix));
			
			inputOutputMatrixByLayer.put(k, inputOutputMap);
			
		}
		
		return inputOutputMatrixByLayer;
		
	}
	
	private HashMap<Integer,Matrix> calculateErrors(
			HashMap<Integer,HashMap<String,Matrix>> inputOutputMatrixByLayer,
			Matrix target){
		
		HashMap<Integer,Matrix> errorMatrixByLayer = new HashMap<Integer,Matrix>();
		
		int maxUnits = unitsByLayer.get(Constant.NUM_OF_NN_LAYERS-1);
		
		Matrix errorMatrix = new Matrix(1, maxUnits);
		Matrix lastLayerOutputMatrix = inputOutputMatrixByLayer.get(Constant.NUM_OF_NN_LAYERS-1).
				get(Constant.OUTPUT);
		
		for(int p = 0 ; p < maxUnits;p++){
			
			Double calculatedOutputValue = lastLayerOutputMatrix.get(0,p);
			Double targetValue = target.get(0, p);
			errorMatrix.set(0, p, (calculatedOutputValue*
					(1-calculatedOutputValue)*(targetValue-calculatedOutputValue)));
			
		}
		errorMatrixByLayer.put(Constant.NUM_OF_NN_LAYERS-1, errorMatrix);
		
		// Calculate the error for all units except first and last layer units
		
		for(int k = (Constant.NUM_OF_NN_LAYERS-2) ; k > 0; k--){
			
			maxUnits = unitsByLayer.get(k);
			errorMatrix = new Matrix(1, maxUnits);
			
			Matrix tempMatrix = errorMatrixByLayer.get(k+1).times(weightMap.get(k).transpose());
			Matrix currentLayerOutputMatrix = inputOutputMatrixByLayer.get(k).
					get(Constant.OUTPUT);
			
			for(int p = 0 ; p < maxUnits;p++){
				
				Double calculatedOutputValue = currentLayerOutputMatrix.get(0,p);
				errorMatrix.set(0, p, (calculatedOutputValue*(1-calculatedOutputValue)
						* tempMatrix.get(0, p)));
				
			}
			errorMatrixByLayer.put(k, errorMatrix);
		}
		
		return errorMatrixByLayer;
	}
	
	
	
	private void updateWeightAndBias(HashMap<Integer,HashMap<String,Matrix>> 
	inputOutputMatrixByLayer, HashMap<Integer,Matrix> errorMatrixByLayer){
		
		for(int l = 0;l<(Constant.NUM_OF_NN_LAYERS-1);l++){
			
			Matrix higherLayerErrorMatrix = errorMatrixByLayer.get(l+1);
			Matrix currentLayerOutputMatrix = inputOutputMatrixByLayer.get(l).get(Constant.OUTPUT);
			
			Matrix weight = weightMap.get(l);
			
			int r = weight.getRowDimension();
			int c = weight.getColumnDimension();
			
			double newWeight[][] = new double[r][c];
			
			Double updatedWeight = 0d;
			for(int m=0;m<r;m++){
				for(int n=0;n<c;n++){
					
					updatedWeight = weight.get(m, n) + ( higherLayerErrorMatrix.get(0, n)
							* currentLayerOutputMatrix.get(0, m) * Constant.LEARNING_RATE);
					
					newWeight[m][n] = updatedWeight;
				}
			}
			
			weightMap.put(l, new Matrix(newWeight));
		}
		
		for(int l = 1 ; l < Constant.NUM_OF_NN_LAYERS;l++){
			
			Matrix biasMatrix = biasMap.get(l);
			Matrix error = errorMatrixByLayer.get(l);
			
			int r = biasMatrix.getRowDimension();
			int c = biasMatrix.getColumnDimension();
			
			double newBias[][] = new double[r][c];
			
			Double updatedBias = 0d;
			
			for(int m=0;m<r;m++){
				for(int n=0;n<c;n++){
					
					updatedBias = biasMatrix.get(m, n) + 
							(Constant.LEARNING_RATE * error.get(m, n));
					newBias[m][n]= updatedBias;
				}
			}
		}
	}
	
	
	private Double calculatePredictionError(Matrix finalLayerOutput){
		
		int totalUnits = finalLayerOutput.getColumnDimension();
		
		double totalError = 0d;
		
		for(int i = 0 ; i < totalUnits ; i++){
			
			totalError += Math.pow((finalLayerOutput.get(0, i) - targetMatrix.get(0,i)),2)/2;
			
		}
		
		return totalError;
	}
	
}
