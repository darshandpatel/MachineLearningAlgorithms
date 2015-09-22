package ml;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Iterator;
import java.util.stream.Stream;

import Jama.Matrix;

/**
 * This class contains all the methods related to read/write operation of the DataSet.
 * @author Darshan
 *
 */
public class FileOperations {

	
	/**
	 * 
	 * @return The matrix which rows represent the DataPoints (line in the file)
	 *  and which columns represent the feature values. 
	 */
	public Matrix fetchTrainingDataPoints(){
		
		double dataPoints[][] = new double
				[Constant.NUM_OF_TRAINING_DATAPOINTS][Constant.NUM_OF_FEATURES+1];
		try{
			
			Path trainFilepath = Paths.get(Constant.FILE_PATH,Constant.TRAINDATA_FILE);
			Integer lineCounter = 0;
			try(Stream<String> lines = Files.lines(trainFilepath)){
				Iterator<String> lineIterator = lines.iterator();
				while(lineIterator.hasNext()){
					String line = lineIterator.next();
					String parts[] = line.trim().split("\\s+");
					for(int i=0;i<parts.length;i++){
						dataPoints[lineCounter][i] = Double.parseDouble(parts[i]);
					}
					lineCounter++;
				}
			}
			
		}catch(IOException e){
			System.out.println(e.getMessage());
		}
		return new Matrix(dataPoints);
	}
	
	
	/**
	 * 
	 * @return The matrix which rows represent the DataPoints (line in the file)
	 *  and which columns represent the feature values. 
	 */
	public Matrix fetchTestingDataPoints(){
		
		double dataPoints[][] = new double
				[Constant.NUM_OF_TESTING_DATAPOINTS][Constant.NUM_OF_FEATURES+1];
		try{
			
			Path trainFilepath = Paths.get(Constant.FILE_PATH,Constant.TESTDATA_FILE);
			Integer lineCounter = 0;
			try(Stream<String> lines = Files.lines(trainFilepath)){
				Iterator<String> lineIterator = lines.iterator();
				while(lineIterator.hasNext()){
					String line = lineIterator.next();
					String parts[] = line.trim().split("\\s+");
					for(int i=0;i<parts.length;i++){
						dataPoints[lineCounter][i] = Double.parseDouble(parts[i]);
					}
					lineCounter++;
				}
			}
			
		}catch(IOException e){
			System.out.println(e.getMessage());
		}
		return new Matrix(dataPoints);
	}
	
}
