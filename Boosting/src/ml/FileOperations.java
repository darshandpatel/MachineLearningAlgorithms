package ml;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
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
	 * @param filePath	: Path of a source file
	 * @param fileName	: Source file
	 * @param numOfDP	: Number of data points inside the file
	 * @param numOfAttribute	: Number of attribute inside the file
	 * @param splitOperator		: The regular expression for the line split
	 * @return a matrix which contains the attribute and target values
	 */
	public Matrix fetchDataPointsFromFile(String filePath,String fileName,Integer numOfDP,
			Integer numOfAttribute,String splitOperator){
		
		double dataPoints[][] = new double[numOfDP][numOfAttribute];
		try{
			
			Path trainFilepath = Paths.get(filePath,fileName);
			Integer lineCounter = 0;
			try(Stream<String> lines = Files.lines(trainFilepath)){
				Iterator<String> lineIterator = lines.iterator();
				while(lineIterator.hasNext()){
					String line = lineIterator.next();
					if(!line.equals("")){
						String parts[] = line.trim().split(splitOperator);
						for(int i=0;i< parts.length;i++){
							dataPoints[lineCounter][i] = Double.parseDouble(parts[i]);
						}
						if(dataPoints[lineCounter][parts.length-1] == 0)
							dataPoints[lineCounter][parts.length-1] = -1d;
						lineCounter++;
					}
				}
			}
			
		}catch(IOException e){
			System.out.println(e.getMessage());
		}
		return new Matrix(dataPoints);
	}
	
	/**
	 * 
	 * @param filePath	: Path of a source file
	 * @param fileName	: Source file
	 * @param numOfDP	: Number of data points inside the file
	 * @param numOfAttribute	: Number of attribute inside the file
	 * @param splitOperator		: The regular expression for the line split
	 * @return a matrix which contains the attribute and target values
	 */
	public HashMap<String,Matrix> fetchAttributesTargetFromFile(String filePath,String fileName,
			Integer numOfDP,Integer numOfAttribute,String splitOperator){
		
		double attributeValues[][] = new double[numOfDP][numOfAttribute];
		double targetValues[][] = new double[numOfDP][1];
		
		try{
			
			Path trainFilepath = Paths.get(filePath,fileName);
			Integer lineCounter = 0;
			try(Stream<String> lines = Files.lines(trainFilepath)){
				Iterator<String> lineIterator = lines.iterator();
				while(lineIterator.hasNext()){
					String line = lineIterator.next();
					if(!line.equals("")){
						String parts[] = line.trim().split(splitOperator);
						int targetValueIndex = parts.length - 1;
						for(int i=0;i<targetValueIndex;i++){
							attributeValues[lineCounter][i] = Double.parseDouble(parts[i]);
						}
						targetValues[lineCounter][0] = Double.parseDouble(parts[targetValueIndex]);
						lineCounter++;
					}
				}
			}
			
		}catch(IOException e){
			System.out.println(e.getMessage());
		}
		HashMap<String,Matrix> matrixHashMap = new HashMap<String,Matrix>();
		matrixHashMap.put(Constant.ATTRIBUTES, new Matrix(attributeValues));
		matrixHashMap.put(Constant.TARGET, new Matrix(targetValues));
		return matrixHashMap;
	}

	
}
