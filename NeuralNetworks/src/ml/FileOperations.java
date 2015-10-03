package ml;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map.Entry;
import java.util.stream.Stream;

import Jama.Matrix;

public class FileOperations {
	
	
	public HashMap<String,Object> fetchDataPointsFromFile(String filePath,String fileName,
			Integer numOfDP,Integer numOfAttribute,String splitOperator){
		
		double attributeValues[][] = new double[numOfDP][numOfAttribute+1];
		double targetValues[][] = new double[numOfDP][1];
		
		HashMap<Integer,HashMap<String,Double>> featureMinMaxMap = 
				new HashMap<Integer,HashMap<String,Double>>();
		
		for(int i = 1 ; i <= Constant.PERCEPTRON_NO_OF_FEATURES ; i++){
			
			HashMap<String,Double> minMaxMap = new HashMap<String,Double>();
			minMaxMap.put(Constant.MIN, Double.POSITIVE_INFINITY);
			minMaxMap.put(Constant.MAX, Double.NEGATIVE_INFINITY);
			featureMinMaxMap.put(Integer.valueOf(i),minMaxMap);
			
		}
		//printFeatureMinMaxMap(featureMinMaxMap);
		
		try{
			
			Path trainFilepath = Paths.get(filePath,fileName);
			Integer lineCounter = 0;
			
			try(Stream<String> lines = Files.lines(trainFilepath)){
				
				Iterator<String> lineIterator = lines.iterator();
				
				while(lineIterator.hasNext()){
					
					String line = lineIterator.next();
					
					if(!line.equals("")){
						
						String parts[] = line.trim().split(splitOperator);
						
						// First column value of any row will be dummy value (1)
						attributeValues[lineCounter][0] = 1;
						int targetValueIndex = parts.length -1;
						
						for(int i=0;i<targetValueIndex;i++){
							
							Double attributeValue = Double.parseDouble(parts[i]);
							attributeValues[lineCounter][i+1] = attributeValue;
							
							// Update Min and Max value
							HashMap<String,Double> minMaxMap = featureMinMaxMap.get(i+1);
							
							if(attributeValue < minMaxMap.get(Constant.MIN))
								minMaxMap.put(Constant.MIN,attributeValue);
							if(attributeValue > minMaxMap.get(Constant.MAX))
								minMaxMap.put(Constant.MAX,attributeValue);
						}
						
						targetValues[lineCounter][0] = Double.parseDouble(parts[targetValueIndex]);
						lineCounter++;
					}
				}
			}
			
		}catch(IOException e){
			System.out.println(e.getMessage());
		}
		
		//printFeatureMinMaxMap(featureMinMaxMap);
		
		HashMap<String,Object> matrixHashMap = new HashMap<String,Object>();
		matrixHashMap.put(Constant.ATTRIBUTES, new Matrix(attributeValues));
		matrixHashMap.put(Constant.TARGET, new Matrix(targetValues));
		matrixHashMap.put(Constant.MIN_MAX_MAP,featureMinMaxMap);
		
		return matrixHashMap;
	}
	
	private void printFeatureMinMaxMap(HashMap<Integer,HashMap<String,Double>> featureMinMaxMap){
		
		java.util.Iterator<Entry<Integer, HashMap<String, Double>>> iterator = 
				featureMinMaxMap.entrySet().iterator();
		
		while(iterator.hasNext()){
			
			Entry<Integer, HashMap<String, Double>> entry = iterator.next();
			Integer featureIndex = entry.getKey();
			HashMap<String, Double> minMaxMap = entry.getValue();
			
			System.out.println("Feature index : "+ featureIndex);
			System.out.println("Min value : " + minMaxMap.get(Constant.MIN));
			System.out.println("Max value : " + minMaxMap.get(Constant.MAX));
			
		}
		
	}

}
