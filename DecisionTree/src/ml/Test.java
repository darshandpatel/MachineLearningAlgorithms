package ml;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.Queue;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import Jama.Matrix;

public class Test {
	
	public static void main(String args[]){

		DescriptiveStatistics descriptiveStatistics = new DescriptiveStatistics();
		ArrayList<Integer> values = new ArrayList(Arrays.asList(31,32,30,31,30));
		for(Integer row : values){
			descriptiveStatistics.addValue(row);
		}
		
		System.out.println(descriptiveStatistics.getVariance()*5);
		
	}

}
