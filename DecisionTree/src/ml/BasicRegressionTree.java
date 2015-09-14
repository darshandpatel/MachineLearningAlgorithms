package ml;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;

public class BasicRegressionTree {
	
	ArrayList<Feature> features;
	LinkedList<Feature> selectedFeature = new LinkedList<Feature>();
	
	public Node featureSelection(){
		
		for(Feature feature : features){
			
			if(feature.getType().equals(Constant.NUMERIC)){
				
				for(Float value : (ArrayList<Float>) feature.getValues()){
					
					// Pass the feature name, index and its value to FileOperation method
				}
				
			}
				
				
			
		}
		
		
		return null;
	}
	
	

}
