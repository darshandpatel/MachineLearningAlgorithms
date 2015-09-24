package ml;

import java.util.ArrayList;

public class Feature {
	
	String name;
	String type;
	ArrayList<Double> thresholdValues;
	Integer index;
	
	
	public Feature(String name,String type, ArrayList<Double> thresholdValues,Integer index){
		this.name = name;
		this.type = type;
		this.thresholdValues = thresholdValues;
		this.index = index;
	}
	
	public Feature(String type, ArrayList<Double> thresholdValues,Integer index){
		this.type = type;
		this.thresholdValues = thresholdValues;
		this.index = index;
	}
	
	public String getName() {
		return name;
	}
	public void setName(String name) {
		this.name = name;
	}
	public String getType() {
		return type;
	}
	public void setType(String type) {
		this.type = type;
	}
	public ArrayList<Double> getThresholdValues() {
		return thresholdValues;
	}
	public void setThresholdValues(ArrayList<Double> thresholdValues) {
		this.thresholdValues = thresholdValues;
	}

	public Integer getIndex() {
		return index;
	}

	public void setIndex(Integer index) {
		this.index = index;
	}
	
	
}
