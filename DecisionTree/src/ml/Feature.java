package ml;

import java.util.ArrayList;

public class Feature {
	
	String name;
	String type;
	ArrayList<Double> values;
	Integer index;
	
	
	public Feature(String name,String type, ArrayList<Double> values,Integer index){
		this.name = name;
		this.type = type;
		this.values = values;
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
	public ArrayList<Double> getValues() {
		return values;
	}
	public void setValues(ArrayList<Double> values) {
		this.values = values;
	}

	public Integer getIndex() {
		return index;
	}

	public void setIndex(Integer index) {
		this.index = index;
	}
	
	
}
