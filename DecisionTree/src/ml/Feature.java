package ml;

import java.util.ArrayList;

public class Feature {
	
	String name;
	String type;
	Object values;
	Integer index;
	
	
	public Feature(String name,String type, ArrayList<Float> values,Integer index){
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
	public Object getValues() {
		return values;
	}
	public void setValues(Object values) {
		this.values = values;
	}

	public Integer getIndex() {
		return index;
	}

	public void setIndex(Integer index) {
		this.index = index;
	}
	
	
}
