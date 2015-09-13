package ml;

import java.util.ArrayList;

public class Node {

	String featureName;
	Float criteriaValue;
	String condition;
	Float labelValue;
	ArrayList<Node> childNodes;
	Node parentNode;
	
	
	public String getFeatureName() {
		return featureName;
	}
	public void setFeatureName(String featureName) {
		this.featureName = featureName;
	}
	public Float getCriteriaValue() {
		return criteriaValue;
	}
	public void setCriteriaValue(Float criteriaValue) {
		this.criteriaValue = criteriaValue;
	}
	public String getCondition() {
		return condition;
	}
	public void setCondition(String condition) {
		this.condition = condition;
	}
	public Float getLabelValue() {
		return labelValue;
	}
	public void setLabelValue(Float labelValue) {
		this.labelValue = labelValue;
	}
	public ArrayList<Node> getChildNodes() {
		return childNodes;
	}
	public void setChildNodes(ArrayList<Node> childNodes) {
		this.childNodes = childNodes;
	}
	public Node getParentNode() {
		return parentNode;
	}
	public void setParentNode(Node parentNode) {
		this.parentNode = parentNode;
	}
	
}
