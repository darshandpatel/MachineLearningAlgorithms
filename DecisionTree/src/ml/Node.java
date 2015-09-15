package ml;

import java.util.ArrayList;
import java.util.HashMap;

public class Node {

	ArrayList<Integer> dataPoints;
	Feature feature;
	Boolean isLeaf;
	Double criteriaValue;
	Double labelValue;
	Double variance;
	Node leftChildNode;  // A node which satisfies the criteria value
	Node rightChildNode; // A node which does not satisfy the criteria value
	Node parentNode;
	
	public ArrayList<Integer> getDataPoints() {
		return dataPoints;
	}
	public void setDataPoints(ArrayList<Integer> dataPoints) {
		this.dataPoints = dataPoints;
	}
	public Feature getFeature() {
		return feature;
	}
	public void setFeature(Feature feature) {
		this.feature = feature;
	}
	public Boolean getIsLeaf() {
		return isLeaf;
	}
	public void setIsLeaf(Boolean isLeaf) {
		this.isLeaf = isLeaf;
	}
	public Double getCriteriaValue() {
		return criteriaValue;
	}
	public void setCriteriaValue(Double criteriaValue) {
		this.criteriaValue = criteriaValue;
	}
	public Double getLabelValue() {
		return labelValue;
	}
	public void setLabelValue(Double labelValue) {
		this.labelValue = labelValue;
	}
	public Double getVariance() {
		return variance;
	}
	public void setVariance(Double variance) {
		this.variance = variance;
	}
	public Node getLeftChildNode() {
		return leftChildNode;
	}
	public void setLeftChildNode(Node leftChildNode) {
		this.leftChildNode = leftChildNode;
	}
	public Node getRightChildNode() {
		return rightChildNode;
	}
	public void setRightChildNode(Node rightChildNode) {
		this.rightChildNode = rightChildNode;
	}
	public Node getParentNode() {
		return parentNode;
	}
	public void setParentNode(Node parentNode) {
		this.parentNode = parentNode;
	}
	
	
	
}
