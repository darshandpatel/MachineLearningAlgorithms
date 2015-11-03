package ml;

import java.util.ArrayList;


public class Node {

	ArrayList<Integer> dataPoints;
	Boolean isLeaf;
	Double thresholdValue;
	Double labelValue;
	Double MSE;
	Node leftChildNode;  // A node which satisfies the threshold value
	Node rightChildNode; // A node which does not satisfy the threshold value
	Node parentNode;
	Double error;
	Integer featureIndex;
	Double entropy;
	
	public ArrayList<Integer> getDataPoints() {
		return dataPoints;
	}
	public void setDataPoints(ArrayList<Integer> dataPoints) {
		this.dataPoints = dataPoints;
	}
	public Boolean getIsLeaf() {
		return isLeaf;
	}
	public void setIsLeaf(Boolean isLeaf) {
		this.isLeaf = isLeaf;
	}
	public Double getThresholdValue() {
		return thresholdValue;
	}
	public void setThresholdValue(Double thresholdValue) {
		this.thresholdValue = thresholdValue;
	}
	public Double getLabelValue() {
		return labelValue;
	}
	public void setLabelValue(Double labelValue) {
		this.labelValue = labelValue;
	}
	public Double getMSE() {
		return MSE;
	}
	public void setMSE(Double mSE) {
		MSE = mSE;
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
	
	public void setDataPoints(Integer totalLines){
		
		dataPoints = new ArrayList<Integer>(totalLines);
		for(int i=0;i<totalLines;i++){
			dataPoints.add(i);
		}
	}
	public Double getError() {
		return error;
	}
	public void setError(Double error) {
		this.error = error;
	}
	public Integer getFeatureIndex() {
		return featureIndex;
	}
	public void setFeatureIndex(Integer featureIndex) {
		this.featureIndex = featureIndex;
	}
	public Double getEntropy() {
		return entropy;
	}
	public void setEntropy(Double entropy) {
		this.entropy = entropy;
	}
	
}
