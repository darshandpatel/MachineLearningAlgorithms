package ml;

public class Run {
	
	public static void main(String args[]){
		
		// Basic Regression Tree
		/*
		BasicRegressionTree basicRegressionTree = new BasicRegressionTree();
		basicRegressionTree.formRegressionTree();
		//System.out.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
		basicRegressionTree.printRegressionTree();
		basicRegressionTree.evaluateTestDataSet();
		*/
		
		// Decision Tree
		DecisionTree decisionTree = new DecisionTree();
		decisionTree.formMultipleDecisionTrees();
		
		
	}

}
