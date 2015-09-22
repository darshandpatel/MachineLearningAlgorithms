package ml;

public class Run {
	
	public static void main(String args[]){
		
		BasicRegressionTree basicRegressionTree = new BasicRegressionTree();
		basicRegressionTree.formRegressionTree();
		//System.out.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
		basicRegressionTree.printRegressionTree();
		basicRegressionTree.evaluateTestDataSet();
		
	}

}
