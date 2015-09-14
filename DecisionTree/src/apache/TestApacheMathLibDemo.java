package apache;

import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;


public class TestApacheMathLibDemo {

/**
* @param args
*/
public static void main(String[] args) {

	RandomGenerator randomGenerator = new JDKRandomGenerator();
	System.out.println(randomGenerator.nextInt());
	System.out.println(randomGenerator.nextDouble());
	
	/**
	* Descriptive Statistics like MEAN,GP,SD,MAX
	**/
	DescriptiveStatistics stats = new DescriptiveStatistics();
	stats.addValue(1);
	stats.addValue(2);
	stats.addValue(3);
	stats.addValue(4);
	stats.addValue(5);
	stats.addValue(6);
	stats.addValue(7);
	System.out.print("Mean : "+stats.getMean()+"n");
	System.out.print("Standard deviation : "+stats.getStandardDeviation()+"n");
	System.out.print("Max : "+stats.getMax()+"n");
	
	/**
	* Complex number format a+bi
	**/
	Complex c1 = new Complex(1, 2);
	Complex c2 = new Complex(2, 3);
	System.out.print("Absolute of c1 "+c1.abs()+"n");
	System.out.print("Addition : "+(c1.add(c2))+"n");
	}

}