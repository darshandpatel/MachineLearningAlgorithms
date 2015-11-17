package ml;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream; 

/**
 * @author Darshan
 *
 */
public class Test {
	
	public static void main(String args[]){
		
		List<Integer> numbers = IntStream.iterate(1, n -> n + 10).limit(100)
				.boxed().collect(Collectors.toList());
	    System.out.println(numbers);
	    
	    
	    System.out.println(Math.log(1d));
		
	}

}
