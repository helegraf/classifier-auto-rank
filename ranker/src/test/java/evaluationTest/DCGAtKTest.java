package evaluationTest;

import static org.junit.Assert.assertEquals;

import java.util.Arrays;
import java.util.List;

import org.junit.Test;

import ranker.core.evaluation.measures.rank.DCGAtK;

public class DCGAtKTest {
	
	@Test
	public void testComputeDCGAtKEqualNoCutoff() {
		List<String> perfectRanking = Arrays.asList("ZeroR", "OneR", "RandomForest");
		List<Double> performanceValues = Arrays.asList(100.0, 66.0, 33.0);
		
		DCGAtK dcgAtK = new DCGAtK(3);
		double result = dcgAtK.evaluate(perfectRanking, perfectRanking, performanceValues, performanceValues);
		System.out.println("Equal, No Cutoff: " + result);
		assertEquals(1.4954575474976748, result, 0.0001);
	}
	
	@Test
	public void testComputeDCGAtKEqualCutoff() {
		List<String> perfectRanking = Arrays.asList("ZeroR", "OneR", "RandomForest");
		List<Double> performanceValues = Arrays.asList(100.0, 66.0, 33.0);
		
		DCGAtK dcgAtK = new DCGAtK(1);
		double result = dcgAtK.evaluate(perfectRanking, perfectRanking, performanceValues, performanceValues);
		System.out.println("Equal, Cutoff: " + result);
		assertEquals(0.9102392266268373, result, 0.0001);
	}
	
	@Test
	public void testComputeDCGAtKUnequalNoCutoff() {
		List<String> perfectRanking = Arrays.asList("ZeroR", "OneR", "RandomForest");
		List<String> predictedRanking = Arrays.asList("RandomForest", "OneR", "ZeroR");
		List<Double> performanceValues = Arrays.asList(100.0, 66.0, 33.0);
		
		DCGAtK dcgAtK = new DCGAtK(3);
		double result = dcgAtK.evaluate(predictedRanking, perfectRanking, performanceValues, performanceValues);
		System.out.println("Unequal, No Cutoff: " + result);
		assertEquals(1.2816455623436975, result, 0.0001);
	}
	
	@Test
	public void testComputeDCGAtKUnequalCutoff() {
		List<String> perfectRanking = Arrays.asList("ZeroR", "OneR", "RandomForest");
		List<String> predictedRanking = Arrays.asList("RandomForest", "OneR", "ZeroR");
		List<Double> performanceValues = Arrays.asList(100.0, 66.0, 33.0);
		
		DCGAtK dcgAtK = new DCGAtK(1);
		double result = dcgAtK.evaluate(predictedRanking, perfectRanking, performanceValues, performanceValues);
		System.out.println("Unequal, Cutoff: " + result);
		assertEquals(0.23659033544034497, result, 0.0001);
	}

}
