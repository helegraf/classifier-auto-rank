package evaluation_test;

import static org.junit.Assert.assertEquals;

import java.util.Arrays;
import java.util.List;

import org.junit.Test;

import ranker.core.evaluation.measures.rank.DCGAtK;
import ranker.core.evaluation.measures.rank.DCGAtK.ExponentMode;

public class DCGAtKTest {
	
	@Test
	public void testComputeDCGAtKEqualNoCutoff() {
		List<String> perfectRanking = Arrays.asList("ZeroR", "OneR", "RandomForest");
		List<Double> performanceValues = Arrays.asList(100.0, 66.0, 33.0);
		
		DCGAtK dcgAtK = new DCGAtK(3);
		dcgAtK.setExponentMode(ExponentMode.INTEGER);
		double result = dcgAtK.evaluate(perfectRanking, perfectRanking, performanceValues, performanceValues);
		System.out.println("Equal, No Cutoff: " + result);
		assertEquals(9.392789, result, 0.0001);
	}
	
	@Test
	public void testComputeDCGAtKEqualCutoff() {
		List<String> perfectRanking = Arrays.asList("ZeroR", "OneR", "RandomForest");
		List<Double> performanceValues = Arrays.asList(100.0, 66.0, 33.0);
		
		DCGAtK dcgAtK = new DCGAtK(1);
		dcgAtK.setExponentMode(ExponentMode.INTEGER);
		double result = dcgAtK.evaluate(perfectRanking, perfectRanking, performanceValues, performanceValues);
		System.out.println("Equal, Cutoff: " + result);
		assertEquals(7, result, 0.0001);
	}
	
	@Test
	public void testComputeDCGAtKUnequalNoCutoff() {
		List<String> perfectRanking = Arrays.asList("ZeroR", "OneR", "RandomForest");
		List<String> predictedRanking = Arrays.asList("RandomForest", "OneR", "ZeroR");
		List<Double> performanceValues = Arrays.asList(100.0, 66.0, 33.0);
		
		DCGAtK dcgAtK = new DCGAtK(3);
		dcgAtK.setExponentMode(ExponentMode.INTEGER);
		double result = dcgAtK.evaluate(predictedRanking, perfectRanking, performanceValues, performanceValues);
		System.out.println("Unequal, No Cutoff: " + result);
		assertEquals(6.392789, result, 0.0001);
	}
	
	@Test
	public void testComputeDCGAtKUnequalCutoff() {
		List<String> perfectRanking = Arrays.asList("ZeroR", "OneR", "RandomForest");
		List<String> predictedRanking = Arrays.asList("RandomForest", "OneR", "ZeroR");
		List<Double> performanceValues = Arrays.asList(100.0, 66.0, 33.0);
		
		DCGAtK dcgAtK = new DCGAtK(1);
		dcgAtK.setExponentMode(ExponentMode.INTEGER);
		double result = dcgAtK.evaluate(predictedRanking, perfectRanking, performanceValues, performanceValues);
		System.out.println("Unequal, Cutoff: " + result);
		assertEquals(1, result, 0.0001);
	}

}
