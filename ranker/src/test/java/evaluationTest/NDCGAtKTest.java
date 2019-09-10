package evaluationTest;

import static org.junit.Assert.assertEquals;

import java.util.Arrays;
import java.util.List;

import org.junit.Test;

import ranker.core.evaluation.measures.rank.NDCGAtK;

public class NDCGAtKTest {

	@Test
	public void testComputeNDCGAtKEqualNoCutoff() {
		List<String> perfectRanking = Arrays.asList("ZeroR", "OneR", "RandomForest");
		List<Double> performanceValues = Arrays.asList(100.0, 66.0, 33.0);

		NDCGAtK dcgAtK = new NDCGAtK(3);
		double result = dcgAtK.evaluate(perfectRanking, perfectRanking, performanceValues, performanceValues);
		System.out.println("Equal, No Cutoff: " + result);
		assertEquals(1, result, 0.0001);
	}

	@Test
	public void testComputeNDCGAtKEqualCutoff() {
		List<String> perfectRanking = Arrays.asList("ZeroR", "OneR", "RandomForest");
		List<Double> performanceValues = Arrays.asList(100.0, 66.0, 33.0);

		NDCGAtK dcgAtK = new NDCGAtK(1);
		double result = dcgAtK.evaluate(perfectRanking, perfectRanking, performanceValues, performanceValues);
		System.out.println("Equal, Cutoff: " + result);
		assertEquals(1, result, 0.0001);
	}

	@Test
	public void testComputeNDCGAtKUnequalNoCutoff() {
		List<String> perfectRanking = Arrays.asList("ZeroR", "OneR", "RandomForest");
		List<String> predictedRanking = Arrays.asList("RandomForest", "OneR", "ZeroR");
		List<Double> performanceValues = Arrays.asList(100.0, 66.0, 33.0);

		NDCGAtK dcgAtK = new NDCGAtK(3);
		double result = dcgAtK.evaluate(predictedRanking, perfectRanking, performanceValues, performanceValues);
		System.out.println("Unequal, No Cutoff: " + result);
		assertEquals(0.8570257072748435, result, 0.0001);
	}

	@Test
	public void testComputeNDCGAtKUnequalCutoff() {
		List<String> perfectRanking = Arrays.asList("ZeroR", "OneR", "RandomForest");
		List<String> predictedRanking = Arrays.asList("RandomForest", "OneR", "ZeroR");
		List<Double> performanceValues = Arrays.asList(100.0, 66.0, 33.0);

		NDCGAtK dcgAtK = new NDCGAtK(1);
		double result = dcgAtK.evaluate(predictedRanking, perfectRanking, performanceValues, performanceValues);
		System.out.println("Unequal, Cutoff: " + result);
		assertEquals(0.2599210498948732, result, 0.0001);
	}

	@Test
	public void testDescendingValues() {
		List<String> perfectRanking = Arrays.asList("ZeroR", "OneR", "RandomForest");
		List<String> predictedRanking = Arrays.asList("RandomForest", "OneR", "ZeroR");
		List<Double> performanceValues = Arrays.asList(100.0, 66.0, 33.0);

		NDCGAtK dcgAtK = new NDCGAtK(1);
		double result = dcgAtK.evaluate(predictedRanking, perfectRanking, performanceValues, performanceValues);
		System.out.println("Unequal, Cutoff: " + result);
	}
}
