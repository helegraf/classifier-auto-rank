package evluationTest;

import static org.junit.Assert.assertEquals;

import java.util.Arrays;
import java.util.List;

import org.junit.Test;

import ranker.core.evaluation.RootMeanSquareError;

public class RootMeanSquareErrorTest {
	
	@Test
	public void testRMSE0() {
		List<Double> estimates = Arrays.asList(1.0,2.0,3.0,4.0,5.0);
		RootMeanSquareError rmse = new RootMeanSquareError();
		assertEquals(0, rmse.computeRMSE(estimates, estimates),0.0001);
	}
	
	@Test
	public void testRMSE10() {
		List<Double> estimates = Arrays.asList(100.0,100.0,100.0,100.0,100.0);
		List<Double> actual = Arrays.asList(0.0,0.0,0.0,0.0,0.0);
		RootMeanSquareError rmse = new RootMeanSquareError();
		assertEquals(100, rmse.computeRMSE(estimates, actual),0.0001);
	}
	
	@Test
	public void testRMSEWorksWithNans() {
		List<Double> estimates = Arrays.asList(100.0,100.0,Double.NaN,100.0,100.0);
		List<Double> actual = Arrays.asList(0.0,0.0,0.0,0.0,0.0);
		RootMeanSquareError rmse = new RootMeanSquareError();
		assertEquals(100, rmse.computeRMSE(estimates, actual),0.0001);
	}

}
