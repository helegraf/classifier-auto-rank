package ranker.core.evaluation;

import java.util.ArrayList;
import java.util.List;

import weka.classifiers.Classifier;

/**
 * Calculated the difference between the predicted performance value of the
 * learning algorithm and its actual value and thus fives information about how
 * well the Ranker has learned this model. Rankers evaluated with this class
 * need to be sub types of {@link ranker.core.algorithms.regression.RegressionRanker}.
 * 
 * @author Helena Graf
 *
 */
public class RootMeanSquareError extends RankerEvaluationMeasure {

	@Override
	public double evaluate(List<Classifier> predictedRanking, List<Classifier> perfectRanking, List<Double> estimates,
			List<Double> performanceMeasures) {
		System.out.print("RMSE ");
		List<Double> estimations = new ArrayList<Double>();
		List<Double> actualValues = new ArrayList<Double>();

		// Find corresponding classifier values and compare
		for (int i = 0; i < predictedRanking.size(); i++) {
			for (int j = 0; j < perfectRanking.size(); j++) {
				if (predictedRanking.get(i).getClass().getName()
						.equals(perfectRanking.get(j).getClass().getName())) {
					// Have found position, now remember value
					estimations.add(estimates.get(i));
					actualValues.add(performanceMeasures.get(j));
					break;
				}
			}
		}
		

		double result = computeRMSE(estimations, actualValues);
		System.out.println(result);
		return result;
	}
	
	public double computeRMSE(List<Double> estimations, List<Double> actualValues) {
		double result = 0;
		int numCalculated = estimations.size();
		
		// Add squares of differences
		for (int i = 0; i < estimations.size(); i++) {
			double difference = estimations.get(i) - actualValues.get(i);
			double squared = Math.pow(difference, 2);
			if (!Double.isNaN(squared)) {
				result += squared;
			} else {
				numCalculated--;
			}	
		}

		// Divide by n
		if (numCalculated == 0) {
			result = Double.NaN;
		} else {
			result /= numCalculated;
		}
		
		// Root
		result = Math.sqrt(result);
		
		return result;
	}

}
