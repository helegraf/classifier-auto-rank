package rankerEvaluation;

import java.util.List;

import weka.classifiers.Classifier;

/**
 * Calculated the difference between the predicted performance value of the
 * learning algorithm and its actual value and thus fives information about how
 * well the Ranker has learned this model. Rankers evaluated with this class
 * need to be sub types of {@link ranker.algorithms.RegressionRanker}.
 * 
 * @author Helena Graf
 *
 */
public class RootMeanSquaredError extends RankerEvaluationMeasure {

	@Override
	public double evaluate(List<Classifier> predictedRanking, List<Classifier> perfectRanking, List<Double> estimates,
			List<Double> performanceMeasures) {
		System.out.print("RMSE ");
		// Initialize result, catch NaNs
		double result = 0;
		int numCalculated = predictedRanking.size();
		
		// Find corresponding classifier values and compare
		for (int i = 0; i < predictedRanking.size(); i++) {
			for (int j = 0; j < perfectRanking.size(); j++) {
				if (predictedRanking.get(i).getClass().getName()
						.equals(perfectRanking.get(j).getClass().getName())) {
					// Have found position, now compare predicted value with actual Value
					double difference = estimates.get(i) - performanceMeasures.get(j);
					double squared =  Math.pow(difference, 2);

					if (!Double.isNaN(squared)) {
						result += squared;
					} else {
						numCalculated--;
					}					
					break;
				}
			}
		}
		
		if (numCalculated == 0) {
			result = Double.NaN;
		} else {
			result /= numCalculated;
		}
		
		result =  Math.sqrt(result);
		System.out.println(result);
		return result;
	}

}
