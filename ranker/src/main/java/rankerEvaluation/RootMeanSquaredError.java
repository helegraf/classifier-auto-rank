package rankerEvaluation;

import java.util.List;

import ranker.algorithms.PerfectRanker;
import ranker.algorithms.Ranker;
import ranker.algorithms.RegressionRanker;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Calculated the difference between the predicted performance value of the
 * learning algorithm and its actual value and thus fives information about how
 * well the Ranker has learned this model. Rankers evaluated with this class
 * need to be sub types of {@link ranker.algorithms.RegressionRanker}.
 * 
 * @author Helena Graf
 *
 */
public class RootMeanSquaredError implements RankerEvaluationMeasure {

	@Override
	public double evaluate(Ranker regressionRanker, Instances train, Instances test, List<Integer> targetAttributes) {
		// TODO maybe something better than causing an exception here?
		RegressionRanker ranker = (RegressionRanker) regressionRanker;

		double result = 0;
		int numInstancesCalculated = test.numInstances();

		try {
			PerfectRanker oracle = new PerfectRanker();
			oracle.buildRanker(train, targetAttributes);
			ranker.buildRanker(train, targetAttributes);

			for (Instance instance : test) {
				try {
					List<Classifier> perfectRanking = oracle.predictRankingforInstance(instance);
					List<Classifier> predictedRanking = ranker.predictRankingforInstance(instance);
					
					List<Double> performanceValues = oracle.getPerformanceMeasuresForRanking();
					List<Double> predictedPerformances = ranker.getEstimatesForRanking();

					// Find corresponding classifier values and compare
					for (int i = 0; i < predictedRanking.size(); i++) {
						for (int j = 0; j < perfectRanking.size(); j++) {
							if (predictedRanking.get(i).getClass().getName()
									.equals(perfectRanking.get(j).getClass().getName())) {
								// Have found position, now compare predicted value with actual Value
								double difference = predictedPerformances.get(i) - performanceValues.get(j);
								result += Math.pow(difference, 2);
								break;
							}
						}
					}

				} catch (Exception e) {
					// TODO log
					numInstancesCalculated--;
				}
			}

		} catch (Exception e) {
			// TODO log
		}

		// TODO maybe better solution than this (+ also for Kendall)
		if (numInstancesCalculated != 0) {
			result /= numInstancesCalculated;
			result = Math.sqrt(result);
		}

		return result;
	}

}
