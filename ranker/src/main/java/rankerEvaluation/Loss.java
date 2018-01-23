package rankerEvaluation;

import java.util.List;

import ranker.algorithms.PerfectRanker;
import ranker.algorithms.Ranker;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

public class Loss implements RankerEvaluationMeasure {

	@Override
	public double evaluate(Ranker ranker, Instances train, Instances test, List<Integer> targetAttributes) {
		int result = 0;
		int numInstancesCalculated = test.numInstances();

		try {
			PerfectRanker oracle = new PerfectRanker();
			oracle.buildRanker(train, targetAttributes);
			ranker.buildRanker(train, targetAttributes);

			for (Instance instance : test) {
				try {
					List<Classifier> perfectRanking = oracle.predictRankingforInstance(instance);
					List<Classifier> predictedRanking = ranker.predictRankingforInstance(instance);
					
					// Find the place at which the ranker has put the algorithm that in reality is
					// the best
					for (int i = 0; i < predictedRanking.size(); i++) {
						if (perfectRanking.get(0).getClass().getName()
								.equals(predictedRanking.get(i).getClass().getName())) {
							// Compute difference
							List<Double> performanceMeasures = oracle.getPerformanceMeasuresForRanking();
							double loss = performanceMeasures.get(0) - performanceMeasures.get(i);
							result += loss;
							break;
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
			result = result/numInstancesCalculated;
		}
		
		return result;
	}
}
