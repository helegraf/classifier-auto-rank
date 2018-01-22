package rankerEvaluation;

import java.util.List;

import ranker.algorithms.Ranker;
import weka.core.Instances;

public interface RankerEvaluationMeasure {
	/**
	 * Trains the Ranker on the training data set and evaluates its performance
	 * on the test data set according to the implemented measure, e.g. predictive
	 * accuracy or the area under the ROC curve.
	 * 
	 * @param ranker
	 *            The Ranker to be used for evaluation.
	 * @param train
	 *            The Data set for training the ranker.
	 * @param test
	 *            The Data set for testing the ranker.
	 * @param targetAttributes TODO
	 * @return The performance of the classifier on the data set.
	 * @throws Exception
	 */
	public double evaluate(Ranker ranker, Instances train, Instances test, List<Integer> targetAttributes);
}
