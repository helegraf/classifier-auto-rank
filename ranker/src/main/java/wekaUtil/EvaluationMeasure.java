package ranker;

import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 * Evaluation measure interface. Represents the performance according to which a
 * classifier is evaluated on a data set, e.g. predictive accuracy or area under
 * the ROC curve.
 * 
 * @author Helena Graf
 *
 */
@FunctionalInterface
public interface EvaluationMeasure {
	/**
	 * Trains the classifier on the training data set and evaluates its performance
	 * on the test data set according to the implemented measure, e.g. predictive
	 * accuracy or the area under the ROC curve.
	 * 
	 * @param classifier
	 *            The Classifier to be used for evaluation.
	 * @param train
	 *            The Data set for training the classifier.
	 * @param test
	 *            The Data set for testing the classifier.
	 * @return The performance of the classifier on the data set.
	 * @throws Exception
	 */
	public double evaluate(Classifier classifier, Instances train, Instances test) throws Exception;
}
