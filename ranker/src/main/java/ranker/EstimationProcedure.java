package ranker;

import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 * Estimation procedure interface. Represents a way of measuring the performance
 * of a classifier on a given data set, e.g. by using 10-fold-Crossvaliation. 
 * 
 * @author Helena Graf
 *
 */
@FunctionalInterface
public interface EstimationProcedure {
	/**
	 * Estimates the performance of the classifier on the data set according to the
	 * evaluation measure given. Examples of estimation procedures are 33% Holdout
	 * Set or 10-fold-Crossvalidation.
	 * 
	 * @param classifier
	 *            The classifier to be used for estimation.
	 * @param evaluationMeasure
	 *            The measure according to which the performance of the classifier
	 *            is evaluated.
	 * @param dataset
	 *            The data set on which the performance of the classifier is
	 *            estimated.
	 * @return The performance of the classifier on the dataset.
	 * @throws Exception
	 */
	public double estimate(Classifier classifier, EvaluationMeasure evaluationMeasure, Instances dataset)
			throws Exception;
}
