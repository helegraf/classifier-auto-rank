package ranker.util.wekaUtil;

import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 * Estimation procedure interface. Represents a way of measuring the performance
 * of a classifier on a given data set, e.g. by using 10-fold cross-validation. 
 * 
 * @author Helena Graf
 *
 */
@FunctionalInterface
public interface EstimationProcedure {
	/**
	 * Estimates the performance of the classifier on the data set according to the
	 * evaluation measure given. Examples of estimation procedures are 33% hold-out
	 * set or 10-fold cross-validation.
	 * 
	 * @param classifier
	 *            The classifier to be used for estimation.
	 * @param evaluationMeasure
	 *            The measure according to which the performance of the classifier
	 *            is evaluated.
	 * @param dataSet
	 *            The data set on which the performance of the classifier is
	 *            estimated.
	 * @return The performance of the classifier on the datas et.
	 * @throws Exception if the estimate cannot be made
	 */
	public double estimate(Classifier classifier, EvaluationMeasure evaluationMeasure, Instances dataSet)
			throws Exception;
}
