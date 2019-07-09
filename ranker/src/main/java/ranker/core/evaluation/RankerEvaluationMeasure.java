package ranker.core.evaluation;

import java.util.List;

import weka.classifiers.Classifier;

public abstract class RankerEvaluationMeasure {
	
	protected StringBuilder summaryString = new StringBuilder();
	
	/**
	 * Trains the Ranker on the training data set and evaluates its performance
	 * on the test data set according to the implemented measure, e.g. predictive
	 * accuracy or the area under the ROC curve.
	 * 
	 * @param predictedRanking
	 * @param perfectRanking
	 * @param estimates
	 * @param performanceMeasures
	 * @return
	 */
	public abstract double evaluate(List<Classifier> predictedRanking, List<Classifier> perfectRanking, List<Double> estimates, List<Double> performanceMeasures);
	
	public String getSummary() {
		return summaryString.toString();
	}

	public String getName() {
		return this.getClass().getSimpleName();
	}
}
