package ranker.core.evaluation.measures;

import java.util.List;

/**
 * A measures used to evaluate the quality of the estimated rankings produced by a ranking algorithm.
 * 
 * @author helegraf
 *
 */
public abstract class RankerEvaluationMeasure {

	protected StringBuilder summaryString = new StringBuilder();

	/**
	 * Trains the Ranker on the training data set and evaluates its performance on
	 * the test data set according to the implemented measure, e.g. predictive
	 * accuracy or the area under the ROC curve.
	 * 
	 * @param predictedRanking    the ranking that has been predicted by an
	 *                            algorithm
	 * @param perfectRanking      the correct ranking for the instance
	 * @param estimates           the performance estimated for each algorithm in
	 *                            the ranking, may be null
	 * @param performanceMeasures the actual performance value for each algorithm
	 * @return a value indicating the quality of the predicted ranking
	 */
	public abstract double evaluate(List<String> predictedRanking, List<String> perfectRanking, List<Double> estimates,
			List<Double> performanceMeasures);

	public String getSummary() {
		return summaryString.toString();
	}

	public String getName() {
		return this.getClass().getSimpleName();
	}
}
