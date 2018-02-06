package ranker.util.wekaUtil;

import java.util.List;
import java.util.Random;

import jaicore.ml.WekaUtil;
import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 * Measuring the performance of a classifier on a given data set by means of Monte Carlo Crossvalidation.
 * 
 * @author Helena Graf
 *
 */
public class StratifiedMCCV implements EstimationProcedure {
	final private int times;
	final private double holdout;

	/**
	 * Construct a StratifiedMCCV estimation procedure.
	 * 
	 * @param times How many times the procedure is to be repeated (results averaged).
	 * @param holdout The size of the holdout set.
	 */
	public StratifiedMCCV(int times, double holdout) {
		this.times = times;
		this.holdout = holdout;
	}

	@Override
	public double estimate(Classifier classifier, EvaluationMeasure evaluationMeasure, Instances dataset)
			throws Exception {
		double result = 0;
		for (int i = 0; i < times; i++) {
			List<Instances> splits = WekaUtil.getStratifiedSplit(dataset, new Random(i), 1 - holdout);
			result += evaluationMeasure.evaluate(classifier, splits.get(0), splits.get(1));
		}
		result /= times;
		return result;
	}

}
