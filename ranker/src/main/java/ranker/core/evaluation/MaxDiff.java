package ranker.core.evaluation;

import java.util.List;

import weka.classifiers.Classifier;

public class MaxDiff extends RankerEvaluationMeasure {

	@Override
	public double evaluate(List<Classifier> predictedRanking, List<Classifier> perfectRanking, List<Double> estimates,
			List<Double> performanceMeasures) {
		double result = performanceMeasures.get(0)-performanceMeasures.get(performanceMeasures.size()-1);
		return result;
	}

}
