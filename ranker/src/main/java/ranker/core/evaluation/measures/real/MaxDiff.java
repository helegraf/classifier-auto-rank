package ranker.core.evaluation.measures.real;

import java.util.List;

import ranker.core.evaluation.measures.RankerEvaluationMeasure;

public class MaxDiff extends RankerEvaluationMeasure {

	@Override
	public double evaluate(List<String> predictedRanking, List<String> perfectRanking, List<Double> estimates,
			List<Double> performanceMeasures) {
		double result = performanceMeasures.get(0)-performanceMeasures.get(performanceMeasures.size()-1);
		return result;
	}

}
