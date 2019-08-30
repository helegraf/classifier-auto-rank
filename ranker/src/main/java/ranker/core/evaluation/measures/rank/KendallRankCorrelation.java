package ranker.core.evaluation.measures.rank;

import java.util.List;

import org.apache.commons.math3.stat.correlation.KendallsCorrelation;

import ranker.core.evaluation.measures.RankerEvaluationMeasure;

public class KendallRankCorrelation extends RankerEvaluationMeasure {

	@Override
	public double evaluate(List<String> predictedRanking, List<String> perfectRanking, List<Double> estimates,
			List<Double> performanceMeasures) {
		// Initialize temporary variables
		double[] xArray = new double[predictedRanking.size()];
		double[] yArray = new double[perfectRanking.size()];

		// Prepare computation of correlation
		for (int i = 0; i < yArray.length; i++) {
			yArray[i] = i;
			for (int j = 0; j < perfectRanking.size(); j++) {
				if (predictedRanking.get(i).equals(perfectRanking.get(j))) {
					xArray[j] = i;
					break;
				}
			}
		}

		KendallsCorrelation correlation = new KendallsCorrelation();
		double result = correlation.correlation(xArray, yArray);
		
		if (result < -1 || result > 1) {
			System.err.println(result);
		}
		
		return result;
	}

}
