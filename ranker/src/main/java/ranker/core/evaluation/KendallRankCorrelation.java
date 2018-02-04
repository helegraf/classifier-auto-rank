package ranker.core.evaluation;

import java.util.List;

import org.apache.commons.math3.stat.correlation.KendallsCorrelation;

import weka.classifiers.Classifier;

public class KendallRankCorrelation extends RankerEvaluationMeasure {

	@Override
	public double evaluate(List<Classifier> predictedRanking, List<Classifier> perfectRanking, List<Double> estimates,
			List<Double> performanceMeasures) {
		System.out.print("Kendall ");
		// Initialize temporary variables
		double[] xArray = new double[predictedRanking.size()];
		double[] yArray = new double[perfectRanking.size()];

		// Prepare computation of correlation
		for (int i = 0; i < yArray.length; i++) {
			yArray[i] = i;
			for (int j = 0; j < perfectRanking.size(); j++) {
				if (predictedRanking.get(i).getClass().getName().equals(perfectRanking.get(j).getClass().getName())) {
					xArray[j] = i;
					break;
				}
			}
		}

		KendallsCorrelation correlation = new KendallsCorrelation();
		double result = correlation.correlation(xArray, yArray);
		System.out.println(result);
		return result;
	}

}
