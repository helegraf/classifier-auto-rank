package ranker.core.evaluation;

import java.util.List;

import org.apache.commons.math3.stat.correlation.KendallsCorrelation;

import weka.classifiers.Classifier;

public class KendallRankCorrelationWithTies extends RankerEvaluationMeasure {

	@Override
	public double evaluate(List<Classifier> predictedRanking, List<Classifier> perfectRanking, List<Double> estimates,
			List<Double> performanceMeasures) {
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
		
		// Put those with the same actual performance value on the same rank because their order is arbitrary?
		// But only for the reference-ranking because for the other one we dont actually know 
		for (int i = 0; i < performanceMeasures.size() - 1; i++) {
			if(performanceMeasures.get(i).equals(performanceMeasures.get(i+1)) || (performanceMeasures.get(i).isNaN() && performanceMeasures.get(i+1).isNaN())) {
				// if they have the same performance value they should have the same rank
				yArray[i+1]=yArray[i];
			}
		}

		KendallsCorrelation correlation = new KendallsCorrelation();
		double result = correlation.correlation(xArray, yArray);
		return result;
	}
}
