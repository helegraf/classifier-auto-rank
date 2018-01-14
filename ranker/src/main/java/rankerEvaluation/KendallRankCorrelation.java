package rankerEvaluation;

import java.util.List;

import org.apache.commons.math3.stat.correlation.KendallsCorrelation;

import ranker.PerfectRanker;
import ranker.Ranker;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

public class KendallRankCorrelation implements RankerEvaluationMeasure {

	@Override
	public double evaluate(Ranker ranker, Instances train, Instances test) {
		double correlation = 0;
		double numInstancesCalculated = test.numInstances();
		Ranker perfectRanker = new PerfectRanker();
		try {
			perfectRanker.buildRanker(train);
			ranker.buildRanker(train);

			for (Instance instance : test) {
				List<Classifier> perfectRanking;
				try {
					perfectRanking = perfectRanker.predictRankingforInstance(instance);
					List<Classifier> ranking = ranker.predictRankingforInstance(instance);
					correlation += calculateKendallRankCorrelation(ranking,perfectRanking);
				} catch (Exception e) {
					e.printStackTrace();
					numInstancesCalculated--;
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

		if (numInstancesCalculated != 0) {
			correlation /= numInstancesCalculated;
		}
		return correlation;
	}

	public static double calculateKendallRankCorrelation(List<Classifier> predictedOrdering,
			List<Classifier> actualOrdering) {
		double[] xArray = new double[predictedOrdering.size()];
		double[] yArray = new double[actualOrdering.size()];

		for (int i = 0; i < yArray.length; i++) {
			yArray[i] = i;
			for (int j = 0; j < actualOrdering.size(); j++) {
				if (predictedOrdering.get(i).getClass().getName().equals(actualOrdering.get(j).getClass().getName())) {
					xArray[j] = i;
				}
			}
		}

		System.out.println("\n Predicted Ranking:");
		for (double d : xArray) {
			System.out.print(d + ", ");
		}
		
		KendallsCorrelation correlation = new KendallsCorrelation();
	

		double result = correlation.correlation(xArray, yArray);
		System.out.println("Correlation: " + result);
		return result;
	}

}
