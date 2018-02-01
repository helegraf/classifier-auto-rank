package rankerEvaluation;

import java.util.ArrayList;
import java.util.List;

import weka.classifiers.Classifier;

public class BestThreeLoss extends RankerEvaluationMeasure {

	protected PerformanceOrder order = PerformanceOrder.DESCENDING;

	@Override
	public double evaluate(List<Classifier> predictedRanking, List<Classifier> perfectRanking, List<Double> estimates,
			List<Double> performanceMeasures) {
		// Initialize local variables
		double loss = Double.NaN;
		Loss lossEvaluation = new Loss();
		lossEvaluation.setPerformanceOrder(order);

		// Calculate first loss
		double firstLoss = lossEvaluation.evaluate(predictedRanking, perfectRanking, estimates, performanceMeasures);
		loss = firstLoss;

		// Remove the first learning algorithm from both lists & calculate second loss
		List<Classifier> newPredictedRanking = predictedRanking.subList(1, predictedRanking.size());
		//List<Double> newEstimates = estimates.subList(1, estimates.size());
		List<Classifier> newPerfectRanking = new ArrayList<Classifier>();
		List<Double> newPerformanceMeasures = new ArrayList<Double>();
		for (int i = 0; i < perfectRanking.size(); i++) {
			for (int j = 0; j < newPredictedRanking.size(); j++) {
				if (newPredictedRanking.get(j).getClass().getName().equals(perfectRanking.get(i).getClass().getName())) {
					newPerfectRanking.add(perfectRanking.get(i));
					newPerformanceMeasures.add(performanceMeasures.get(i));
					break;
				}
			}
		}
		double secondLoss = lossEvaluation.evaluate(newPredictedRanking, newPerfectRanking, estimates,
				newPerformanceMeasures);
		if (!Double.isNaN(secondLoss) && ((!Double.isNaN(loss) && secondLoss < loss) || Double.isNaN(loss))) {
			loss = secondLoss;
		}

		// Remove the first learning algorithm from both lists & calculate second loss
		newPredictedRanking = newPredictedRanking.subList(1, newPredictedRanking.size());
		//newEstimates = newEstimates.subList(1, newEstimates.size());
		newPerfectRanking = new ArrayList<Classifier>();
		newPerformanceMeasures = new ArrayList<Double>();
		for (int i = 0; i < perfectRanking.size(); i++) {
			for (int j = 0; j < newPredictedRanking.size(); j++) {
				if (newPredictedRanking.get(j).getClass().getName().equals(perfectRanking.get(i).getClass().getName())) {
					newPerfectRanking.add(perfectRanking.get(i));
					newPerformanceMeasures.add(performanceMeasures.get(i));
					break;
				}
			}
		}
		double thirdLoss = lossEvaluation.evaluate(newPredictedRanking, newPerfectRanking, estimates,
				performanceMeasures);
		if (!Double.isNaN(thirdLoss) && ((!Double.isNaN(loss) && thirdLoss < loss) || Double.isNaN(loss))) {
			loss = thirdLoss;
		}
		
		System.out.print("Best 3 Loss ");
		System.out.println(loss);
		return loss;
	}

	/**
	 * Gets the Performance Order. Standard is {@link PerformanceOrder#DESCENDING}
	 * 
	 * @param order
	 */
	public PerformanceOrder getPerformanceOrder() {
		return order;
	}

	/**
	 * Sets the Performance Order. Standard is {@link PerformanceOrder#DESCENDING}
	 * 
	 * @param order
	 */
	public void setPerformanceOrder(PerformanceOrder order) {
		this.order = order;
	}

}
