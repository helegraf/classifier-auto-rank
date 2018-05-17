package ranker.core.evaluation;

import java.util.List;

import ranker.core.algorithms.PerformanceOrder;
import weka.classifiers.Classifier;

public class Loss extends RankerEvaluationMeasure {
	
	protected PerformanceOrder order = PerformanceOrder.DESCENDING;

	@Override
	public double evaluate(List<Classifier> predictedRanking, List<Classifier> perfectRanking, List<Double> estimates,
			List<Double> performanceMeasures) {
		System.out.print("Loss ");
		double loss = Double.NaN;
		
		// Find where the best algorithm in the predicted ranking is actually placed
		for (int i = 0; i < perfectRanking.size(); i++) {
			if (predictedRanking.get(0).getClass().getName().equals(perfectRanking.get(i).getClass().getName())) {
				// Get actual values of both
				double best = performanceMeasures.get(0);
				double predicted = performanceMeasures.get(i);

				// Calculate loss 
				if (order == PerformanceOrder.ASCENDING) {
					// best >= predicted
					loss = best - predicted;
				} else {
					// best <= predicted
					loss = predicted - best;
				}
			}
		}
		System.out.println(loss);
		return loss;
	}

	/**
	 * Gets the Performance Order. Standard is {@link PerformanceOrder#DESCENDING}
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
