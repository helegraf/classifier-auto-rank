package ranker.core.evaluation.measures.real;

import java.util.List;

import ranker.core.algorithms.PerformanceOrder;
import ranker.core.evaluation.measures.RankerEvaluationMeasure;

/**
 * Computes the minimal loss among the first n elements. Assumes max. loss of 100.
 * 
 * @author Helena Graf
 *
 */
public class BestNLoss extends RankerEvaluationMeasure {

	protected PerformanceOrder order = PerformanceOrder.ASCENDING;
	
	private int n;
	
	/**
	 * Constructs a new BestNLoss object using the given value as a cutoff.
	 * 
	 * @param n the cutoff point (inclusive, starting from 1)
	 */
	public BestNLoss(int n) {
		this.n = n;
	}

	@Override
	public double evaluate(List<String> predictedRanking, List<String> perfectRanking, List<Double> estimates,
			List<Double> performanceMeasures) {
		// assuming no NaNs
		double loss = Double.NaN;
		
		// get actual loss of the first in the predicted ranking
		for (int i = 0; i < n; i++) {
			double nthloss = getNthLoss(i,predictedRanking,perfectRanking,performanceMeasures);
			if (!Double.isNaN(nthloss)) {
				if (Double.isNaN(loss)) {
					loss = nthloss;
				} else {
					loss = Math.min(loss, nthloss);
				}				
			}		
		}
		
		return loss;
	}

	private double getNthLoss(int n, List<String> predictedRanking, List<String> perfectRanking, List<Double> performanceMeasures) {
		// Get the classifier for which to get the loss
		String classifierName = predictedRanking.get(n);
		double performance = Double.NaN;
		
		for (int i = 0; i < perfectRanking.size(); i++) {
			if (perfectRanking.get(i).equals(classifierName)) {
				performance = performanceMeasures.get(i);
				break;
			}
		}
		
		// Compute difference between this classifier and the best classifier
		return performanceMeasures.get(0) - performance;
	}

	/**
	 * Gets the Performance Order. Standard is {@link PerformanceOrder#DESCENDING}
	 * 
	 * @return the used performance order
	 */
	public PerformanceOrder getPerformanceOrder() {
		return order;
	}

	/**
	 * Sets the Performance Order. Standard is {@link PerformanceOrder#DESCENDING}
	 * 
	 * @param order the new order
	 */
	public void setPerformanceOrder(PerformanceOrder order) {
		this.order = order;
	}
	
	@Override
	public String getName() {
		return "Best" + n + "Loss";
	}

}
