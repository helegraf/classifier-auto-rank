package ranker.core.evaluation;

import java.util.List;

import weka.classifiers.Classifier;

/**
 * An implementation of the Discounted Cumulative Gain (DCG)
 * according to Definition 1 in Weimer, Markus, et al. "Cofi rank-maximum margin
 * matrix factorization for collaborative ranking." Advances in neural
 * information processing systems. 2008.
 * 
 * @author Helena Graf
 *
 */
public class DCGAtK extends RankerEvaluationMeasure {
	
	private int truncationThresholdK;
	
	/**
	 * Create an DCG@k evaluator with a truncation threshold at the given value
	 * (index starting at 1).
	 * 
	 * @param truncationThresholdK
	 *            the truncation threshold
	 */
	public DCGAtK (int truncationThresholdK) {
		this.truncationThresholdK = truncationThresholdK;
	}

	@Override
	public double evaluate(List<Classifier> predictedRanking, List<Classifier> perfectRanking, List<Double> estimates,
			List<Double> performanceMeasures) {
		
		// Generate permutation with ratings
		double[] permutationPi = new double[predictedRanking.size()];
		for (int i = 0; i < perfectRanking.size(); i++) {			
			for (int j = 0; j < predictedRanking.size(); j++) {
				if (predictedRanking.get(j).getClass().getName().equals(perfectRanking.get(i).getClass().getName())) {
					permutationPi[i] = ((double)permutationPi.length - j)/permutationPi.length;
					break;
				}
			}
		}
		
		// Compute dcgAtK
		double dcgAtK = 0;
		for (int i = 0; i < truncationThresholdK; i++) {
			dcgAtK += (Math.pow(2, permutationPi[i]) - 1) / Math.log((i+1) + 2);
		}		
		return dcgAtK;
	}
	
	@Override
	public String getName() {
		return this.getClass().getSimpleName() + "_" + truncationThresholdK;
	}

}
