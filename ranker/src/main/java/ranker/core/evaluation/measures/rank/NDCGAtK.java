package ranker.core.evaluation.measures.rank;

import java.util.List;

import ranker.core.evaluation.measures.RankerEvaluationMeasure;

/**
 * An implementation of the Normalized Discounted Cumulative Gain (NDCG)
 * according to Definition 1 in Weimer, Markus, et al. "Cofi rank-maximum margin
 * matrix factorization for collaborative ranking." Advances in neural
 * information processing systems. 2008.
 * 
 * @author Helena Graf
 *
 */
public class NDCGAtK extends RankerEvaluationMeasure {

	private DCGAtK dcgAtk;

	/**
	 * Create an NDCG@k evaluator with a truncation threshold at the given value
	 * (index starting at 1).
	 * 
	 * @param truncationThresholdK
	 *            the truncation threshold
	 */
	public NDCGAtK(int truncationThresholdK) {
		this.dcgAtk = new DCGAtK(truncationThresholdK);
	}

	@Override
	public double evaluate(List<String> predictedRanking, List<String> perfectRanking, List<Double> estimates,
			List<Double> performanceMeasures) {
		return dcgAtk.evaluate(predictedRanking, perfectRanking, estimates, performanceMeasures)
				/ dcgAtk.evaluate(perfectRanking, perfectRanking, performanceMeasures, performanceMeasures);
	}
	
	@Override
	public String getName() {
		return this.getClass().getSimpleName() + "_" + dcgAtk.getName();
	}
}
