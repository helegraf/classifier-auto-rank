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

	private DCGAtK dcgAtK;

	/**
	 * Create an NDCG@k evaluator with a truncation threshold at the given value
	 * (index starting at 1).
	 * 
	 * @param truncationThresholdK
	 *            the truncation threshold
	 */
	public NDCGAtK(int truncationThresholdK) {
		this.dcgAtK = new DCGAtK(truncationThresholdK);
	}
	
	public NDCGAtK(DCGAtK dcgAtK) {
		this.dcgAtK = dcgAtK;
	}

	@Override
	public double evaluate(List<String> predictedRanking, List<String> perfectRanking, List<Double> estimates,
			List<Double> performanceMeasures) {
		return dcgAtK.evaluate(predictedRanking, perfectRanking, estimates, performanceMeasures)
				/ dcgAtK.evaluate(perfectRanking, perfectRanking, performanceMeasures, performanceMeasures);
	}
	
	@Override
	public String getName() {
		return this.getClass().getSimpleName() + "_" + dcgAtK.getName();
	}
}
