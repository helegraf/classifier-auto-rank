package ranker.core.evaluation.measures.rank;

/**
 * Implements the DCG function according to the implementation of COFI-Rank [0].
 * 
 * <p>
 * [0] Weimer, Markus, et al. "Cofi rank-maximum margin matrix factorization for
 * collaborative ranking." Advances in neural information processing systems.
 * 2008.
 * 
 * @author helegraf
 *
 */
public class COFIDCGFunction implements IDCGFunction {

	@Override
	public double dcgValue(double permutationValue, int index) {
		return (Math.pow(2, permutationValue) - 1) / COFIDCGFunction.log2(index + 1);
	}

	/**
	 * Compute log2 of n.
	 * 
	 * @param n the value for which to compute the log
	 * @return the base to log of n
	 */
	public static double log2(int n) {
		return (Math.log(n) / Math.log(2));
	}
}
