package ranker.core.evaluation.measures.rank;

/**
 * A Function computing the DCG for 1 part of the DCG-sum for a given
 * permutation-value and index.
 * 
 * @author helegraf
 *
 */
@FunctionalInterface
public interface IDCGFunction {

	/**
	 * Compute the DCG for 1 part of its sum.
	 * 
	 * @param permutationValue the relevance of the label ranked at the given index
	 * @param index            the current index
	 * @return the DCG-value
	 */
	public double dcgValue(double permutationValue, int index);
}
