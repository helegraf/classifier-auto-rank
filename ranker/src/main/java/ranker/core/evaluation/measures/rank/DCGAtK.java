package ranker.core.evaluation.measures.rank;

import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ranker.core.evaluation.measures.RankerEvaluationMeasure;

/**
 * An implementation of the Discounted Cumulative Gain (DCG) according to
 * Definition 1 in Weimer, Markus, et al. "Cofi rank-maximum margin matrix
 * factorization for collaborative ranking." Advances in neural information
 * processing systems. 2008.
 * 
 * @author Helena Graf
 *
 */
public class DCGAtK extends RankerEvaluationMeasure {

	private int truncationThresholdK;
	private ExponentMode exponentMode = ExponentMode.INTEGER;
	private IDCGFunction dcgFunction = new COFIDCGFunction();
	private Logger logger = LoggerFactory.getLogger(DCGAtK.class);

	public enum ExponentMode {
		SCALED, TRUE_VALUE, INTEGER
	}

	/**
	 * Create an DCG@k evaluator with a truncation threshold at the given value
	 * (index starting at 1).
	 * 
	 * @param truncationThresholdK the truncation threshold
	 */
	public DCGAtK(int truncationThresholdK) {
		this.truncationThresholdK = truncationThresholdK;
	}

	@Override
	public double evaluate(List<String> predictedRanking, List<String> perfectRanking, List<Double> estimates,
			List<Double> performanceMeasures) {

		if (truncationThresholdK > predictedRanking.size()) {
			logger.warn("Truncation theshold {} is bigger than the length of the ranking. Setting Threshold to {}.",
					truncationThresholdK, predictedRanking.size());
			truncationThresholdK = predictedRanking.size();
		}

		// Generate permutation with ratings
		double[] permutationPi = new double[predictedRanking.size()];
		for (int i = 0; i < predictedRanking.size(); i++) {
			for (int j = 0; j < perfectRanking.size(); j++) {
				if (predictedRanking.get(i).equals(perfectRanking.get(j))) {
					switch (exponentMode) {
					case INTEGER:
						permutationPi[i] = ((double) permutationPi.length - j);
						break;
					case SCALED:
						permutationPi[i] = ((double) permutationPi.length - j) / permutationPi.length;
						break;
					case TRUE_VALUE:
						permutationPi[i] = performanceMeasures.get(j);
						break;
					}
					break;
				}
			}
		}

		// Compute dcgAtK
		double dcgAtK = 0;
		for (int i = 0; i < truncationThresholdK; i++) {
			dcgAtK += dcgFunction.dcgValue(permutationPi[i], i + 1);
		}
		return dcgAtK;
	}

	@Override
	public String getName() {
		return this.getClass().getSimpleName() + "_" + truncationThresholdK;
	}

	public ExponentMode getExponentMode() {
		return exponentMode;
	}

	public void setExponentMode(ExponentMode exponentMode) {
		this.exponentMode = exponentMode;
	}

	public IDCGFunction getDcgFunction() {
		return dcgFunction;
	}

	public void setDcgFunction(IDCGFunction dcgFunction) {
		this.dcgFunction = dcgFunction;
	}

}
