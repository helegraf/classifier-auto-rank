package ranker.core.evaluation.significance;

/**
 * Encapsulates results of the computation of the Mann-Whitney U (U and p
 * value).
 * 
 * @author Helena Graf
 *
 */
public class MannWhitneyUEvaluationResult {
	private final double MannWhitneyU;
	private final double pValue;
	private String firstCombination;
	private String secondCombination;
	private String measureName;

	/**
	 * Constructs a new {@link MannWhitneyUEvaluationResult} object with the given
	 * values.
	 *
	 * @param MannWhitneyU
	 *            The Mann-Whitney U
	 * @param pValue
	 *            The p value
	 * @param firstCombination
	 *            The first combination of ranker and data set used in the
	 *            computation of the U and p value
	 * @param secondCombination
	 *            The second combination of ranker and data set used in the
	 *            computation of the U and p value
	 * @param measureName
	 *            The name of the measure used in the computation of the U and p
	 *            value
	 */
	public MannWhitneyUEvaluationResult(double MannWhitneyU, double pValue, String firstCombination,
			String secondCombination, String measureName) {
		this.MannWhitneyU = MannWhitneyU;
		this.pValue = pValue;
		this.firstCombination = firstCombination;
		this.secondCombination = secondCombination;
		this.measureName = measureName;
	}

	/**
	 * Gets the Mann-Whitney U.
	 * 
	 * @return The Mann-Whitney U
	 */
	public double getMannWhitneyU() {
		return MannWhitneyU;
	}

	/**
	 * Gets the p value.
	 * 
	 * @return The p value
	 */
	public double getPValue() {
		return pValue;
	}

	/**
	 * Gets the name of the first combination of ranker and data set used in the
	 * computation of the Mann-Whitney U and p value.
	 * 
	 * @return The first combination of ranker and data set
	 */
	public String getfirstCombination() {
		return firstCombination;
	}

	/**
	 * Gets the name of the second combination of ranker and data set used in the
	 * computation of the Mann-Whitney U and p value.
	 * 
	 * @return The second combination of ranker and data set
	 */
	public String getSecondCombination() {
		return secondCombination;
	}

	@Override
	public String toString() {
		return firstCombination + " vs " + secondCombination + " regarding " + measureName + System.lineSeparator()
				+ "U: " + MannWhitneyU + " p value: " + pValue;
	}
}
