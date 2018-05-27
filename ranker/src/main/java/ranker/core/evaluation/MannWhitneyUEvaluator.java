package ranker.core.evaluation;

import java.io.IOException;
import java.nio.file.FileSystems;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.math3.stat.inference.MannWhitneyUTest;
import org.apache.commons.math3.stat.ranking.NaNStrategy;
import org.apache.commons.math3.stat.ranking.TiesStrategy;

import ranker.core.algorithms.preference.InstanceBasedLabelRankingKemenyYoung;
import ranker.core.algorithms.preference.InstanceBasedLabelRankingKemenyYoungSQRTN;
import ranker.core.algorithms.preference.InstanceBasedLabelRankingRanker;
import ranker.core.algorithms.preference.PairwiseComparisonRanker;
import ranker.core.algorithms.regression.LinearRegressionRanker;
import ranker.core.algorithms.regression.M5PRanker;
import ranker.core.algorithms.regression.REPTreeRanker;
import ranker.core.algorithms.regression.RandomForestRanker;
import ranker.util.CSVHelper;
import ranker.util.ColumnNotFoundException;

/**
 * Computes the Mann-Whitney U and p-value for given combinations of rankers,
 * data sets and measures.
 * 
 * @author Helena Graf
 *
 */
public class MannWhitneyUEvaluator {
	// TODO move these somewhere else

	public static final String ALL_META_DATA_DATA_SET_NAME = "metaData_small_allPerformanceValues";
	public static final String ONLY_PROBING_META_DATA_SET_NAME = "metaData_small_allPerformanceValues-weka.filters.unsupervised.attribute.Remove-R1-48,94-104";
	public static final String NO_PROBING_META_DATA_DATA_SET_NAME = "metaData_small_allPerformanceValues-weka.filters.unsupervised.attribute.Remove-R49-93";

	public static final List<String> ALL_DATA_SET_NAMES = Arrays.asList(ALL_META_DATA_DATA_SET_NAME,
			ONLY_PROBING_META_DATA_SET_NAME, NO_PROBING_META_DATA_DATA_SET_NAME);

	public static final List<String> ALL_MEASURE_NAMES = Arrays.asList(KendallRankCorrelation.class.getSimpleName(),
			Loss.class.getSimpleName(), BestThreeLoss.class.getSimpleName());

	public static final List<String> ALL_RANKER_NAMES = Arrays.asList(
			InstanceBasedLabelRankingRanker.class.getSimpleName(),
			InstanceBasedLabelRankingKemenyYoung.class.getSimpleName(),
			InstanceBasedLabelRankingKemenyYoungSQRTN.class.getSimpleName(),
			PairwiseComparisonRanker.class.getSimpleName(), RandomForestRanker.class.getSimpleName(),
			LinearRegressionRanker.class.getSimpleName(), REPTreeRanker.class.getSimpleName(),
			M5PRanker.class.getSimpleName());

	/**
	 * Creates a new {@link MannWhitneyUEvaluationResult} for the given results. The
	 * results of the first ranker with the first meta data set and the second
	 * ranker with the second meta data set are compared according to the given
	 * measure.
	 * 
	 * @param firstRankerName
	 *            The name of the first ranker
	 * @param firstDataSetName
	 *            The name of the first data set
	 * @param secondRankerName
	 *            The name of the second ranker
	 * @param secondDataSetName
	 *            The name of the second data set
	 * @param measureName
	 *            The measure according to which to compare the evaluation results
	 *            (must be a column name)
	 * @return
	 * @throws IOException
	 *             If an Exception occurs while reading one of the files
	 * @throws ColumnNotFoundException
	 *             If the given measure is not a valid column name in one of the
	 *             files
	 */
	public static MannWhitneyUEvaluationResult computeWhitneyU(String firstRankerName, String firstDataSetName,
			String secondRankerName, String secondDataSetName, String measureName)
			throws IOException, ColumnNotFoundException {
		// get the values
		double[] xArray = CSVHelper.getColumnValues(
				FileSystems.getDefault().getPath(CSVHelper.getCSVPath(firstRankerName, firstDataSetName)), measureName);
		double[] yArray = CSVHelper.getColumnValues(
				FileSystems.getDefault().getPath(CSVHelper.getCSVPath(secondRankerName, firstDataSetName)),
				measureName);

		// compute result
		MannWhitneyUTest uTest = new MannWhitneyUTest(NaNStrategy.REMOVED, TiesStrategy.AVERAGE);
		double u = uTest.mannWhitneyU(xArray, yArray);
		double p = uTest.mannWhitneyUTest(xArray, yArray);

		// return result
		return new MannWhitneyUEvaluationResult(u, p, firstRankerName + firstDataSetName,
				secondRankerName + secondDataSetName, measureName);
	}
}
