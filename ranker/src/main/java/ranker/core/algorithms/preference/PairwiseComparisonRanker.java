package ranker.core.algorithms.preference;

import de.upb.cs.is.jpl.api.algorithm.learningalgorithm.labelranking.labelrankingbypairwisecomparison.LabelRankingByPairwiseComparisonLearningAlgorithm;
import de.upb.cs.is.jpl.api.dataset.labelranking.LabelRankingDataset;
import de.upb.cs.is.jpl.api.math.RandomGenerator;

/**
 * A ranker that uses label ranking by pairwise comparison to predict rankings.
 * 
 * @author Helena Graf
 *
 */
public class PairwiseComparisonRanker extends PreferenceRanker {

	@Override
	protected void initialize() throws Exception {
		RandomGenerator.initializeRNG(1234);
		LabelRankingDataset dataset = convertToLabelRankingDataSet(data);
		LabelRankingByPairwiseComparisonLearningAlgorithm learningAlgorithm = new LabelRankingByPairwiseComparisonLearningAlgorithm();
		learningModel = learningAlgorithm.train(dataset);
	}

}
