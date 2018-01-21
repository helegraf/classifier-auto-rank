package ranker.algorithms;

import de.upb.cs.is.jpl.api.algorithm.learningalgorithm.labelranking.labelrankingbypairwisecomparison.LabelRankingByPairwiseComparisonLearningAlgorithm;
import de.upb.cs.is.jpl.api.dataset.labelranking.LabelRankingDataset;

/**
 * A ranker that uses label ranking by pairwise comparison to predict rankings.
 * 
 * @author Helena Graf
 *
 */
public class PairwiseComparisonRanker extends PreferenceRanker {

	@Override
	protected void initialize() throws Exception {
		LabelRankingDataset dataset = convertToLabelRankingDataSet(data);
		LabelRankingByPairwiseComparisonLearningAlgorithm learningAlgorithm = new LabelRankingByPairwiseComparisonLearningAlgorithm();
		learningModel = learningAlgorithm.train(dataset);
	}

}
