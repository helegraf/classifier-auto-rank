package ranker;

import de.upb.cs.is.jpl.api.algorithm.learningalgorithm.labelranking.labelrankingbypairwisecomparison.LabelRankingByPairwiseComparisonLearningAlgorithm;
import de.upb.cs.is.jpl.api.dataset.labelranking.LabelRankingDataset;
import weka.core.Instances;

public class PairwiseComparisonRanker extends PreferenceRanker {

	@Override
	public void buildRanker(Instances data) throws Exception {
		System.out.println("Building Ranker");
		LabelRankingDataset dataset = convertToLabelRankingDataSet(data);
		LabelRankingByPairwiseComparisonLearningAlgorithm learningAlgorithm = new LabelRankingByPairwiseComparisonLearningAlgorithm();
		learningModel = learningAlgorithm.train(dataset);

	}

}
