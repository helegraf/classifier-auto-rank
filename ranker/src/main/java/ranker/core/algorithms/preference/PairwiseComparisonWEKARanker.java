package ranker.core.algorithms.preference;

import de.upb.cs.is.jpl.api.algorithm.baselearner.regression.logisticWEKA.LogisticRegressionWEKA;
import de.upb.cs.is.jpl.api.algorithm.learningalgorithm.labelranking.labelrankingbypairwisecomparison.LabelRankingByPairwiseComparisonLearningAlgorithm;
import de.upb.cs.is.jpl.api.dataset.labelranking.LabelRankingDataset;
import de.upb.cs.is.jpl.api.math.RandomGenerator;

public class PairwiseComparisonWEKARanker extends PreferenceRanker {

	@Override
	protected void initialize() throws Exception {
		RandomGenerator.initializeRNG(1234);
		LabelRankingDataset dataSet = convertToLabelRankingDataSet(data);
		LabelRankingByPairwiseComparisonLearningAlgorithm algo = new LabelRankingByPairwiseComparisonLearningAlgorithm();
		LogisticRegressionWEKA baseLearnerAlgorithm = new LogisticRegressionWEKA();
		algo.getAlgorithmConfiguration().setBaseLearnerAlgorithm(baseLearnerAlgorithm);
		learningModel = algo.train(dataSet);
	}

}