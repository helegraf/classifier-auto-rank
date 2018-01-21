package ranker.algorithms;

import de.upb.cs.is.jpl.api.algorithm.learningalgorithm.labelranking.instancebasedlabelranking.InstanceBasedLabelRankingLearningAlgorithm;
import de.upb.cs.is.jpl.api.dataset.labelranking.LabelRankingDataset;
import weka.core.Instances;

public class InstanceBasedLabelRankingRanker extends PreferenceRanker {
	
	@Override
	public void buildRanker(Instances data) throws Exception {
		LabelRankingDataset dataset = convertToLabelRankingDataSet(data);
		InstanceBasedLabelRankingLearningAlgorithm learningAlgorithm = new InstanceBasedLabelRankingLearningAlgorithm();
		learningModel = learningAlgorithm.train(dataset);
	}
}
