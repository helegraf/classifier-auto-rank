package ranker.core.algorithms.preference;

import de.upb.cs.is.jpl.api.algorithm.learningalgorithm.labelranking.instancebasedlabelranking.InstanceBasedLabelRankingLearningAlgorithm;
import de.upb.cs.is.jpl.api.dataset.labelranking.LabelRankingDataset;
import de.upb.cs.is.jpl.api.math.RandomGenerator;

/**
 * A ranker that uses an instance based label ranking learning algorithm to predict rankings.
 * 
 * @author Helena Graf
 *
 */
public class InstanceBasedLabelRankingRanker extends PreferenceRanker {

	@Override
	protected void initialize() throws Exception {
		RandomGenerator.initializeRNG(1234);
		LabelRankingDataset dataset = convertToLabelRankingDataSet(data);
		InstanceBasedLabelRankingLearningAlgorithm learningAlgorithm = new InstanceBasedLabelRankingLearningAlgorithm();
		learningModel = learningAlgorithm.train(dataset);
	}
}
