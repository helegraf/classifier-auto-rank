package ranker.core.algorithms.preference;

import de.upb.cs.is.jpl.api.algorithm.baselearner.classification.knearestneighbor.KNearestNeighborConfiguration;
import de.upb.cs.is.jpl.api.algorithm.learningalgorithm.labelranking.instancebasedlabelranking.InstanceBasedLabelRankingConfiguration;
import de.upb.cs.is.jpl.api.algorithm.learningalgorithm.labelranking.instancebasedlabelranking.InstanceBasedLabelRankingLearningAlgorithm;
import de.upb.cs.is.jpl.api.algorithm.learningalgorithm.rankaggregation.kemenyyoung.KemenyYoungLearningAlgorithm;
import de.upb.cs.is.jpl.api.dataset.labelranking.LabelRankingDataset;
import de.upb.cs.is.jpl.api.math.RandomGenerator;

public class InstanceBasedLabelRankingKemenyYoungSQRTN extends PreferenceRanker {
	
	@Override
	protected void initialize() throws Exception {
		RandomGenerator.initializeRNG(1234);
		LabelRankingDataset dataset = convertToLabelRankingDataSet(data);
		InstanceBasedLabelRankingLearningAlgorithm learningAlgorithm = new InstanceBasedLabelRankingLearningAlgorithm();
		InstanceBasedLabelRankingConfiguration config = learningAlgorithm.getAlgorithmConfiguration();
		config.setRankAggregationAlgorithm(new KemenyYoungLearningAlgorithm());
		KNearestNeighborConfiguration baseConfig = (KNearestNeighborConfiguration) config.getBaseLearnerAlgorithm().getAlgorithmConfiguration();
		baseConfig.setNumberOfNeighbors((int)Math.floor(Math.sqrt(dataset.getNumberOfInstances())));
		learningModel = learningAlgorithm.train(dataset);
	}

}
