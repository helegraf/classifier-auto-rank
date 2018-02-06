package ranker.core.algorithms;

import de.upb.cs.is.jpl.api.algorithm.baselearner.IBaselearnerAlgorithm;
import de.upb.cs.is.jpl.api.algorithm.baselearner.classification.knearestneighbor.KNearestNeighborConfiguration;
import de.upb.cs.is.jpl.api.algorithm.learningalgorithm.labelranking.instancebasedlabelranking.InstanceBasedLabelRankingConfiguration;
import de.upb.cs.is.jpl.api.algorithm.learningalgorithm.labelranking.instancebasedlabelranking.InstanceBasedLabelRankingLearningAlgorithm;
import de.upb.cs.is.jpl.api.algorithm.learningalgorithm.rankaggregation.bordacount.BordaCountConfiguration;
import de.upb.cs.is.jpl.api.algorithm.learningalgorithm.rankaggregation.bordacount.BordaCountLearningAlgorithm;
import de.upb.cs.is.jpl.api.algorithm.learningalgorithm.rankaggregation.kemenyyoung.KemenyYoungLearningAlgorithm;
import de.upb.cs.is.jpl.api.algorithm.learningalgorithm.rankaggregation.plackettluce.PlackettLuceConfiguration;
import de.upb.cs.is.jpl.api.algorithm.learningalgorithm.rankaggregation.plackettluce.PlackettLuceLearningAlgorithm;
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
		InstanceBasedLabelRankingConfiguration config = learningAlgorithm.getAlgorithmConfiguration();
		config.setRankAggregationAlgorithm(new KemenyYoungLearningAlgorithm());
//		KNearestNeighborConfiguration baseConfig = (KNearestNeighborConfiguration) config.getBaseLearnerAlgorithm().getAlgorithmConfiguration();
//		baseConfig.setNumberOfNeighbors((int)Math.floor(Math.sqrt(dataset.getNumberOfInstances())));
		learningModel = learningAlgorithm.train(dataset);
	}
}
