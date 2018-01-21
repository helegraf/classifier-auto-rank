package ranker.algorithms;

import java.util.List;

import de.upb.cs.is.jpl.api.exception.algorithm.PredictionFailedException;
import weka.classifiers.Classifier;
import weka.core.Instance;

/**
 * Ranker used for evaluation purposes. Has to be initialized for the purpose of
 * knowing which attributes are targets and which are meta features. Query
 * instance has to include performance values for learning algorithm.
 * 
 * @author Helena Graf
 *
 */
public class PerfectRanker extends PreferenceRanker {

	@Override
	protected void initialize() throws Exception {
		initializeLabels();
	}

	@Override
	public List<Classifier> predictRankingforInstance(Instance instance) throws PredictionFailedException {
		return mapToOrdering(getRankingForInstance(instance));
	}

}
