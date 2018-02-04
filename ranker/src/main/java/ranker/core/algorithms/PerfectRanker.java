package ranker.core.algorithms;

import java.util.ArrayList;
import java.util.List;

import de.upb.cs.is.jpl.api.dataset.defaultdataset.relative.Ranking;
import de.upb.cs.is.jpl.api.exception.algorithm.PredictionFailedException;
import weka.classifiers.Classifier;
import weka.core.Instance;

/**
 * Ranker used for evaluation purposes. Has to be built with
 * {@link #buildRanker(weka.core.Instances, List)} to know which attributes are
 * targets and which are meta features. Query instance given to
 * {@link #predictRankingforInstance(Instance)} has to include performance
 * values for the learning algorithm learning algorithms in order to return a
 * correct ranking. The method {@link #getEstimates()} then likewise returns the
 * correct performance values ordered according to the last returned ranking.
 * 
 * @author Helena Graf
 *
 */
public class PerfectRanker extends PreferenceRanker {

	/**
	 * List with actual performance values ordered according to the last returned
	 * ranking. Null if no ranking has been predicted.
	 */
	protected List<Double> performanceMeasures;

	@Override
	public List<Classifier> predictRankingforInstance(Instance instance) throws PredictionFailedException {
		Ranking ranking = getRankingForInstance(instance);

		// Get actual performance measures
		performanceMeasures = new ArrayList<Double>();
		for (int label : ranking.getObjectList()) {
			int algorithmIndex = reverseIndicesMap.get(label);
			performanceMeasures.add(instance.value(algorithmIndex));
		}

		return mapToOrdering(ranking);
	}

	@Override
	public List<Double> getEstimates() {
		return performanceMeasures;
	}
	
	@Override
	protected void initialize() throws Exception {
		initializeLabels();
	}
}
