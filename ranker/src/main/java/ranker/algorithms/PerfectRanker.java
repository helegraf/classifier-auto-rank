package ranker.algorithms;

import java.util.ArrayList;
import java.util.List;

import de.upb.cs.is.jpl.api.dataset.defaultdataset.relative.Ranking;
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
	
	protected List<Double> performanceMeasures;

	@Override
	protected void initialize() throws Exception {
		initializeLabels();
	}

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

	/**
	 * Returns the actual performance measures for the last ranking that was
	 * requested. The value at position i is the performance measure of the learning
	 * algorithm at position i in the returned ranking.
	 * 
	 * @return The performance measures
	 */
	public List<Double> getPerformanceMeasuresForRanking() {
		return performanceMeasures;
	}

}
