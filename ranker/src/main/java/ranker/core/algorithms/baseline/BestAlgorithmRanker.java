package ranker.core.algorithms.baseline;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import ranker.core.algorithms.preference.PreferenceRanker;
import weka.core.Instance;

/**
 * A ranker which always returnes the same best-algorithm ranking. This is
 * computed by iteratively finding the learning algorithm which is ranked the
 * highest and ordering them accordingly.
 * 
 * @author Helena Graf
 *
 */
public class BestAlgorithmRanker extends PreferenceRanker {

	protected List<String> bestRanking;
	protected List<Integer> classifierStats;

	@Override
	protected void initialize() throws Exception {
		initializeLabels();

		// Build oracle for aggregating
		PerfectRanker oracle = new PerfectRanker();
		oracle.buildRanker(data, targetAttributes);

		// Initialize variables
		bestRanking = new ArrayList<>();
		classifierStats = new ArrayList<>();
		List<String> remaining = new ArrayList<>();
		classifiersMap.values().forEach(classifier -> remaining.add(classifier));

		// Find #Classifiers many times the best classifier
		for (int i = 0; i < targetAttributes.size(); i++) {
			// Scores count how many times an algorithm is ranked first
			HashMap<String, Integer> scores = new HashMap<>();
			remaining.forEach(classifier -> scores.put(classifier, 0));

			// Aggregate scores
			for (int j = 0; j < data.numInstances(); j++) {
				// Get ranking for instance
				List<String> ranking = oracle.predictRankingforInstance(data.get(j));

				// Find first (which algorithm is the first one in the ranking)
				for (int k = 0; k < ranking.size(); k++) {

					// First cannot be one that is already better
					boolean alreadyRanked = false;
					for (String bestClassifier : bestRanking) {
						if (bestClassifier.equals(ranking.get(k))) {
							alreadyRanked = true;
							break;
						}
					}

					if (alreadyRanked) {
						continue;
					}

					// Increment score
					for (String cl : scores.keySet()) {
						if (ranking.get(k).equals(cl)) {
							int previousScore = scores.get(cl);
							int newScore = previousScore + 1;
							scores.put(cl, newScore);
							break;
						}
					}
					
					// After found the first one, can stop
					break;
				}
			}

			// Find algorithm that is ranked first most often
			String bestClassifier = null;
			int highScore = -1;

			// Iterate over classifiers in order!
			for (int attributeIndex : targetAttributes) {
				for (String classifier : scores.keySet()) {
					if (classifier
							.equals(classifiersMap.get(attributeIndex))) {
						int score = scores.get(classifier);
						if (score > highScore) {
							bestClassifier = classifier;
							highScore = score;
						}
						break;
					}
				}
			}

			// Remove algorithm from remaining list and add to ranking
			bestRanking.add(bestClassifier);
			classifierStats.add(new Integer(highScore));
			remaining.remove(bestClassifier);
		}

	}

	@Override
	public List<String> predictRankingforInstance(Instance instance) {
		return bestRanking;
	}

	public List<Integer> getClassifierStats() {
		return classifierStats;
	}

}
