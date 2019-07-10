package ranker.core.algorithms;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import ranker.core.algorithms.preference.PreferenceRanker;
import weka.classifiers.Classifier;
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

	protected List<Classifier> bestRanking;
	protected List<Integer> classifierStats;

	@Override
	protected void initialize() throws Exception {
		initializeLabels();

		// Build oracle for aggregating
		PerfectRanker oracle = new PerfectRanker();
		oracle.buildRanker(data, targetAttributes);

		// Initialize variables
		bestRanking = new ArrayList<Classifier>();
		classifierStats = new ArrayList<Integer>();
		List<Classifier> remaining = new ArrayList<Classifier>();
		classifiersMap.values().forEach(classifier -> remaining.add(classifier));

		// Find #Classifiers many times the best classifier
		for (int i = 0; i < targetAttributes.size(); i++) {
			// Scores count how many times an algorithm is ranked first
			HashMap<Classifier, Integer> scores = new HashMap<Classifier, Integer>();
			remaining.forEach(classifier -> scores.put(classifier, 0));

			// Aggregate scores
			for (int j = 0; j < data.numInstances(); j++) {
				// Get ranking for instance
				List<Classifier> ranking = oracle.predictRankingforInstance(data.get(j));

				// Find first (which algorithm is the first one in the ranking)
				for (int k = 0; k < ranking.size(); k++) {

					// First cannot be one that is already better
					boolean alreadyRanked = false;
					for (Classifier bestClassifier : bestRanking) {
						if (bestClassifier.getClass().getName().equals(ranking.get(k).getClass().getName())) {
							alreadyRanked = true;
							break;
						}
					}

					if (alreadyRanked) {
						continue;
					}

					// Increment score
					for (Classifier cl : scores.keySet()) {
						if (ranking.get(k).getClass().getName().equals(cl.getClass().getName())) {
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
			Classifier bestClassifier = null;
			int highScore = -1;

			// Iterate over classifiers in order!
			for (int attributeIndex : targetAttributes) {
				for (Classifier classifier : scores.keySet()) {
					if (classifier.getClass().getName()
							.equals(classifiersMap.get(attributeIndex).getClass().getName())) {
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
	public List<Classifier> predictRankingforInstance(Instance instance) {
		return bestRanking;
	}

	public List<Integer> getClassifierStats() {
		return classifierStats;
	}

}
