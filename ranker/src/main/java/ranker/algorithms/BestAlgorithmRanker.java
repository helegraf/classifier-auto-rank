package ranker.algorithms;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

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

	@Override
	protected void initialize() throws Exception {
		initializeLabels();

		// Build oracle for aggregating
		PerfectRanker oracle = new PerfectRanker();
		oracle.buildRanker(data, targetAttributes);

		// Initialize variables
		bestRanking = new ArrayList<Classifier>();
		List<Classifier> remaining = new ArrayList<Classifier>();
		classifiersMap.values().forEach(classifier -> remaining.add(classifier));

		// Find ranking
		for (int i = 0; i < targetAttributes.size(); i++) {
			// Scores count how many times an algorithm is ranked first
			HashMap<Classifier, Integer> scores = new HashMap<Classifier, Integer>();
			remaining.forEach(classifier -> scores.put(classifier, 0));

			// Aggregate scores
			for (int j = 0; j < data.numInstances(); j++) {
				// Get ranking for instance
				List<Classifier> ranking = oracle.predictRankingforInstance(data.get(j));

				// Find first
				for (int k = 0; k < ranking.size(); k++) {
					if (!bestRanking.contains(ranking.get(k))) {
						// Increment score
						int previousScore = scores.get(ranking.get(k));
						int newScore = previousScore++;
						scores.put(ranking.get(k), newScore);
						break;
					}
				}
			}

			// Find algorithm that is ranked first most often
			Classifier bestClassifier = null;
			int highScore = -1;
			for (Classifier classifier : scores.keySet()) {
				int score = scores.get(classifier);
				if (score > highScore) {
					bestClassifier = classifier;
					highScore = score;
				}
			}

			// Remove algorithm from remaining list and add to ranking
			bestRanking.add(bestClassifier);
			remaining.remove(bestClassifier);
		}

	}

	@Override
	public List<Classifier> predictRankingforInstance(Instance instance) {
		return bestRanking;
	}

}
