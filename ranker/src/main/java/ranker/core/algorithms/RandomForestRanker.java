package ranker.core.algorithms;

import java.util.HashMap;
import java.util.Map;

import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

/**
 * A ranker that uses random forests for predicting rankings.
 * 
 * @author Helena Graf
 *
 */
public class RandomForestRanker extends RegressionRanker {

	@Override
	protected void buildRegressionModels(Map<Classifier, Instances> train) throws Exception {
		regressionModels = new HashMap<Classifier,Classifier>();
		for (Classifier classifier : train.keySet()) {
			RandomForest forest = new RandomForest();
			forest.buildClassifier(train.get(classifier));
			regressionModels.put(classifier, forest);
		}
	}

}
