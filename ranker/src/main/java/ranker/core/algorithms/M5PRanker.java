package ranker.core.algorithms;

import java.util.HashMap;
import java.util.Map;

import weka.classifiers.Classifier;
import weka.classifiers.trees.M5P;
import weka.core.Instances;

public class M5PRanker extends RegressionRanker {

	@Override
	protected void buildRegressionModels(Map<Classifier, Instances> train) throws Exception {
		regressionModels = new HashMap<Classifier,Classifier>();
		for (Classifier classifier : train.keySet()) {
			M5P m5p = new M5P();
			m5p.buildClassifier(train.get(classifier));
			regressionModels.put(classifier, m5p);
		}
	}
}
