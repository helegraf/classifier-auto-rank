package ranker.core.algorithms.regression;
import java.util.HashMap;
import java.util.Map;

import weka.classifiers.Classifier;
import weka.classifiers.trees.REPTree;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

public class REPTreeRanker extends RegressionRanker {

	@Override
	protected void buildRegressionModels(Map<Classifier, Instances> train) throws Exception {
		regressionModels = new HashMap<Classifier,Classifier>();
		for (Classifier classifier : train.keySet()) {
			REPTree repTree = new REPTree();
			repTree.buildClassifier(train.get(classifier));
			regressionModels.put(classifier, repTree);
		}
	}

}
