package ranker.core.algorithms;

import java.util.HashMap;
import java.util.Map;

import weka.classifiers.Classifier;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;

public class LinearRegressionRanker extends RegressionRanker {

	@Override
	protected void buildRegressionModels(Map<Classifier, Instances> train) throws Exception {
		regressionModels = new HashMap<Classifier,Classifier>();
		for (Classifier classifier : train.keySet()) {
			LinearRegression lineReg = new LinearRegression();
			lineReg.buildClassifier(train.get(classifier));
			regressionModels.put(classifier, lineReg);
		}
	}

}
