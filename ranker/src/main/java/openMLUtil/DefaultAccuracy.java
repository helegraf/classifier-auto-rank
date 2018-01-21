package openMLUtil;

import java.util.HashMap;
import java.util.Map;

import org.openml.webapplication.fantail.dc.Characterizer;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.rules.ZeroR;
import weka.core.Instances;

public class DefaultAccuracy extends Characterizer {
	
	private String[] ids = {"DefaultAccuracy"};

	@Override
	public String[] getIDs() {
		return ids;
	}

	@Override
	public Map<String, Double> characterize(Instances instances) {
		Classifier classifier = new ZeroR();
		double result;
		try {
			classifier.buildClassifier(instances);
			Evaluation eval = new Evaluation(instances);
			eval.evaluateModel(classifier, instances);
			result = eval.pctCorrect();
		} catch (Exception e) {
			result = 0;
		}
		HashMap<String,Double> results = new HashMap<String,Double>();
		results.put(ids[0], result);
		return results;
	}

}
