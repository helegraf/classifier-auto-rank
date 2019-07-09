package ranker.core.algorithms.regression;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ranker.core.algorithms.Ranker;
import ranker.core.evaluation.EvaluationHelper;
import ranker.core.evaluation.MCCV;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

public class WEKARegressionRanker extends RegressionRanker {
	
	private String name;
	
	public WEKARegressionRanker(String name) {
		this.name = name;
	}

	@Override
	protected void buildRegressionModels(Map<Classifier, Instances> train) throws Exception {
		regressionModels = new HashMap<Classifier,Classifier>();
		
		for (Map.Entry<Classifier, Instances> entry : train.entrySet()) {
			AbstractClassifier model = (AbstractClassifier) AbstractClassifier.forName(name,null);
			model.setDoNotCheckCapabilities(true);
			model.buildClassifier(entry.getValue());
			regressionModels.put(entry.getKey(), model);
		}
	}

	public String getAlgorithm() {
		return name;
	}
	
	public static void testM5pname(Instances data, List<Integer> targetAttributes) throws Exception {
		Ranker ranker = new WEKARegressionRanker("weka.classifiers.trees.M5P");
		ranker.buildRanker(data, targetAttributes);
	}
	
	public static void testM5preal(Instances data, List<Integer> targetAttributes) throws Exception {
		Ranker ranker = new M5PRanker();
		ranker.buildRanker(data, targetAttributes);
	}
	
	private static List<Integer> detectTargetAttributes(Instances data) {
		ArrayList<Integer> result = new ArrayList<>();
		
		for(int i = 0; i < data.numAttributes(); i++) {
			if (data.attribute(i).name().startsWith("weka.")) {
				result.add(i);
				System.out.println(data.attribute(i).name() + " is target.");
			}
		}
		
		System.out.println(result.size() + " targets");
		
		return result;
	}
	
	@Override public String getName() {
		return super.getName() + "_" + name;
	}
	
	@Override public String getClassifierString() {
		return super.getClassifierString() + "_" + name;
	}

}
