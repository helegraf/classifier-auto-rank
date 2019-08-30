package ranker.core.algorithms.decomposition.regression;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ranker.core.algorithms.Ranker;
import ranker.core.algorithms.decomposition.DecompositionRanker;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instances;

public class WEKARegressionRanker extends DecompositionRanker {
	
	private String name;
	
	public WEKARegressionRanker(String name) {
		this.name = name;
	}

	@Override
	protected void buildRegressionModels(Map<String, Instances> train) throws Exception {
		regressionModels = new HashMap<String,Classifier>();
		
		for (Map.Entry<String, Instances> entry : train.entrySet()) {
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
