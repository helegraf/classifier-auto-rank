package ranker.core.algorithms.decomposition.regression;

import java.util.HashMap;
import java.util.Map;

import ranker.core.algorithms.decomposition.DecompositionRanker;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 * Ranks algorithms by predicting the performance of each algorithm with a WEKA classifier.
 * 
 * @author helegraf
 *
 */
public class WEKARegressionRanker extends DecompositionRanker {
	
	private String name;
	
	/**
	 * Constructs a new WEKARegressionRanker that will use the given classifier to predict the peformance of each algorithm.
	 * 
	 * @param name the fully qualified name of a weka classifier
	 */
	public WEKARegressionRanker(String name) {
		this.name = name;
	}

	@Override
	protected void buildModels(Map<String, Instances> train) throws Exception {
		models = new HashMap<String,Classifier>();
		
		for (Map.Entry<String, Instances> entry : train.entrySet()) {
			AbstractClassifier model = (AbstractClassifier) AbstractClassifier.forName(name,null);
			model.setDoNotCheckCapabilities(true);
			model.buildClassifier(entry.getValue());
			models.put(entry.getKey(), model);
		}
	}

	public String getAlgorithm() {
		return name;
	}
	
	@Override public String getName() {
		return super.getName() + "_" + name;
	}
	
	@Override public String getClassifierString() {
		return super.getClassifierString() + "_" + name;
	}

}
