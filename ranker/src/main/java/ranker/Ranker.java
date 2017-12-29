package ranker;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;

public class Ranker {

	private final RankingMode rankingMode;
	// classifier / regression model
	private Map<Classifier,Classifier> regressionAlgorithms;

	public Ranker(RankingMode rankingMode) {
		this.rankingMode = rankingMode;
		// TODO Initialize with pre-computed values: all WEKA classifiers + openML data sets
	}
	
	public void buildRanker (Map<Classifier,Instances> train) {
		// TODO Train regression / preference algorithm
		regressionAlgorithms = new HashMap<Classifier,Classifier>();
		train.forEach((classifier,trainSet)->{
			RandomForest forest = new RandomForest();
			try {
				forest.buildClassifier(trainSet);
				regressionAlgorithms.put(classifier, forest);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		});
	}
	
	public List<Classifier> rank (Instance instance) throws Exception {
		TreeMap<Double,Classifier> predictions = new TreeMap<Double,Classifier>();
		ArrayList<Classifier> results = new ArrayList<Classifier>();
		
		for (Classifier classifier : regressionAlgorithms.keySet()) {
			double result = regressionAlgorithms.get(classifier).classifyInstance(instance);
			predictions.put(result, classifier);
		}
		predictions.descendingKeySet().forEach(value->results.add(predictions.get(value)));
		return results;
	}
	
	public RankingMode getRankingMode() {
		return rankingMode;
	}
}
