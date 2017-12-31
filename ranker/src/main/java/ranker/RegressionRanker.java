package ranker;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class RegressionRanker extends Ranker {

	private Map<Classifier,Classifier> regressionAlgorithms;
	
	public void buildRanker (Map<Classifier,Instances> train) throws Exception{
		regressionAlgorithms = new HashMap<Classifier,Classifier>();
		for (Classifier classifier : train.keySet()) {
			RandomForest forest = new RandomForest();
			forest.buildClassifier(train.get(classifier));
			regressionAlgorithms.put(classifier, forest);
		}
	}

	@Override
	public void buildRanker(Instances data) throws Exception {
		getClassifiersAndMetaFeatures(data);
		HashMap<Classifier,Instances> map = new HashMap<Classifier,Instances>();
		ArrayList<Attribute> attributes= new ArrayList<Attribute>();
		for (int i : features) {
			attributes.add(new Attribute(data.attribute(i).name()));
		}
		attributes.add(new Attribute("Performance"));
		for (int i : classifierIndices) {
			map.put(classifiersMap.get(i), new Instances(classifiersMap.get(i).getClass().getName(), attributes, 0));
			map.get(classifiersMap.get(i)).setClassIndex(features.size());
		}
		for (Instance instance : data) {
			double [] featureValues = getFeatureValuesForInstance(instance);
			for (int i : classifierIndices) {
				double[] newFeatureValues = new double[featureValues.length+1];
				for (int j = 0; j < featureValues.length;j++) {
					newFeatureValues[j] = featureValues[j];
				}
				newFeatureValues[featureValues.length] = instance.value(i);
			}
		}
		buildRanker(map);
	}

	@Override
	public List<Classifier> predictRankingforInstance(Instance instance) throws Exception {
		TreeMap<Double,List<Classifier>> predictions = new TreeMap<Double,List<Classifier>>();
		ArrayList<Classifier> results = new ArrayList<Classifier>();
		
		for (Classifier classifier : regressionAlgorithms.keySet()) {
			double result = regressionAlgorithms.get(classifier).classifyInstance(instance);
			if (predictions.containsKey(result)) {
				predictions.get(result).add(classifier);
			} else {
				ArrayList<Classifier>  classifiers = new ArrayList<Classifier>();
				predictions.put(result, classifiers);
			}
		}
		predictions.descendingKeySet().forEach(value->results.addAll(predictions.get(value)));
		return results;
	}
}
