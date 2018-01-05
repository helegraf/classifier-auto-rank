package ranker;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public abstract class RegressionRanker extends Ranker {

	/**
	 * Maps that contains a regression model for each classifier (classifier->regressionmodel)
	 */
	Map<Classifier,Classifier> regressionModels;
	HashMap<Classifier,Instances> map;
	
	/**
	 * Builds a regression model for each classifier. Has to initialize the list of regressionModels
	 * 
	 * @param train
	 * @throws Exception 
	 */
	abstract void buildRegressionModels (Map<Classifier,Instances> train) throws Exception;

	@Override
	public void buildRanker(Instances data) throws Exception {
		getClassifiersAndMetaFeatures(data);
		map = new HashMap<Classifier,Instances>();
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
				map.get(classifiersMap.get(i)).add(new DenseInstance(newFeatureValues.length, newFeatureValues));
			}
		}
		buildRegressionModels(map);
	}

	@Override
	public List<Classifier> predictRankingforInstance(Instance instance) throws Exception {
		TreeMap<Double,List<Classifier>> predictions = new TreeMap<Double,List<Classifier>>();
		ArrayList<Classifier> results = new ArrayList<Classifier>();
		
		
		double[] newFeatures = new double [features.size()+1];
		for (int i : features) {
			newFeatures[i] = instance.value(i);
		}
		newFeatures[newFeatures.length-1] = Double.NaN;
		
		for (Classifier classifier : regressionModels.keySet()) {
			Instance newInstance = new DenseInstance(newFeatures.length,newFeatures);
			newInstance.setDataset(map.get(classifier));
			double result = regressionModels.get(classifier).classifyInstance(newInstance);
			if (predictions.containsKey(result)) {
				predictions.get(result).add(classifier);
			} else {
				ArrayList<Classifier>  classifiers = new ArrayList<Classifier>();
				classifiers.add(classifier);
				predictions.put(result, classifiers);
			}
		}
		predictions.forEach((value,classifierList)->results.addAll(classifierList));
		return results;
	}
}
