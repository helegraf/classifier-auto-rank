package ranker;

import java.util.List;
import java.util.Map;

import org.openml.webapplication.fantail.dc.Characterizer;

import ranker.core.algorithms.Ranker;
import weka.classifiers.Classifier;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class Rankprediction {
	
	protected Instances metaData;
	protected List<Integer> targetAttributes;
	protected Characterizer characterizer;
	protected Ranker ranker;
	
	public Rankprediction(Instances metaData, List<Integer> targetAttributes, Characterizer characterizer, Ranker ranker) throws Exception {
		this.metaData = metaData;
		this.targetAttributes = targetAttributes;
		this.characterizer = characterizer;
		this.ranker = ranker;
		
		ranker.buildRanker(metaData, targetAttributes);
	}
	
	public List<Classifier> predictRanking (Instances instances) throws Exception {
		// Get meta features for instance
		Map<String,Double> metaFeatures = characterizer.characterize(instances);
		
		// Insert in the right order in instance
		Instance queryInstance = new DenseInstance(metaData.numAttributes());
		for (int i = 0; i < metaData.numAttributes(); i++) {
			if (!targetAttributes.contains(i)) {
				queryInstance.setValue(i, metaFeatures.get(metaData.attribute(i).name()));
			}
		}
		
		// Predict a ranking
		return ranker.predictRankingforInstance(queryInstance);
	}

}
