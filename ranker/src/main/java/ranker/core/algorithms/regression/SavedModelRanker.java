package ranker.core.algorithms.regression;

import java.util.HashMap;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.SerializationHelper;

/**
 * A regression ranker that is initialized from saved models.
 * 
 * @author Helena Graf
 *
 */
public class SavedModelRanker extends RegressionRanker {
	
	private String prefix = "models/split_";
	
	public SavedModelRanker(int seed) {
		this.prefix = prefix + seed + "/";
	}

	private static final Logger LOGGER = LoggerFactory.getLogger(SavedModelRanker.class);

	@Override
	protected void buildRegressionModels(Map<Classifier, Instances> train) throws Exception {
		regressionModels = new HashMap<Classifier, Classifier>();
		
		train.forEach((classifier,dataset)->{
			Classifier regressionModel;
			try {
				regressionModel = (Classifier) SerializationHelper.read(prefix + classifier.getClass().getName() + ".txt");
			} catch (Exception e) {
				LOGGER.warn("Could not initialize model {} due to {}, try using RandomForest instead",classifier.getClass().getName(),e);
				System.err.println("Not init " + classifier.getClass().getName() + " due to " + e);
				regressionModel = new RandomForest();
				try {
					regressionModel.buildClassifier(dataset);
				} catch (Exception e1) {
					LOGGER.error("No classifier could be trained for {}",classifier.getClass().getName());
				}
			}
			regressionModels.put(classifier, regressionModel);
		});

	}
}
