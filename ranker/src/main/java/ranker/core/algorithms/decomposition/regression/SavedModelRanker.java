package ranker.core.algorithms.decomposition.regression;

import java.util.HashMap;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ranker.core.algorithms.decomposition.DecompositionRanker;
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
public class SavedModelRanker extends DecompositionRanker {
	
	private String prefix = "models/split_";
	
	public SavedModelRanker(int seed) {
		this.prefix = prefix + seed + "/";
	}

	private static final Logger LOGGER = LoggerFactory.getLogger(SavedModelRanker.class);

	@Override
	protected void buildRegressionModels(Map<String, Instances> train) throws Exception {
		regressionModels = new HashMap<String, Classifier>();
		
		train.forEach((item,dataset)->{
			Classifier regressionModel;
			try {
				regressionModel = (Classifier) SerializationHelper.read(prefix + item + ".txt");
			} catch (Exception e) {
				LOGGER.warn("Could not initialize model {} due to {}, try using RandomForest instead",item,e);
				System.err.println("Not init " + item + " due to " + e);
				regressionModel = new RandomForest();
				try {
					regressionModel.buildClassifier(dataset);
				} catch (Exception e1) {
					LOGGER.error("No classifier could be trained for {}",item);
				}
			}
			regressionModels.put(item, regressionModel);
		});

	}
}
