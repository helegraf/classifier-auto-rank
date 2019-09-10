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

	private String prefix = "models/";
	private String suffix = "";

	/**
	 * Constructs a new ranker from saved models, which are located by prefix + item
	 * + suffix, where item is the label of the algorithms to be ranked. E.g. if the
	 * prefix is "models/", an algorithm "weka.classifiers.RandomForest", and the
	 * suffix ".txt", the searched location for a model for this algorithms will be
	 * "models/weka.classifiers.RandomForest.txt".
	 * 
	 * @param prefix string that is prepended to a label to search for its model
	 *               (default: models/)
	 * @param suffix string that is appended to a label to search for its model
	 *               (default: empty)
	 */
	public SavedModelRanker(String prefix, String suffix) {
		this.prefix = prefix;
		this.suffix = suffix;
	}

	private Logger logger = LoggerFactory.getLogger(SavedModelRanker.class);

	@Override
	protected void buildModels(Map<String, Instances> train) throws Exception {
		models = new HashMap<String, Classifier>();

		train.forEach((item, dataset) -> {
			Classifier regressionModel;
			try {
				regressionModel = (Classifier) SerializationHelper.read(prefix + item + suffix);
			} catch (Exception e) {
				logger.warn("Could not initialize model {} due to {}, try using RandomForest instead", item, e);
				regressionModel = new RandomForest();
				try {
					regressionModel.buildClassifier(dataset);
				} catch (Exception e1) {
					logger.error("No classifier could be trained for {}", item);
				}
			}
			models.put(item, regressionModel);
		});

	}
}
