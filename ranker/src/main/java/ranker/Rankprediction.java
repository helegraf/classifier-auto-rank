package ranker;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.openml.webapplication.fantail.dc.Characterizer;

import ranker.core.algorithms.Ranker;
import ranker.core.algorithms.regression.RandomForestRanker;
import ranker.core.metafeatures.NoProbingCharacterizer;
import weka.classifiers.Classifier;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * Used to predict a ranking of classifiers for a new data set according to a
 * given ranker and set of meta features.
 * 
 * @author Helena Graf
 *
 */
public class Rankprediction {

	/**
	 * Instances holding the meta data
	 */
	protected Instances metaData;
	protected List<Integer> targetAttributes;
	protected Characterizer characterizer;
	protected Ranker ranker;

	/**
	 * @param metaData
	 * @param targetAttributes
	 * @param characterizer
	 * @param ranker
	 * @throws Exception
	 */
	public Rankprediction(Instances metaData, List<Integer> targetAttributes, Characterizer characterizer,
			Ranker ranker) throws Exception {
		this.metaData = metaData;
		this.targetAttributes = targetAttributes;
		this.characterizer = characterizer;
		this.ranker = ranker;

		ranker.buildRanker(metaData, targetAttributes);
	}

	/**
	 * Predicts rankings of classifiers with a RandomForestRanker and uses a
	 * standard set of meta features (without probing) as a basis.
	 * 
	 * @throws Exception
	 */
	public Rankprediction() throws Exception {
		ClassLoader classLoader = Thread.currentThread().getContextClassLoader();
		InputStream inputStream = classLoader.getResourceAsStream(Util.META_DATA_SMALL_DATA_SETS_COMPUTED_NO_PROBING);
		DataSource source = new DataSource(inputStream);
		metaData = source.getDataSet();
		metaData.deleteAttributeAt(0);

		targetAttributes = new ArrayList<Integer>();
		for (int i = 58; i < 80; i++) {
			targetAttributes.add(i);
		}

		characterizer = new NoProbingCharacterizer();

		this.ranker = new RandomForestRanker();

		ranker.buildRanker(metaData, targetAttributes);

	}

	/**
	 * Predicts a ranking of classifiers for the given data set.
	 * 
	 * @param instances
	 * @return
	 * @throws Exception
	 */
	public List<Classifier> predictRanking(Instances instances) throws Exception {
		// Get meta features for instance
		Map<String, Double> metaFeatures = characterizer.characterize(instances);

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
