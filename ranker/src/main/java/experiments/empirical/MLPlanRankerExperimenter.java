package experiments.empirical;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import org.aeonbits.owner.ConfigCache;
import org.aeonbits.owner.ConfigFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.eventbus.Subscribe;

import de.upb.crc901.mlplan.core.events.ClassifierFoundEvent;
import de.upb.crc901.mlplan.multiclass.wekamlplan.weka.model.MLPipeline;
import jaicore.basic.SQLAdapter;
import jaicore.experiments.ExperimentDBEntry;
import jaicore.experiments.ExperimentRunner;
import jaicore.experiments.IExperimentDatabaseHandle;
import jaicore.experiments.IExperimentIntermediateResultProcessor;
import jaicore.experiments.IExperimentSetEvaluator;
import jaicore.experiments.databasehandle.ExperimenterSQLHandle;
import jaicore.experiments.exceptions.ExperimentDBInteractionFailedException;
import jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import jaicore.experiments.exceptions.IllegalExperimentSetupException;
import jaicore.ml.WekaUtil;
import ranker.core.algorithms.decomposition.regression.MLPlanRegressionRanker;
import ranker.core.evaluation.EvaluationHelper;
import ranker.core.evaluation.strategies.MCCV;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

public class MLPlanRankerExperimenter implements IExperimentSetEvaluator {
	private static final String CLASSIFIER_FIELD = "classifier";
	private static final String PREPROCESSOR_FIELD = "preprocessor";
	private int experimentID;

	private static final Logger L = LoggerFactory.getLogger(MLPlanRankerExperimenter.class);

	private static MLPlanRankerExperimenterConfig experimentConfig = ConfigCache
			.getOrCreate(MLPlanRankerExperimenterConfig.class);
	private SQLAdapter adapter;

	public MLPlanRankerExperimenter(final File configFile) {
		super();
		Properties props = new Properties();
		try {
			props.load(new FileInputStream(configFile));
		} catch (IOException e) {
			L.error("Could not find or access config file {}", configFile, e);
			System.exit(1);
		}
		experimentConfig = ConfigFactory.create(MLPlanRankerExperimenterConfig.class, props);
		if (experimentConfig.evaluationsTable() == null) {
			throw new IllegalArgumentException("No evaluations table (db.evalTable) set in config");
		}
	}

	@Override
	public void evaluate(final ExperimentDBEntry experimentEntry,
			final IExperimentIntermediateResultProcessor processor) throws ExperimentEvaluationFailedException {
		try {
			this.experimentID = experimentEntry.getId();
			this.adapter = new SQLAdapter(experimentConfig.getDBHost(), experimentConfig.getDBUsername(),
					experimentConfig.getDBPassword(), experimentConfig.getDBDatabaseName());
			Map<String, String> experimentValues = experimentEntry.getExperiment().getValuesOfKeyFields();

			L.error("Evaluate Ranker");
			// Load data
			BufferedReader reader = new BufferedReader(new FileReader("resources/complete.arff"));
			ArffReader arff = new ArffReader(reader);
			Instances data = arff.getData();		
			
			// Init ranker
			MLPlanRegressionRanker ranker = new MLPlanRegressionRanker(Integer.parseInt(experimentValues.get("seed")),
					experimentEntry.getExperiment().getNumCPUs(), Integer.parseInt(experimentValues.get("timeout"))/5,
					Integer.parseInt(experimentValues.get("evaluationTimeout")),this, "full");

			List<Integer> targetAttributes = new ArrayList<>();
			for (int j = 104; j < 126; j++) {
				targetAttributes.add(j);
			}
			
			// Evalate
			List<Double> result = EvaluationHelper.evaluateRanker(new MCCV(5,.7,experimentValues.get("seed")),ranker, data, targetAttributes);
			L.info("result: {}", result);		
			
			Map<String, Object> results = new HashMap<>();
			// kendall, rmse, loss, b3l
			results.put("kendall", result.get(0));
			results.put("rmse", result.get(1));
			results.put("loss", result.get(2));
			results.put("b3l", result.get(3));
			results.put(CLASSIFIER_FIELD, ranker.getSelectedModelString());
			processor.processResults(results);
			L.info("Experiment done.");
		} catch (Exception e) {
			throw new ExperimentEvaluationFailedException(e);
		}
	}
	
	@Subscribe
	public void rcvHASCOSolutionEvent(final ClassifierFoundEvent e) {
		if (adapter != null) {
			try {
				String classifier = "";
				String preprocessor = "";
				if (e.getSolutionCandidate() instanceof MLPipeline) {
					MLPipeline solution = (MLPipeline) e.getSolutionCandidate();
					preprocessor = solution.getPreprocessors().isEmpty() ? ""
							: solution.getPreprocessors().get(0).toString();
					classifier = WekaUtil.getClassifierDescriptor(solution.getBaseClassifier());
				} else {
					classifier = WekaUtil.getClassifierDescriptor(e.getSolutionCandidate());
				}
				Map<String, Object> eval = new HashMap<>();
				eval.put("experiment_id", experimentID);
				eval.put(PREPROCESSOR_FIELD, preprocessor);
				eval.put(CLASSIFIER_FIELD, classifier);
				eval.put("rmse", e.getScore());
				eval.put("time_train", e.getTimestamp());
				adapter.insert(experimentConfig.evaluationsTable(), eval);
			} catch (Exception e1) {
				L.error("Could not store hasco solution in database", e1);
			}
		} else {
			L.error("no adapter!");
		}
	}

	public static void main(String[] args)
			throws ExperimentDBInteractionFailedException, IllegalExperimentSetupException {
		File configFile = new File("conf/mlplan-ranker.properties");
		IExperimentDatabaseHandle dbHandle = new ExperimenterSQLHandle(experimentConfig);
		ExperimentRunner runner = new ExperimentRunner(experimentConfig, new MLPlanRankerExperimenter(configFile),
				dbHandle);
		runner.randomlyConductExperiments(1, false);
	}
}
