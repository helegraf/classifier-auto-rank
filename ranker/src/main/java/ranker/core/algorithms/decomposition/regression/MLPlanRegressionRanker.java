package ranker.core.algorithms.decomposition.regression;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import de.upb.crc901.mlplan.core.AbstractMLPlanBuilder;
import de.upb.crc901.mlplan.core.MLPlan;
import de.upb.crc901.mlplan.core.MLPlanWekaRegressionBuilder;
import de.upb.crc901.mlplan.multiclass.wekamlplan.weka.model.MLPipeline;
import jaicore.basic.TimeOut;
import jaicore.basic.algorithm.AlgorithmExecutionCanceledException;
import jaicore.basic.algorithm.exceptions.AlgorithmException;
import jaicore.basic.algorithm.exceptions.AlgorithmTimeoutedException;
import jaicore.ml.WekaUtil;
import ranker.core.algorithms.decomposition.DecompositionRanker;
import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

/**
 * Constructs a single instance of {@link MLPlan} to predict the performance of
 * each algorithm.
 * 
 * @author helegraf
 *
 */
public class MLPlanRegressionRanker extends DecompositionRanker {

	private int seed;
	private int numCPUS;
	private int totalTimeoutSeconds;
	private int evaluationTimeoutSeconds;
	private Object listener;
	private String searchSpace;

	private static final Logger LOGGER = LoggerFactory.getLogger(MLPlanRegressionRanker.class);

	/**
	 * Constructs a new MLPlanRegressionRanker that is using the given parameters
	 * for constructing new MLPlan-instances.
	 * 
	 * @param seed                     the random seed
	 * @param numCPUS                  the number of available cpus
	 * @param totalTimeoutSeconds      the total available timeout in seconds (will
	 *                                 be divided by the number of MLPlan-instances
	 *                                 that need to be built)
	 * @param evaluationTimeoutSeconds the evaluation timeout to be used
	 * @param listener                 a listener object that will be added to each
	 *                                 MLPlan-instance
	 * @param searchSpace              the search space to be used
	 */
	public MLPlanRegressionRanker(int seed, int numCPUS, int totalTimeoutSeconds, int evaluationTimeoutSeconds,
			Object listener, String searchSpace) {
		this.seed = seed;
		this.numCPUS = numCPUS;
		this.totalTimeoutSeconds = totalTimeoutSeconds;
		this.evaluationTimeoutSeconds = evaluationTimeoutSeconds;
		this.listener = listener;
		this.searchSpace = searchSpace;
	}

	@Override
	protected void buildModels(Map<String, Instances> train) throws Exception {
		models = new HashMap<>();

		train.forEach((item, instances) -> {

			/*
			 * initialize ML-Plan with the same config file that has been used to specify
			 * the experiments
			 */
			MLPlanWekaRegressionBuilder builder;
			try {
				builder = AbstractMLPlanBuilder.forWekaRegression();

				switch (searchSpace) {
				case "full":
					builder.withWEKARegressionConfiguration();
					break;
				case "rfClean":
					builder.withRandomForestOnlyAndNoPreprocessorsConfiguration();
					break;
				case "rfPreproc":
					builder.withRandomForestAndPreprocessorsOnlyConfiguration();
					break;
				case "fullnoSMOreg":
					builder.withWEKARegressionConfigurationNoSMO();
					break;
				default:
					throw new IllegalArgumentException(searchSpace + " is not a valid search space identifier!");
				}

				builder.withTimeOut(new TimeOut(totalTimeoutSeconds / train.size(), TimeUnit.SECONDS));
				builder.withNodeEvaluationTimeOut(new TimeOut(evaluationTimeoutSeconds, TimeUnit.SECONDS));
				builder.withCandidateEvaluationTimeOut(new TimeOut(evaluationTimeoutSeconds, TimeUnit.SECONDS));
				builder.withNumCpus(numCPUS);

				MLPlan mlplan = new MLPlan(builder, instances);
				mlplan.setLoggerName("mlplan");
				mlplan.setRandomSeed(seed);
				mlplan.setId("mlplan_" + item);
				if (listener != null) {
					mlplan.registerListener(listener);
				}				
				mlplan.setTimeout(new TimeOut(totalTimeoutSeconds / train.size(), TimeUnit.SECONDS));
				mlplan.setTimeoutPrecautionOffset(1000);

				Classifier optimizedClassifier = mlplan.call();

				models.put(item, optimizedClassifier);
			} catch (IOException | AlgorithmTimeoutedException | AlgorithmException | InterruptedException
					| AlgorithmExecutionCanceledException e) {
				LOGGER.warn("Could not train model for {} due to {}, using random forest.", item, e);

				RandomForest forest = new RandomForest();
				try {
					forest.buildClassifier(instances);
				} catch (Exception e1) {
					LOGGER.error("No model could be built for {} at all. {}", item, e);
				}
				models.put(item, forest);
			}

		});
	}

	public String getSelectedModelString() {
		StringBuilder builder = new StringBuilder();

		this.models.values().forEach(mlplan -> {
			if (mlplan instanceof MLPlan) {
				Classifier selectedClassifier = ((MLPlan) mlplan).getSelectedClassifier();

				if (selectedClassifier instanceof MLPipeline) {
					MLPipeline mlpipeline = (MLPipeline) selectedClassifier;

					builder.append("pl: [");
					builder.append("pre: [");
					builder.append(mlpipeline.getPreprocessors().toString());
					builder.append("] class: [");
					builder.append(WekaUtil.getClassifierDescriptor(mlpipeline.getBaseClassifier()));
					builder.append("], ");
				} else {
					builder.append("class: [");
					builder.append(WekaUtil.getClassifierDescriptor(selectedClassifier));
					builder.append("], ");
				}
			} else {
				builder.append("class: [");
				builder.append(WekaUtil.getClassifierDescriptor(mlplan));
				builder.append("], ");
			}

		});

		return builder.toString();
	}

	@Override
	public String getName() {
		return super.getName() + "_" + seed + "_" + numCPUS + "_" + totalTimeoutSeconds + "_" + evaluationTimeoutSeconds
				+ "_" + searchSpace;
	}

	@Override
	public String getClassifierString() {
		return getSelectedModelString();
	}

}
