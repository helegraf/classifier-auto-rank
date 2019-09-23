package experiments.two_part.part_two.execution;

import java.sql.SQLException;
import java.util.HashMap;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.eventbus.Subscribe;

import de.upb.crc901.mlplan.core.events.ClassifierFoundEvent;
import de.upb.crc901.mlplan.multiclass.wekamlplan.weka.model.MLPipeline;
import jaicore.basic.SQLAdapter;
import jaicore.ml.WekaUtil;
import ranker.core.algorithms.Ranker;
import ranker.core.algorithms.decomposition.regression.MLPlanRegressionRanker;

public class MLPlanRegressionRankerExecutor extends RankerExecutor {

	private Logger logger = LoggerFactory.getLogger(MLPlanRegressionRankerExecutor.class);
	private SQLAdapter adapter;
	private String intermediateResultsTable;
	private int experimentId;

	@Override
	protected Class<? extends RankerConfig> getRankerConfigClass() {
		return MLPlanRegressionRankerConfig.class;
	}

	@Override
	protected Ranker instantiate(RankerConfig configuration) {
		MLPlanRegressionRankerConfig config = (MLPlanRegressionRankerConfig) configuration;

		try {
			if (config.uploadIntermediateResults()) {
				adapter = new SQLAdapter(config.getHost(), config.getUser(), config.getPassword(),
						config.getDatabase());
				intermediateResultsTable = config.getIntermediateResultsTable();
				experimentId = config.getExperimentId();
				createIntermediateResultsTableIfNotExists(intermediateResultsTable);
			}
		} catch (SQLException e) {
			logger.warn("Will not upload intermediate results due to {}", e);
			adapter = null;
		}

		return new MLPlanRegressionRanker(config.getSeed(), config.getNumCPUs(), config.getTotalTimeoutSeconds(),
				config.getEvaluationTimeoutSeconds(), this, config.getSearchSpace());
	}

	private void createIntermediateResultsTableIfNotExists(String table) throws SQLException {
		String sql = String.format("CREATE TABLE IF NOT EXISTS `%s` (\r\n"
				+ " `evaluation_id` int(10) NOT NULL AUTO_INCREMENT,\r\n" + " `experiment_id` int(8) NOT NULL,\r\n"
				+ " `preprocessor` text COLLATE utf8_bin NOT NULL,\r\n"
				+ " `classifier` text COLLATE utf8_bin NOT NULL,\r\n" + " `rmse` double DEFAULT NULL,\r\n"
				+ " `time_train` bigint(8) DEFAULT NULL,\r\n" + " `time_predict` int(8) DEFAULT NULL,\r\n"
				+ " `evaluation_timestamp_finish` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,\r\n"
				+ " `exception` text COLLATE utf8_bin,\r\n" + " PRIMARY KEY (`evaluation_id`)\r\n"
				+ ") ENGINE=InnoDB AUTO_INCREMENT=8281430 DEFAULT CHARSET=utf8 COLLATE=utf8_bin", table);

		adapter.update(sql);
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
				eval.put("experiment_id", experimentId);
				eval.put("preprocessor", preprocessor);
				eval.put("classifier", classifier);
				if (!Double.isNaN(e.getScore())) {
					eval.put("rmse", e.getScore());
				} else {
					logger.warn("Uploading incomplete intermediate solution!");
				}

				eval.put("time_train", e.getTimestamp());
				adapter.insert(intermediateResultsTable, eval);
			} catch (Exception e1) {
				logger.error("Could not store hasco solution in database", e1);
			}
		} else {
			logger.error("no adapter!");
		}
	}
}