package experiments;

import org.aeonbits.owner.Config.Sources;

import jaicore.experiments.IDatabaseConfig;
import jaicore.ml.experiments.IMultiClassClassificationExperimentConfig;

@Sources({ "file:conf/mlplan-ranker-models.properties" })
public interface RankerModelTrainingExperimenterConfig extends IMultiClassClassificationExperimentConfig, IDatabaseConfig {

	public static final String DB_EVAL_TABLE = "db.evalTable";

	@Key(DB_EVAL_TABLE)
	@DefaultValue("evaluations_mls")
	public String evaluationsTable();
	
	@Key("gui.show")
	@DefaultValue("false")
	public boolean showGUI();

}
