package experiments.two_part.part_two.execution;

import org.aeonbits.owner.Config;

public interface RankerConfig extends Config {

	/**
	 * Parameter that determined whether hyperparameters shall be optimized for this
	 * ranker.
	 * 
	 * @return whether hyperparmeters shall be optimized
	 */
	@Key("ranker.optimize_hyperparameters")
	@DefaultValue("false")
	public boolean optimizeHyperparameters();

	@Key("ranker.optimize_json_file")
	public String getOptimizationJSONFileLocationRelativeToConfigLocation();
}
