package experiments.two_part.part_one;

import org.aeonbits.owner.Config.Sources;

import jaicore.experiments.IDatabaseConfig;
import jaicore.ml.experiments.IMultiClassClassificationExperimentConfig;

@Sources({ "file:conf/twopartranker.properties" })
public interface TwoPartExperimenterConfig extends IMultiClassClassificationExperimentConfig, IDatabaseConfig {


}
