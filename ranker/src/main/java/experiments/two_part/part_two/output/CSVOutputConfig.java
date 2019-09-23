package experiments.two_part.part_two.output;

import org.aeonbits.owner.Config;

public interface CSVOutputConfig extends Config {
	
	@Key("outfile")
	@DefaultValue("output.csv")
	public String getOutFilePath();
	
	@Key("column.experiment")
	@DefaultValue("experiment_id")
	public String getExperimentIdColumn();
	
	@Key("column.datasetId")
	@DefaultValue("dataset_id")
	public String getDatasetIdColumn();
	
	@Key("column.trueRanking")
	@DefaultValue("true_ranking")
	public String getTrueRankingColumn();
	
	@Key("column.trueValues")
	@DefaultValue("true_values")
	public String getTrueValueColumn();
	
	@Key("column.predictedRanking")
	@DefaultValue("predicted_ranking")
	public String getPredictedRankingColumn();
	
	@Key("column.predictedValues")
	@DefaultValue("predicted_values")
	public String getPredictedValuesColumn();
	
	@Key("column.trainingTime")
	@DefaultValue("training_time")
	public String getTrainingTimeColumn();
	
	@Key("column.predictTime")
	@DefaultValue("predict_time")
	public String getPredictTimeColumn();
}
