package experiments.two_part.part_two.execution;

public interface MLPlanRegressionRankerConfig extends RankerConfig {

	@Key("mlplan.seed")
	public int getSeed();
	
	@Key("mlplan.numCPUs")
	public int getNumCPUs();
	
	@Key("mlplan.totalTimeoutSeconds")
	public int getTotalTimeoutSeconds();
	
	@Key("mlplan.evaluationTimeoutSeconds")
	public int getEvaluationTimeoutSeconds();
	
	@Key("mlplan.searchSpace")
	public String getSearchSpace();
	
	@Key("mlplan.db.upload_intermediate_results")
	public boolean uploadIntermediateResults();
	
	@Key("mlplan.db.host")
	public String getHost();
	
	@Key("mlplan.db.user")
	public String getUser();
	
	@Key("mlplan.db.pw")
	public String getPassword();
	
	@Key("mlplan.db.db")
	public String getDatabase();
	
	@Key("mlplan.db.intermediate_results_table")
	public String getIntermediateResultsTable();
	
	@Key("mlplan.db.experiment_id")
	public int getExperimentId();
}
