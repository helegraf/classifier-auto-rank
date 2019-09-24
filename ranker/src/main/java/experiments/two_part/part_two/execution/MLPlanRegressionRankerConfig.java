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
	
	@Key("db.upload_intermediate_results")
	public boolean uploadIntermediateResults();
	
	@Key("db.host")
	public String getHost();
	
	@Key("db.user")
	public String getUser();
	
	@Key("db.pw")
	public String getPassword();
	
	@Key("db.db")
	public String getDatabase();
	
	@Key("db.intermediate_results_table")
	public String getIntermediateResultsTable();
	
	@Key("db.experiment_id")
	public int getExperimentId();
}
