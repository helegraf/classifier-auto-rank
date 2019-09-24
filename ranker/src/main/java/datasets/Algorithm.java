package datasets;

public class Algorithm {

	private String id;
	private boolean deterministic;

	public Algorithm(String id, boolean deterministic) {
		super();
		this.id = id;
		this.deterministic = deterministic;
	}

	public String getId() {
		return id;
	}

	public boolean isDeterministic() {
		return deterministic;
	}

}
