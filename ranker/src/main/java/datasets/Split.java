package datasets;

public class Split {

	private Instance instance;
	private int splitId;

	public Split(Instance instance, int splitId) {
		super();
		this.instance = instance;
		this.splitId = splitId;
	}

	public Instance getInstance() {
		return instance;
	}

	public int getSplitId() {
		return splitId;
	}

}
