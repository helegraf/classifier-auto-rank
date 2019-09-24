package datasets;

public class Instance {

	private String id;
	private double[] features;

	public Instance(String id, double[] features) {
		super();
		this.id = id;
		this.features = features;
	}

	public String getId() {
		return id;
	}

	public double[] getFeatures() {
		return features;
	}

}
