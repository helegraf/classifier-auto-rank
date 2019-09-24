package datasets;

public class PerformanceMeasure {

	private String name;
	private PerformanceMeasureType type;
	private boolean maximize;

	public PerformanceMeasure(String name, PerformanceMeasureType type, boolean maximize) {
		super();
		this.name = name;
		this.type = type;
		this.maximize = maximize;
	}

	public String getName() {
		return name;
	}

	public PerformanceMeasureType getType() {
		return type;
	}

	public boolean isMaximize() {
		return maximize;
	}

}
