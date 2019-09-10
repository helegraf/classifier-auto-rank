package ranker.core.algorithms;

public class Ranking {
	
	public static final int COMPARABLE_ENCODING = 0;
	private int[] objectList;

	public Ranking(int[] objectList) {
		this.objectList = objectList;
	}

	public int[] getObjectList() {
		return objectList;
	}
}