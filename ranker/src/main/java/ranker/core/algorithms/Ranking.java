package ranker.core.algorithms;

public class Ranking {
	
	private int[] objectList;
	private int[] compareOperators;

	public Ranking(int[] objectList, int[] compareOperators) {
		this.objectList = objectList;
		this.compareOperators = compareOperators;
	}

	public static final int COMPARABLE_ENCODING = 0;

	public int[] getObjectList() {
		return objectList;
	}

}
