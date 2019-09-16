package evaluation_test;

import static org.junit.Assert.assertEquals;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.junit.Test;

import ranker.core.evaluation.measures.rank.COFIDCGFunction;
import ranker.core.evaluation.measures.rank.DCGAtK;
import ranker.core.evaluation.measures.rank.NDCGAtK;
import ranker.core.evaluation.measures.rank.DCGAtK.ExponentMode;

public class NDCGAtKTest {

	@Test
	public void testValues() {
		for (int i = 1; i < 23; i++) {
			firstMCorrectTest(i, 0, i);
		}

		for (int i = 1; i < 23; i++) {
			firstMCorrectTest(i, 1, i);
		}

		for (int i = 1; i < 23; i++) {
			firstMCorrectTest(i, 3, i);
		}

		for (int i = 1; i < 23; i++) {
			firstMCorrectTest(i, 5, i);
		}

		for (int i = 1; i < 23; i++) {
			firstMCorrectTest(22, 0, i);
		}

		for (int i = 1; i < 23; i++) {
			firstMCorrectTest(22, 1, i);
		}

		for (int i = 1; i < 23; i++) {
			firstMCorrectTest(22, 3, i);
		}

		for (int i = 1; i < 23; i++) {
			firstMCorrectTest(22, 5, i);
		}

	}

	public void firstMCorrectTest(int n, int m, int c) {
		System.out.println("First " + m + " correct length " + n + ", Cutoff: " + c);

		List<String> perfectRanking = new ArrayList<>();
		List<Double> performanceValues = new ArrayList<>();

		double invertedDCG = 0;
		double correctDCG = 0;
		for (int i = 1; i <= n; i++) {
			if (i <= c) {
				double addToCorrect = (Math.pow(2, (n + 1 - i)) - 1) / (COFIDCGFunction.log2(i + 1));
				correctDCG += addToCorrect;

				if (i <= m) {
					invertedDCG += addToCorrect;
				} else {
					double addToInverse = (Math.pow(2, i - m) - 1) / (COFIDCGFunction.log2(i + 1));
					invertedDCG += addToInverse;
				}
			}

			perfectRanking.add(String.valueOf(i));
			performanceValues.add((double) i);
		}
		Collections.reverse(perfectRanking);

		double ndcg = invertedDCG / correctDCG;

		List<String> predictedRanking = new ArrayList<>();
		predictedRanking.addAll(perfectRanking);
		Collections.reverse(predictedRanking);

		List<String> correctItemsList = new ArrayList<>();

		int maxRemoval = predictedRanking.size() - (m + 1);
		for (int i = predictedRanking.size() - 1; i > maxRemoval; i--) {
			if (!predictedRanking.isEmpty()) {
				String moveElem = predictedRanking.remove(i);
				correctItemsList.add(moveElem);
			} else {
				break;
			}
		}

		correctItemsList.addAll(predictedRanking);
		predictedRanking = correctItemsList;

		System.out.println(predictedRanking + " vs " + perfectRanking);

		DCGAtK dcg = new DCGAtK(c);
		dcg.setExponentMode(ExponentMode.INTEGER);
		NDCGAtK dcgAtK = new NDCGAtK(dcg);
		double result = dcgAtK.evaluate(predictedRanking, perfectRanking, performanceValues, performanceValues);
		System.out.println(
				"Actual: " + (Math.round(result * 1000) / 1000.0) + " Expected: " + (Math.round(ndcg * 1000) / 1000.0));
		assertEquals(ndcg, result, 0.0001);
	}

	public static double valueFor(int n, int m, int c) {
		double invertedDCG = 0;
		double correctDCG = 0;
		for (int i = 1; i <= c; i++) {

			double addToCorrect = (Math.pow(2, (n + 1 - i) / (double) n) - 1) / (COFIDCGFunction.log2(i + 1));
			correctDCG += addToCorrect;

			if (i <= m) {
				invertedDCG += addToCorrect;
			} else {
				double addToInverse = (Math.pow(2, (i - m) / (double) n) - 1) / (COFIDCGFunction.log2(i + 1));
				invertedDCG += addToInverse;
			}

		}

		return invertedDCG / correctDCG;
	}

	public static void writeFiles(String folder) throws IOException {

		try (BufferedWriter writer = new BufferedWriter(new FileWriter(folder))) {
			// writer.write("n m k result");
			writer.newLine();
			for (int i = 1; i < 23; i++) {
				writeExperiment(writer, i, 0, i);
			}

			for (int i = 1; i < 23; i++) {
				writeExperiment(writer, i, 1, i);
			}

			for (int i = 1; i < 23; i++) {
				writeExperiment(writer, i, 3, i);
			}

			for (int i = 1; i < 23; i++) {
				writeExperiment(writer, i, 5, i);
			}

			for (int i = 1; i < 23; i++) {
				writeExperiment(writer, 22, 0, i);
			}

			for (int i = 1; i < 23; i++) {
				writeExperiment(writer, 22, 1, i);
			}

			for (int i = 1; i < 23; i++) {
				writeExperiment(writer, 22, 3, i);
			}

			for (int i = 1; i < 23; i++) {
				writeExperiment(writer, 22, 5, i);
			}
		}

	}

	private static void writeExperiment(BufferedWriter writer, int n, int m, int c) throws IOException {
		writer.write(String.format("%d %d %d %f %n", n, m, c, valueFor(n, m, c)));
	}

	public static void main(String[] args) throws IOException {
		writeFiles("ndcgresults/results.txt");
	}
}
