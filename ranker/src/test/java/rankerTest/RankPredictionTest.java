package rankerTest;

import static org.junit.Assert.fail;

import java.io.BufferedReader;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;

import org.junit.BeforeClass;
import org.junit.Test;
import org.openml.webapplication.fantail.dc.Characterizer;

import ranker.Rankprediction;
import ranker.Util;
import ranker.core.algorithms.PerfectRanker;
import ranker.core.algorithms.Ranker;
import ranker.core.metafeatures.GlobalCharacterizer;
import weka.core.Instances;

public class RankPredictionTest {
	
	static Instances testInstances;
	static List<Integer> targetAttributes;
	static Characterizer characterizer;
	static Ranker ranker;

	@BeforeClass
	public static void loadMetaData() throws Exception {
		BufferedReader reader = Files.newBufferedReader(FileSystems.getDefault().getPath("metaData_small_allPerformanceValues.arff"),
				Util.charset);
		testInstances = new Instances(reader);
		targetAttributes = new ArrayList<Integer>();
		for (int i = 104; i < 131; i++) {
			targetAttributes.add(i);
		}
		characterizer = new GlobalCharacterizer();
		ranker = new PerfectRanker();
	}

	@Test
	public void testCreation() {
		try {
			@SuppressWarnings("unused")
			Rankprediction rankprediction = new Rankprediction(testInstances,targetAttributes,new GlobalCharacterizer(),ranker);
		} catch (Exception e) {
			e.printStackTrace();
			fail("Could not generate a RankPrediction Object");
		}
	}
}
