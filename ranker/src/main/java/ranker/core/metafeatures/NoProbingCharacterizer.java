package ranker.core.metafeatures;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import org.apache.commons.lang.time.StopWatch;
import org.openml.webapplication.fantail.dc.Characterizer;
import org.openml.webapplication.fantail.dc.landmarking.GenericLandmarker;
import org.openml.webapplication.fantail.dc.statistical.Cardinality;
import org.openml.webapplication.fantail.dc.statistical.NominalAttDistinctValues;
import org.openml.webapplication.fantail.dc.statistical.SimpleMetaFeatures;
import org.openml.webapplication.fantail.dc.statistical.Statistical;

import weka.core.Instances;
import weka.core.Utils;

public class NoProbingCharacterizer extends Characterizer {

	private String[] ids;
	private List<Characterizer> characterizers;

	private ArrayList<Characterizer> batchCharacterizers;

	private final String preprocessingPrefix = "-E \"weka.attributeSelection.CfsSubsetEval -P 1 -E 1\" -S \"weka.attributeSelection.BestFirst -D 1 -N 5\" -W ";
	private final String cp1NN = "weka.classifiers.lazy.IBk";
	private final String cpNB = "weka.classifiers.bayes.NaiveBayes";
	private final String cpASC = "weka.classifiers.meta.AttributeSelectedClassifier";
	private final String cpDS = "weka.classifiers.trees.DecisionStump";

	private Map<String, Double> computationTimes = new HashMap<String, Double>();

	public NoProbingCharacterizer() throws Exception {
		Characterizer[] characterizers = { new SimpleMetaFeatures(), // done before, but necessary for streams
				new Statistical(), new NominalAttDistinctValues(),
				// goes wrong new AttributeEntropy(),
				//new GenericLandmarker("kNN1N", cp1NN, 2, null), new GenericLandmarker("NaiveBayes", cpNB, 2, null),
				//new GenericLandmarker("DecisionStump", cpDS, 2, null),
				//new GenericLandmarker("CfsSubsetEval_kNN1N", cpASC, 2, Utils.splitOptions(preprocessingPrefix + cp1NN)),
				//new GenericLandmarker("CfsSubsetEval_NaiveBayes", cpASC, 2,
				//		Utils.splitOptions(preprocessingPrefix + cpNB)),
				//new GenericLandmarker("CfsSubsetEval_DecisionStump", cpASC, 2,
				//		Utils.splitOptions(preprocessingPrefix + cpDS)) 
				};
		batchCharacterizers = new ArrayList<>(Arrays.asList(characterizers));
		// additional parameterized batch landmarkers
//		String zeros = "0";
//		for (int i = 1; i <= 3; ++i) {
//			zeros += "0";
//			String[] j48Option = { "-C", "." + zeros + "1" };
//			batchCharacterizers
//					.add(new GenericLandmarker("J48." + zeros + "1.", "weka.classifiers.trees.J48", 2, j48Option));
//
//			String[] repOption = { "-L", "" + i };
//			batchCharacterizers
//					.add(new GenericLandmarker("REPTreeDepth" + i, "weka.classifiers.trees.REPTree", 2, repOption));
//
//			String[] randomtreeOption = { "-depth", "" + i };
//			batchCharacterizers.add(new GenericLandmarker("RandomTreeDepth" + i, "weka.classifiers.trees.RandomTree", 2,
//					randomtreeOption));
//		}
		batchCharacterizers.add(new Cardinality());

		List<String> metaFeatures = new ArrayList<String>();
		for (Characterizer characterizer : batchCharacterizers) {
			for (String metaFeature : characterizer.getIDs()) {
				metaFeatures.add(metaFeature);
			}
		}

		ids = new String[metaFeatures.size()];
		for (int i = 0; i < metaFeatures.size(); i++) {
			ids[i] = metaFeatures.get(i);
		}
	}

	@Override
	public String[] getIDs() {
		return ids;
	}

	@Override
	public Map<String, Double> characterize(Instances instances) {
		TreeMap<String, Double> metaFeatures = new TreeMap<String, Double>();
		batchCharacterizers.forEach(characterizer -> {
			 try {
				 StopWatch watch = new StopWatch();
				 watch.start();
				 metaFeatures.putAll(characterizer.characterize(instances));
				 watch.stop();
				 computationTimes.put(characterizer.toString(), (double) watch.getTime());
			 } catch (Exception e) {
				 // Put NaN as result and computation time
				 for (String metaFeature : characterizer.getIDs()) {
					 metaFeatures.put(metaFeature, Double.NaN);
				 }
				 computationTimes.put(characterizer.toString(), Double.NaN);
			 }
		});
		return metaFeatures;
	}

	public Map<String,Double> getMetaFeatureComputationTimes () {
		return computationTimes;
	}
	
	public List<String> getCharacterizerNames() {
		List<String> result = new ArrayList<String>();
		for (Characterizer chara : batchCharacterizers) {
			result.add(chara.toString());
		}
		return result;
	}
}
