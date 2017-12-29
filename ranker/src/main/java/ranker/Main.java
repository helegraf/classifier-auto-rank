package ranker;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.math3.stat.correlation.KendallsCorrelation;
import org.openml.apiconnector.io.OpenmlConnector;
import org.openml.apiconnector.settings.Settings;
import org.openml.apiconnector.xml.DataQualityList;
import org.openml.apiconnector.xml.DataSetDescription;
import org.openml.webapplication.attributeCharacterization.AttributeCharacterizer;
import org.openml.webapplication.fantail.dc.Characterizer;
import org.openml.webapplication.fantail.dc.landmarking.GenericLandmarker;
import org.openml.webapplication.fantail.dc.statistical.AttributeEntropy;
import org.openml.webapplication.fantail.dc.statistical.Cardinality;
import org.openml.webapplication.fantail.dc.statistical.NominalAttDistinctValues;
import org.openml.webapplication.fantail.dc.statistical.SimpleMetaFeatures;
import org.openml.webapplication.fantail.dc.statistical.Statistical;
import org.openml.webapplication.features.GlobalMetafeatures;

import de.upb.cs.is.jpl.api.dataset.IInstance;
import de.upb.cs.is.jpl.api.dataset.defaultdataset.DefaultDataset;
import de.upb.cs.is.jpl.api.dataset.defaultdataset.DefaultInstance;
import de.upb.cs.is.jpl.api.dataset.defaultdataset.absolute.DefaultAbsoluteDataset;
import de.upb.cs.is.jpl.api.math.linearalgebra.DenseDoubleVector;
import de.upb.cs.is.jpl.api.math.linearalgebra.IVector;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class Main {

	public static void main(String[] args) throws Exception {
//		OpenmlConnector client = new OpenmlConnector();
//		String[] dataQualities = client.dataQualitiesList().getQualities();
//		System.out.println(dataQualities.length);
//		
//		DefaultDataset<IVector> dataset = new DefaultAbsoluteDataset();
//		IVector ratings = new DenseDoubleVector(2, 5.0);
//		dataset.addInstance(new DefaultInstance<IVector>(2, ratings, dataset));
//		IInstance<?, ?, ?> instance = dataset.getInstance(0);
//		Instance instance = Util.getQualities(5);
//		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
//		for (int i = 0; i < Util.dataQualities.length; i++) {
//			attributes.add(new Attribute(Util.dataQualities[i]));
//		}
//		attributes.add(new Attribute("Performance"));
//		Instances results = new Instances("PerformanceMeasures", attributes, 0);
//		// TODO set prediction
//		results.add(instance);
		
		
//		Util.getAllDataQualities();
//		ArrayList<Integer> holdout = new ArrayList<Integer>();
//		holdout.add(7);
//		Util.makeInstances(holdout);
//		Ranker ranker = new Ranker(RankingMode.REGRESSION);
//		ranker.buildRanker(Util.classifierPerformances);
//		List<Classifier> ranking = ranker.rank(Util.testSet.firstInstance());
//		ranking.forEach(classifier->System.out.println(classifier.getClass().getName()));
		
		GlobalMetafeatures allFeatures = new GlobalMetafeatures(null);
		List<Characterizer> characterizers = allFeatures.getCharacterizers();
		List<String> expectedIds = allFeatures.getExpectedIds();
		int expectedQualities = allFeatures.getExpectedQualities();
		
		
	}

}
