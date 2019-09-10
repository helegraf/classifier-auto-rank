package ranker.core.metafeatures;

import java.util.ArrayList;
import java.util.Arrays;

import org.openml.webapplication.fantail.dc.Characterizer;
import org.openml.webapplication.fantail.dc.statistical.Cardinality;
import org.openml.webapplication.fantail.dc.statistical.NominalAttDistinctValues;
import org.openml.webapplication.fantail.dc.statistical.SimpleMetaFeatures;
import org.openml.webapplication.fantail.dc.statistical.Statistical;

/**
 * A Characterizer that applies several characterizers to a data set, but does
 * not use any probing.
 * 
 * @author Helena Graf
 *
 */
public class NoProbingCharacterizer extends GlobalCharacterizer {

	/**
	 * Constructs a new NoProbingCharacterizer. Construction is the same as for the
	 * {@link ranker.core.metafeatures.GlobalCharacterizer}, except that only Characterizers that do not use probing
	 * are initialized.
	 * 
	 * @throws Exception if the characterizer cannot be initialized successfully
	 */
	public NoProbingCharacterizer() throws Exception {
		super();
	}

	@Override
	protected void initializeCharacterizers() {
		Characterizer[] characterizerArray = { new SimpleMetaFeatures(), new Statistical(),
				new NominalAttDistinctValues(), new Cardinality() };

		characterizers = new ArrayList<>(Arrays.asList(characterizerArray));
	}

}