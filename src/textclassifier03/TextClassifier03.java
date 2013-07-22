package textclassifier03;

import java.io.BufferedReader;
import java.io.FileReader;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.J48;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.EM;
import weka.core.Attribute;
import weka.core.Debug;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class TextClassifier03 {

    private static final String DATA_TRAIN_FILE = "/export/Development/DataMining/Weka/weka-3-6-9/data/ReutersCorn-train.arff";
    private static final String DATA_TEST_FILE = "/export/Development/DataMining/Weka/weka-3-6-9/data/ReutersCorn-test.arff";

    /*
     * 
     */
    public static void main(final String[] args) throws Exception {
        System.out.println("Running");

        final Classifier classifier = new SMO();

        // Read Data
        Instances dataTrain = new Instances(new BufferedReader(new FileReader(DATA_TRAIN_FILE)));
        dataTrain.setClassIndex(dataTrain.numAttributes() - 1);

        System.out.println(dataTrain);

        Remove filter = new Remove();
        filter.setAttributeIndices("" + (dataTrain.classIndex() + 1));
        filter.setInputFormat(dataTrain);
        Instances dataClusterer = Filter.useFilter(dataTrain, filter);

        EM clusterer = new EM();
        clusterer.setNumClusters(3);
        // set further options for EM, if necessary...
        clusterer.buildClusterer(dataClusterer);
        System.out.println(dataClusterer);

        ClusterEvaluation eval = new ClusterEvaluation();
        eval.setClusterer(clusterer);
        eval.evaluateClusterer(dataTrain);

        // evaluate classifier and print some statistics
        System.out.println(eval.clusterResultsToString());
    }
}
