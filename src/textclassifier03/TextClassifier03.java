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

    private static final String DATAFILE = "/export/Development/DataMining/Weka/weka-3-6-9/data/iris.2D.arff";

    /*
     * java -cp %WEKA_HOME% 
     weka.classifiers.meta.FilteredClassifier 
     -t ReutersAcq-train.arff 
     -T ReutersAcq-test.arff 
     -W "weka.classifiers.functions.SMO -N 2" 
     -F "weka.filters.unsupervised.attribute.StringToWordVector -S"
     */
    public static void main(final String[] args) throws Exception {
        System.out.println("Running");

        final Classifier classifier = new SMO();

        // Read Data
        Instances data = new Instances(new BufferedReader(new FileReader(DATAFILE)));
        data.setClassIndex(data.numAttributes() - 1);

        System.out.println(data);

        Remove filter = new Remove();
        filter.setAttributeIndices("" + (data.classIndex() + 1));
        filter.setInputFormat(data);
        Instances dataClusterer = Filter.useFilter(data, filter);

        EM clusterer = new EM();
        clusterer.setNumClusters(3);
        // set further options for EM, if necessary...
        clusterer.buildClusterer(dataClusterer);
        System.out.println(dataClusterer);

        ClusterEvaluation eval = new ClusterEvaluation();
        eval.setClusterer(clusterer);
        eval.evaluateClusterer(data);

        // evaluate classifier and print some statistics
        System.out.println(eval.clusterResultsToString());
    }
}
