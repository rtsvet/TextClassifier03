package textclassifier03;

import java.io.BufferedReader;
import java.io.FileReader;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.meta.FilteredClassifier;
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

        // Read Train Data
        Instances dataTrain = new Instances(new BufferedReader(new FileReader(DATA_TRAIN_FILE)));
        dataTrain.setClassIndex(dataTrain.numAttributes() - 1);

        // Read Test Data
        Instances dataTest = new Instances(new BufferedReader(new FileReader(DATA_TRAIN_FILE)));
        dataTest.setClassIndex(dataTest.numAttributes() - 1);

        StringToWordVector str2Vec = new StringToWordVector();
        // Set Filter Parameters
        //str2Vec.setOptions(new String[] {"-N 1" });
        str2Vec.setLowerCaseTokens(true);
        str2Vec.setInputFormat(dataTrain);
        // Instances fiteredTrainData = Filter.useFilter(dataTrain, str2Vec);

        J48 j48Class = new J48();
        j48Class.setUnpruned(true);

        // meta-classifier
        FilteredClassifier fc = new FilteredClassifier();
        fc.setFilter(str2Vec);
        fc.setClassifier(j48Class);
        // train 
        fc.buildClassifier(dataTrain);

        // make predictions
        for (int i = 0; i < dataTest.numInstances(); i++) {
            double pred = fc.classifyInstance(dataTest.instance(i));
            System.out.print("ID: " + dataTest.instance(i).value(0));
            System.out.print(", actual: " + dataTest.classAttribute().value((int) dataTest.instance(i).classValue()));
            System.out.println(", predicted: " + dataTest.classAttribute().value((int) pred));
        }


        /*

        EM clusterer = new EM();
        clusterer.setNumClusters(3);
        // set further options for EM, if necessary...
        clusterer.buildClusterer(fiteredTrainData);
        System.out.println(fiteredTrainData);

        ClusterEvaluation eval = new ClusterEvaluation();
        eval.setClusterer(clusterer);
        eval.evaluateClusterer(dataTrain);

        // evaluate classifier and print some statistics
        System.out.println(eval.clusterResultsToString());
        */ 
    }
}
