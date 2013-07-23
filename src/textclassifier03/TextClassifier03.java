package textclassifier03;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.NavigableMap;
import java.util.SortedMap;
import java.util.TreeMap;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.EM;
import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.Debug;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.stemmers.SnowballStemmer;
import weka.core.stemmers.Stemmer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.gui.beans.ClassifierPerformanceEvaluator;

public class TextClassifier03 {

    private static final String DATA_TRAIN_FILE = "/export/Development/DataMining/Weka/weka-3-6-9/data/ReutersCorn-train.arff";
    private static final String DATA_TEST_FILE = "/export/Development/DataMining/Weka/weka-3-6-9/data/ReutersCorn-test.arff";


    /*
     * 
     */
    public static void main(final String[] args) throws Exception {
        System.out.println("Running Classifier");

        // Read Train Data
        Instances dataTrain = new Instances(new BufferedReader(new FileReader(DATA_TRAIN_FILE)));
        dataTrain.setClassIndex(dataTrain.numAttributes() - 1);

        // Read Test Data
        Instances dataTest = new Instances(new BufferedReader(new FileReader(DATA_TRAIN_FILE)));
        dataTest.setClassIndex(dataTest.numAttributes() - 1);

        StringToWordVector str2Vec = new StringToWordVector();
        // Set Filter Parameters
        str2Vec.setOptions(new String[]{"-N 1"});
        str2Vec.setLowerCaseTokens(true);
        str2Vec.setIDFTransform(true);
        str2Vec.setStemmer(new SnowballStemmer());
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

        // evaluate classifier and print some statistics
        Evaluation eval = new Evaluation(dataTrain);
        eval.evaluateModel(fc, dataTest);
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));

        System.out.println("Running Clusterer");

        Remove filter = new Remove();
        filter.setAttributeIndices("" + (dataTrain.classIndex() + 1));
        filter.setInputFormat(dataTrain);
        Instances classRemovedData = Filter.useFilter(dataTrain, filter);

        str2Vec.setInputFormat(classRemovedData);
        Instances fiteredTrainData = Filter.useFilter(classRemovedData, str2Vec);

        //EM clusterer = new EM();
        SimpleKMeans clusterer = new SimpleKMeans();

        clusterer.setNumClusters(2);

        // set further options for EM, if necessary...
        clusterer.buildClusterer(fiteredTrainData);

        System.out.println("Clusters:");

        // For EM clusterer
        //double[] clPrios = clusterer.clusterPriors();
        //for (int i = 0; i < clPrios.length; i++) {
        //   System.out.print(i + "=" + clPrios[i]+"; ");
        //}

        // for K-Means
        //NavigableMap<String, Double> centroMap = new TreeMap<String, Double>();
        Map<String, Double> centroMap = new LinkedHashMap<String, Double>();

        Instances centroids = clusterer.getClusterCentroids();
        for (int i = 0; i < centroids.size(); i++) {
            System.out.println("Cluster " + i);
            Instance cen = centroids.get(i);
            for (int j = 0; j < cen.numAttributes(); j++) {
                centroMap.put(cen.attribute(j).name(), cen.value(j));
            }
            centroMap = MapUtil.sortByValue(centroMap);
            int k = 0;
            for (Iterator<Map.Entry<String, Double>> it = centroMap.entrySet().iterator(); it.hasNext();) {
                System.out.println(it.next() +"; ");
                k++;
                if (k > 10) break;
            }
            System.out.println("End ");
            centroMap.clear();
        }

        ClusterEvaluation clEval = new ClusterEvaluation();
        clEval.setClusterer(clusterer);

        str2Vec.setInputFormat(dataTrain);
        Instances strVectorTrainData = Filter.useFilter(dataTrain, str2Vec);
        strVectorTrainData.setClassIndex(0);
        clEval.evaluateClusterer(strVectorTrainData);

        // evaluate classifier and print some statistics
        //System.out.println(clEval.clusterResultsToString());

    }
}
