package com.byeautumn.wb.dl;

import com.byeautumn.wb.output.BasicLSTMDataGenerator;
import org.apache.commons.io.FileUtils;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.examples.recurrent.seqclassification.UCISequenceClassificationExample;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.*;

/**
 * Created by qiangao on 5/16/2017.
 */
public class BasicLSTMRunner {
    private static final Logger log = LoggerFactory.getLogger(BasicLSTMRunner.class);

    private MultiLayerNetwork net;
    private DataSetIterator trainData;
    private DataSetIterator testData;
    private DataSetIterator predictData;

    private int numLabelClasses = -1;
    private String rawDataDirName;
    private int numFeatures = -1;

    public BasicLSTMRunner(String rawDataDirName, int numLabelClasses, int numFeatures)
    {
        if(numLabelClasses < 1)
        {
            log.error("Invaid input(s).");
            return;
        }

        this.numLabelClasses = numLabelClasses;
        this.rawDataDirName = rawDataDirName;
        this.numFeatures = numFeatures;
    }
    private MultiLayerNetwork buildNetwork(int numInput, int numLabelClasses)
    {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)    //Random number generator seed for improved repeatability. Optional.
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .learningRate(0.005)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)  //Not always required, but helps with this data set
                .gradientNormalizationThreshold(0.5)
                .list()
                .layer(0, new GravesLSTM.Builder().activation(Activation.TANH).nIn(numInput).nOut(numInput*2).build())
                .layer(1, new GravesLSTM.Builder().activation(Activation.TANH).nIn(numInput*2).nOut(numInput*2).build())
                .layer(2, new GravesLSTM.Builder().activation(Activation.TANH).nIn(numInput*2).nOut(numInput*2).build())
                .layer(3, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(numInput*2).nOut(numLabelClasses).build())
                .pretrain(false).backprop(true).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(10));

        System.out.println("Number of parameters in network: " + net.numParams());
        for( int i=0; i<net.getnLayers(); i++ ){
            System.out.println("Layer " + i + " nParams = " + net.getLayer(i).numParams());
        }
        return net;
    }

    public void saveNetwork(File modelFile, boolean bSaveUpdater)
    {
        //Where to save the network. Note: the file is in .zip format - can be opened externally
        //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
        try {
            ModelSerializer.writeModel(net, modelFile, bSaveUpdater);
        } catch (IOException ioe)
        {
            System.err.println(ioe.getStackTrace());
        }
    }

    public void saveNetwork(File modelFile)
    {
        saveNetwork(modelFile, true);
    }

    public void loadNetwork(String modelFileName)
    {
        try {
            //Load the model
            net = ModelSerializer.restoreMultiLayerNetwork(modelFileName);
        } catch (IOException ioe)
        {
            System.err.println(ioe.getStackTrace());
        }
    }

    public void trainAndValidate(int numEpochs, boolean bForceRebuildNet)
    {
        if(null == net || bForceRebuildNet)
            net = buildNetwork(numFeatures, numLabelClasses);

        int miniBatchSize = 14;
        buildTrainAndTestDataset(rawDataDirName, 0.9, miniBatchSize);

        String str = "Test set evaluation at epoch %d: Accuracy = %.2f, F1 = %.2f";
        for (int i = 0; i < numEpochs; i++) {
            net.fit(trainData);

            //Evaluate on the test set:
            Evaluation evaluation = net.evaluate(testData);
            log.info(String.format(str, i, evaluation.accuracy(), evaluation.f1()));
            System.out.println(String.format(str, i, evaluation.accuracy(), evaluation.f1()));

            testData.reset();
            trainData.reset();
        }

        log.info("----- BasicLSTMRunner Complete -----");
        System.out.println("----- BasicLSTMRunner Complete -----");
    }

    public void trainAndValidate(int numEpochs)
    {
        trainAndValidate(numEpochs, false);
    }

    private void buildTrainAndTestDataset(String rawDataDirName, double trainDataPercentage, int miniBatchSize)
    {
        File rawDataDir = new File(rawDataDirName);
        if(!rawDataDir.exists()){
            System.err.println("The raw data directory doesn't exist: " + rawDataDirName);
            return;
        }

        int startIdx = 0;
        int endIdx = rawDataDir.list().length - 1;
        int length = endIdx - startIdx + 1;
        int testStartIdx = (int) Math.round(length * trainDataPercentage);

        log.info("Training indices from %d to %d.", startIdx, testStartIdx - 1);
        log.info("Test indices from %d to %d.", testStartIdx, endIdx);

        if(startIdx > endIdx || length < 1 || testStartIdx < 1 || testStartIdx > endIdx)
        {
            log.error("Wrong indexing calculation and buildTrainAndTestDataset function stopped.");
            return;
        }
        SequenceRecordReader trainFeatures = new CSVSequenceRecordReader();
        SequenceRecordReader testFeatures = new CSVSequenceRecordReader();
        try {
            trainFeatures.initialize(new NumberedFileInputSplit(rawDataDir.getAbsolutePath() + "/%d.csv", startIdx, testStartIdx - 1));
            testFeatures.initialize(new NumberedFileInputSplit(rawDataDir.getAbsolutePath() + "/%d.csv", testStartIdx, endIdx));
        } catch (IOException ioe)
        {
            log.error(ioe.getMessage());
            return;
        } catch (InterruptedException ie)
        {
            log.error(ie.getMessage());
            return;
        }

        trainData = new SequenceRecordReaderDataSetIterator(trainFeatures, miniBatchSize, numLabelClasses, numFeatures, false);
        testData = new SequenceRecordReaderDataSetIterator(testFeatures, miniBatchSize, numLabelClasses, numFeatures, false);

        //Normalization. Is it needed?
        //Normalize the training data
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainData);              //Collect training data statistics
        trainData.reset();

        //Use previously collected statistics to normalize on-the-fly. Each DataSet returned by 'trainData' iterator will be normalized
        trainData.setPreProcessor(normalizer);
        testData.setPreProcessor(normalizer);   //Note that we are using the exact same normalization process as the training data
    }

    public void buildPredictionDataset(String rawDataDirName, int miniBatchSize)
    {
        File rawDataDir = new File(rawDataDirName);
        if(!rawDataDir.exists()){
            System.err.println("The raw data directory doesn't exist: " + rawDataDirName);
            return;
        }

        int startIdx = 0;
        int endIdx = rawDataDir.list().length - 1;

        SequenceRecordReader predictFeatures = new CSVSequenceRecordReader();
        try {
            predictFeatures.initialize(new NumberedFileInputSplit(rawDataDir.getAbsolutePath() + "/%d.csv", startIdx, endIdx));

        } catch (IOException ioe)
        {
            log.error(ioe.getMessage());
            return;
        } catch (InterruptedException ie)
        {
            log.error(ie.getMessage());
            return;
        }

        predictData = new SequenceRecordReaderDataSetIterator(predictFeatures, miniBatchSize, numLabelClasses, numFeatures, false);

        //Normalization. Is it needed?
        //Normalize the predictData data
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(predictData);              //Collect prediction data statistics
        predictData.reset();

        //Use previously collected statistics to normalize on-the-fly. Each DataSet returned by 'predictData' iterator will be normalized
        predictData.setPreProcessor(normalizer);
    }

    public void predict(String inputDataDirName, int miniBaichSize)
    {
        //Assuming here that the full predict data set doesn't fit in memory -> load 10 examples at a time
        Map<Integer, String> labelMap = new HashMap<>();
        labelMap.put(0, " down 4% or more");
        labelMap.put(1, " down between 1.5% and 4%");
        labelMap.put(2, " down between 0.5% and 1.5%");
        labelMap.put(3, " sideways between -0.5% and +0.3%");
        labelMap.put(4, " up between 0.3% and 1%");
        labelMap.put(5, " up between 1% and 2%");
        labelMap.put(6, " up 2% or more");

        Evaluation evaluation = new Evaluation(labelMap);

        buildPredictionDataset(inputDataDirName, miniBaichSize);
        while(predictData.hasNext()) {
            DataSet dsPredict = predictData.next();
            INDArray predicted = net.output(dsPredict.getFeatureMatrix(), false);
            INDArray actual = dsPredict.getLabels();
            evaluation.evalTimeSeries(actual, predicted);
        }

        System.out.println(evaluation.stats());
    }

    public static void main( String[] args ) throws Exception {
        RunnerConfigFileReader configReader = new RunnerConfigFileReader("../../WindBell/WindBell/src/com/byeautumn/wb/dl/BasicLSTMRunner.properties");
        System.out.println(configReader.printSelf());

        //Generate Training Data...
        boolean bForceRegenerateTrainingData = Boolean.parseBoolean(configReader.getProperty("forceRegenerateTrainingData"));
        if(bForceRegenerateTrainingData) {
            String symbol = configReader.getProperty("symbol");
            String rawDataSourceDirName = configReader.getProperty("rawDataSourceDir");
            File rawDataSourceDir = new File(rawDataSourceDirName);
            if (!rawDataSourceDir.exists()) {
                System.err.println("The raw data source directory doesn't exist: " + rawDataSourceDirName);
                return;
            }

            String[] rawSourceFileNames = rawSourceFileNames = rawDataSourceDir.list();
            System.out.println("The number of raw source files: " + rawSourceFileNames.length);
            for (String rawFileName : rawSourceFileNames)
                System.out.println(rawFileName);

            String mainSourceFileName = configReader.getProperty("mainSourceFileName");
            List<String> sourceFileNames = new ArrayList<>();
            //Make sure add the main source file first.
            sourceFileNames.add(mainSourceFileName);

            for (String rawFileName : rawSourceFileNames) {
                if (mainSourceFileName.equals(rawFileName))
                    continue;
                sourceFileNames.add(rawFileName);
                System.out.println("The support file will be loaded: " + rawFileName);
            }

            int numSequencePerGeneratedFile = Integer.parseInt(configReader.getProperty("numSequencePerGeneratedFile"));
            BasicLSTMDataGenerator.generateLSTMTrainingData2(symbol, sourceFileNames, numSequencePerGeneratedFile);
        }

        String inputDirName = configReader.getProperty("inputDirName");
        File trainInputDir = new File(inputDirName);
        if(!trainInputDir.exists())
        {
            System.err.println("The training input directory doesn't exist: " + trainInputDir);
            return;
        }

        //Get the feature number by reading the one of the training csv file.
        String[] inputCSVFiles = trainInputDir.list();
        if(null == inputCSVFiles || inputCSVFiles.length < 1)
        {
            System.err.println("There is NO training input csv files in " + trainInputDir);
            return;
        }
        String csvSampleFileName = null;
        for(String csv : inputCSVFiles)
        {
            if(csv.endsWith(".csv"))
                csvSampleFileName = csv;
        }

        if(null == csvSampleFileName)
        {
            System.err.println("There is NO training input csv files in " + trainInputDir);
            return;
        }

        List<String> sampleLines = FileUtils.readLines(new File(trainInputDir, csvSampleFileName));
        if(null == sampleLines || sampleLines.isEmpty())
        {
            System.err.println("The sample csv file is empty:  " + csvSampleFileName);
            return;
        }
        //Pick the middle line...
        String sampleLine = sampleLines.get(sampleLines.size() / 2);
        String[] sampleValues = sampleLine.split(",");
        if(null == sampleValues || sampleValues.length < 2)
        {
            System.err.println("The sample line format of the sample csv file seems invalid:  " + sampleLine);
            return;
        }

        int numFeatures = sampleValues.length - 1;
        System.out.println("The detected numFeatures is: " + numFeatures);

        int numLabelClasses = Integer.parseInt(configReader.getProperty("numLabelClasses"));
        int numEpochs = Integer.parseInt(configReader.getProperty("numEpochs"));
        BasicLSTMRunner runner = new BasicLSTMRunner(inputDirName, numLabelClasses, numFeatures);

        runner.trainAndValidate(numEpochs);

        boolean bSaveModel = Boolean.parseBoolean(configReader.getProperty("saveModel"));
        if(bSaveModel)
        {
            String networkSaveLocation = configReader.getProperty("networkSaveLocation");
            Date now = new Date();
            runner.saveNetwork(new File(networkSaveLocation, String.format("BasicLSTMRunner_%s.zip", new SimpleDateFormat("yyyyMMdd-hhmm").format(now))));
        }

    }
}
