package com.byeautumn.wb.dl;

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
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

/**
 * Created by qiangao on 5/16/2017.
 */
public class BasicLSTMRunner {
    private static final Logger log = LoggerFactory.getLogger(BasicLSTMRunner.class);
    private DataSetIterator trainData;
    private DataSetIterator testData;

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
//        Updater updater = Updater.ADAGRAD;
//        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//                .seed(12345)
//                .regularization(true).l2(0.001) //l2 regularization on all layers
//                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//                .iterations(1)
//                .learningRate(0.04)
//                .list()
//                .layer(4, new GravesLSTM.Builder()
//                        .activation(Activation.SOFTSIGN)
//                        .nIn(50)
//                        .nOut(50)
//                        .weightInit(WeightInit.XAVIER)
//                        .updater(updater)
//                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
//                        .gradientNormalizationThreshold(10)
//                        .learningRate(0.008)
//                        .build())
//                .layer(5, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
//                        .activation(Activation.SOFTMAX)
//                        .nIn(50)
//                        .nOut(4)    //4 possible shapes: circle, square, arc, line
//                        .updater(updater)
//                        .weightInit(WeightInit.XAVIER)
//                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
//                        .gradientNormalizationThreshold(10)
//                        .build())
//                .pretrain(false).backprop(true)
//                .backpropType(BackpropType.TruncatedBPTT)
//                .tBPTTForwardLength(tbpttLength)
//                .tBPTTBackwardLength(tbpttLength)
//                .build();
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)    //Random number generator seed for improved repeatability. Optional.
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .learningRate(0.005)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)  //Not always required, but helps with this data set
                .gradientNormalizationThreshold(0.5)
                .list()
                .layer(0, new GravesLSTM.Builder().activation(Activation.TANH).nIn(numInput).nOut(numInput).build())
                .layer(1, new GravesLSTM.Builder().activation(Activation.TANH).nIn(numInput).nOut(numInput).build())
                .layer(2, new GravesLSTM.Builder().activation(Activation.TANH).nIn(numInput).nOut(numInput).build())
                .layer(3, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(numInput).nOut(numLabelClasses).build())
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

    public void trainAndValidate(int numEpochs)
    {
        MultiLayerNetwork net = buildNetwork(numFeatures, numLabelClasses);

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

    private void buildTrainAndTestDataset(String rawDataDirName, double trainDataPercentage, int miniBatchSize)
    {
        int startIdx = 7528;
        int endIdx = 16887;
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
        File rawDataDir = new File(rawDataDirName);
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

    public static void main( String[] args ) throws Exception {
        String inputDirName = "../../WindBell/WindBell/resources/training/BasicLSTMData/SPX";
        int numLabelClasses = 7;
        int numFeatures = 5;
        int numEpochs = 5;
        BasicLSTMRunner runner = new BasicLSTMRunner(inputDirName, numLabelClasses, numFeatures);

        runner.trainAndValidate(numEpochs);
    }
}
