package com.byeautumn.wb.dl;

import com.byeautumn.wb.data.OHLCUtils;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by qiangao on 5/24/2017.
 */
public class CustomizedLSTMRunner {

    private int detectNumFeaturesFromTrainingData(RunnerConfigFileReader configReader)
    {
        String trainInputDirName = configReader.getProperty("trainInputDirName");
        File trainInputDir = new File(trainInputDirName);
        //Get the feature number by reading the one of the training csv file.
        String[] inputCSVFiles = trainInputDir.list();
        if(null == inputCSVFiles || inputCSVFiles.length < 1)
        {
            System.err.println("There is NO training input csv files in " + trainInputDir);
            return -1;
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
            return -1;
        }

        List<String> sampleLines = null;
        try {
            sampleLines = FileUtils.readLines(new File(trainInputDir, csvSampleFileName));
        } catch (IOException e) {
            e.printStackTrace();
        }
        if(null == sampleLines || sampleLines.isEmpty())
        {
            System.err.println("The sample csv file is empty:  " + csvSampleFileName);
            return -1;
        }
        //Pick the middle line...
        String sampleLine = sampleLines.get(sampleLines.size() / 2);
        String[] sampleValues = sampleLine.split(",");
        if(null == sampleValues || sampleValues.length < 2)
        {
            System.err.println("The sample line format of the sample csv file seems invalid:  " + sampleLine);
            return -1;
        }

        int numFeatures = sampleValues.length - 1;
        System.out.println("The detected numFeatures is: " + numFeatures);

        return numFeatures;
    }

    public MultiLayerNetwork buildNetworkModel(RunnerConfigFileReader configReader)
    {
        int numLabelClasses = Integer.parseInt(configReader.getProperty("numLabelClasses"));
        int numFeatures = detectNumFeaturesFromTrainingData(configReader);
        int neuralSizeMultiplyer = 2;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)    //Random number generator seed for improved repeatability. Optional.
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .learningRate(0.005)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)  //Not always required, but helps with this data set
                .gradientNormalizationThreshold(0.5)
                .list()
                .layer(0, new GravesLSTM.Builder().activation(Activation.TANH).nIn(numFeatures).nOut(numFeatures*neuralSizeMultiplyer).build())
                .layer(1, new GravesLSTM.Builder().activation(Activation.TANH).nIn(numFeatures*neuralSizeMultiplyer).nOut(numFeatures*neuralSizeMultiplyer).build())
                .layer(2, new GravesLSTM.Builder().activation(Activation.TANH).nIn(numFeatures*neuralSizeMultiplyer).nOut(numFeatures*neuralSizeMultiplyer).build())
                .layer(3, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(numFeatures*neuralSizeMultiplyer).nOut(numLabelClasses).build())
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

    private MultiLayerNetwork loadNetworkModel(RunnerConfigFileReader configReader)
    {
        String networkSaveLocation = configReader.getProperty("networkSaveLocation");
        File networkSaveDir = new File(networkSaveLocation);
        if(!networkSaveDir.exists())
        {
            System.err.println("The given network model save directory doesn't exist.");
            return null;
        }

        String[] savedModelFileNames = networkSaveDir.list(new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                if(name.endsWith(".zip"))
                    return true;
                return false;
            }
        });

        if(savedModelFileNames.length < 1)
        {
            System.err.println("There is NO valid network model exist under directory: " + networkSaveLocation);
            return null;
        }

        //Default is to load the last one (latest one hopefully).
        String networkModelFileName = savedModelFileNames[savedModelFileNames.length - 1];

        File modelFile = new File(networkSaveLocation, networkModelFileName);
        if(!modelFile.exists())
        {
            System.err.println("The saved network model file doesn't exist: " + modelFile.getAbsolutePath());
            return null;
        }

        //Load the model
        MultiLayerNetwork net = null;
        try {
            net = ModelSerializer.restoreMultiLayerNetwork(modelFile);
        } catch (IOException ioe)
        {
            System.err.println(ioe.getStackTrace());
            return null;
        }

        System.out.println("The saved network model has been successfully loaded: " + modelFile.getAbsolutePath());

        return net;
    }


    public void predict(RunnerConfigFileReader configReader)
    {
        MultiLayerNetwork net = loadNetworkModel(configReader);

        int numFeatures = detectNumFeaturesFromTrainingData(configReader);
        int numSequencePerGeneratedFile = Integer.parseInt(configReader.getProperty("numSequencePerGeneratedFile"));
        int numLabelClasses = Integer.parseInt((configReader.getProperty("numLabelClasses")));
        String predictInputDirName = configReader.getProperty("predictInputDirName");
        File rawDataDir = new File(predictInputDirName);
        if(!rawDataDir.exists()){
            System.err.println("The raw data directory doesn't exist: " + predictInputDirName);
            return;
        }

        String[] predictFileNames = rawDataDir.list();
        // clear current stance from the last example
        net.rnnClearPreviousState();

        // put the first caracter into the rrn as an initialisation
        List<double[]> predictData = loadPredictionData(rawDataDir.getAbsolutePath() + "/" + predictFileNames[predictFileNames.length - 1], numSequencePerGeneratedFile);


        INDArray testInit = Nd4j.zeros(numFeatures);
        for(int idx = 0; idx < numFeatures; ++idx)
            testInit.putScalar(idx, predictData.get(0)[idx]);

        // run one step -> IMPORTANT: rnnTimeStep() must be called, not
        // output()
        // the output shows what the net thinks what should come next
        INDArray output = net.rnnTimeStep(testInit);

        StringBuffer sb = new StringBuffer("---------------- Prediction Result Distribution -----------------\n");
        // now the net should guess LEARNSTRING.length mor characters
        for (int jj = 1; jj < predictData.size(); jj++) {

            // first process the last output of the network to a concrete
            // neuron, the neuron with the highest output cas the highest
            // cance to get chosen
            double[] outputProbDistribution = new double[numLabelClasses];
            for (int k = 0; k < outputProbDistribution.length; k++) {
                outputProbDistribution[k] = output.getDouble(k);
            }
            sb.append(printDoubleArray(outputProbDistribution)).append("\n");

            // use the last output as input
            INDArray nextInput = Nd4j.zeros(numFeatures);
            for(int idx = 0; idx < numFeatures; ++idx)
                nextInput.putScalar(idx, predictData.get(jj)[idx]);

            output = net.rnnTimeStep(nextInput);

        }
        sb.append("---------------- Prediction Result Distribution -----------------");
        System.out.println(sb.toString());
    }

    private List<double[]> loadPredictionData(String predictionFileName, int numSequencePerGeneratedFile)
    {
        List<double[]> retList = new ArrayList<>(numSequencePerGeneratedFile+1);
        Path path = Paths.get(predictionFileName);
        try
        {
            List<String> strList = Files.readAllLines(path, Charset.defaultCharset());
            for(String line : strList)
            {
                String[] strDoubleArr = line.split(",");
                double[] dArr = new double[strDoubleArr.length];
                int idx = 0;
                for(String strDouble : strDoubleArr)
                {
                    dArr[idx] = Double.parseDouble(strDouble);
                    ++idx;
                }
                retList.add(dArr);
            }
        } catch (IOException e)
        {
            e.printStackTrace();
        }

        return retList;
    }

    private String printDoubleArray(double[] dArr)
    {
        StringBuffer sb = new StringBuffer();
        for(double d : dArr)
            sb.append(d).append(",");
        return sb.toString();
    }
    public static void main( String[] args ) throws Exception {
        RunnerConfigFileReader configReader = new RunnerConfigFileReader("../../WindBell/WindBell/src/com/byeautumn/wb/dl/CustomizedLSTMRunner.properties");
        System.out.println(configReader.printSelf());

        CustomizedLSTMRunner runner = new CustomizedLSTMRunner();
        runner.predict(configReader);
//        runTrainAndValidation(configReader);
//        runLoadAndPredict(configReader);

    }
}
