package com.byeautumn.wb.dl;

import com.byeautumn.wb.output.ILabelClass;
import org.apache.commons.io.FileUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * Created by qiangao on 5/26/2017.
 */
public class MaskableLSTMDataSetIterator  implements DataSetIterator {
    private final Logger log = LoggerFactory.getLogger(this.getClass());
    private File sourceDir;
    private List<String> fileNames;
    private int miniBatchSize;
//    private int numTimeStepPerSequence;
    private int numFeatures;
    private int nextFileIdx = 0;
    private int maxNumTimeSteps;
    private DataSetPreProcessor preProcessor;
    private ILabelClass labelClass;

    public MaskableLSTMDataSetIterator(String sourceDirName, List<String> fileNames, int miniBatchSize, ILabelClass labelClass)
    {
        if(null == sourceDirName || sourceDirName.isEmpty() || null == fileNames || fileNames.isEmpty() || miniBatchSize < 1 || null == labelClass)
        {
            log.error("Invalid input(s).");
            return;
        }
        this.sourceDir = new File(sourceDirName);
        if(!this.sourceDir.exists())
        {
            log.error("The given source directory doesn't exist: " + sourceDirName);
            return;
        }

        this.labelClass = labelClass;
        this.miniBatchSize = miniBatchSize;

        String[] fileNameArr = this.sourceDir.list();
        if(fileNameArr.length < 1)
        {
            log.error("The given source directory is empty: " + sourceDirName);
            return;
        }

        this.fileNames = fileNames;

        String sampleFileName = this.fileNames.get((this.fileNames.size() - 1) / 2);
        this.numFeatures = detectNumFeatures(sourceDirName, sampleFileName);
        this.maxNumTimeSteps = detectMaxNumTimeSteps(sourceDirName, this.fileNames);
    }

    @Override
    public DataSet next(int batchSize) {
        if(!hasNext())
        {
            throw new NoSuchElementException();
        }
        int currBatchSize = Math.min(batchSize, totalExamples() - cursor());
        //Allocate space:
        //Note the order here:
        // dimension 0 = currBatchSize
        // dimension 1 = number of features
        // dimension 2 = number of time steps
        //Why 'f' order here? See http://deeplearning4j.org/usingrnns.html#data section "Alternative: Implementing a custom DataSetIterator"
        INDArray features = Nd4j.create(new int[]{currBatchSize,this.numFeatures,maxNumTimeSteps}, 'f');
        INDArray labels = Nd4j.create(new int[]{currBatchSize,this.labelClass.getNumLabels(),maxNumTimeSteps}, 'f');

        //The number of time steps could vary and the features tensor is build by using max number of time steps, so features mask is required.
        //Mask arrays contain 1 if data is present at that time step for that example, or 0 if data is just padding
        INDArray featuresMask = Nd4j.zeros(currBatchSize, maxNumTimeSteps, 'f');
        INDArray labelsMask = Nd4j.zeros(currBatchSize, maxNumTimeSteps, 'f');
//        System.out.println("FeaturesMask rank: " + featuresMask.rank());
        for (int ii = 0; ii < currBatchSize; ++ii) //File iteration
        {
            String fileName = this.fileNames.get(ii);
            List<double[]> data = getDataFromSourceFile(new File(this.sourceDir, fileName));
            int numTimeSteps = data.size();
            for(int jj = 0; jj < numTimeSteps; ++jj) //Line (time step) iteration
            {
                double[] values = data.get(jj);
                int labelIdx = values.length - 1;
                for(int kk = 0; kk < values.length - 1; ++kk) //Feature iteration (skip the last value which is label
                {
                    features.putScalar(new int[]{ii, kk, jj}, values[kk]);

                }
                labels.putScalar(new int[]{ii, (int)values[labelIdx], jj}, 1.0);

                featuresMask.putScalar(new int[]{ii, jj}, 1.0);
                if(jj == numTimeSteps - 1)
                    labelsMask.putScalar(new int[]{ii, jj}, 1.0); //Only last time step label counts.
            }

        }

        //Very important: update this.nextFileIdx (cursor)
        this.nextFileIdx += currBatchSize;

//        System.out.println("+++++++++++++++++++ Feature Tensor: ++++++++++++++++++++");
//        System.out.println(print3DINDArray(features));
//        System.out.println("+++++++++++++++++++ End Feature Tensor ++++++++++++++++++++");
//
//        System.out.println("+++++++++++++++++++ Label Tensor: ++++++++++++++++++++");
//        System.out.println(print3DINDArray(labels));
//        System.out.println("+++++++++++++++++++ End Label Tensor ++++++++++++++++++++");

        DataSet ret =  new DataSet(features, labels, featuresMask, labelsMask);

        //The preProcessor could be a normalizer or such.
        if(this.preProcessor != null) {
            this.preProcessor.preProcess(ret);
        }

        return ret;
    }

    @Override
    public int totalExamples() {
        return this.fileNames.size();
    }

    @Override
    public int inputColumns() {
        return this.numFeatures;
    }

    @Override
    public int totalOutcomes() {
        return this.labelClass.getNumLabels();
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return false;
    }

    @Override
    public void reset() {
        this.nextFileIdx = 0;
        Collections.shuffle(this.fileNames);
    }

    @Override
    public int batch() {
        return this.miniBatchSize;
    }

    @Override
    public int cursor() {
        return this.nextFileIdx;
    }

    @Override
    public int numExamples() {
        return totalExamples();
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {
        this.preProcessor = dataSetPreProcessor;
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return this.preProcessor;
    }

    @Override
    public List<String> getLabels() {
        return this.labelClass.getLabels();
    }

    @Override
    public boolean hasNext() {
        return this.nextFileIdx < totalExamples();
    }

    @Override
    public DataSet next() {
        return next(this.miniBatchSize);
    }

    private int detectNumFeatures(String sourceDirName, String csvSampleFileName)
    {
        if(null == sourceDirName || null == csvSampleFileName)
        {
            log.error("Invalid input.");
            return -1;
        }

        File csvSampleFile = new File(sourceDirName, csvSampleFileName);
        if(!csvSampleFile.exists())
        {
            log.error("The csv sample file doesn't exist: " + csvSampleFile.getAbsolutePath());
            return -1;
        }

        List<String> sampleLines = null;
        try {
            sampleLines = FileUtils.readLines(csvSampleFile);
        } catch (IOException e) {
            e.printStackTrace();
        }
        if(null == sampleLines || sampleLines.isEmpty())
        {
            log.error("The sample csv file is empty:  " + csvSampleFileName);
            return -1;
        }
        //Pick the middle line...
        String sampleLine = sampleLines.get(sampleLines.size() / 2);
        String[] sampleValues = sampleLine.split(",");
        if(null == sampleValues || sampleValues.length < 2)
        {
            log.error("The sample line format of the sample csv file seems invalid:  " + sampleLine);
            return -1;
        }

        int numFeatures = sampleValues.length - 1;
        log.info("The detected numFeatures is: " + numFeatures);

        return numFeatures;
    }

    private int detectMaxNumTimeSteps(String sourceDirName, List<String> fileNames)
    {
        if(null == sourceDirName || null == fileNames || fileNames.isEmpty())
        {
            log.error("Invalid input.");
            return -1;
        }

        int max = 0;
        for(String fileName : fileNames)
        {
            File f = new File(sourceDirName, fileName);
            if(!f.exists())
            {
                log.info("The given file doesn't exist: " + f.getAbsolutePath() + " Skipped!!!");
                continue;
            }

            List<double[]> data = getDataFromSourceFile(f);
            max = Math.max(max, data.size());
        }

        return max;
    }

    private List<double[]> getDataFromSourceFile(File sourceFile)
    {
        if(!sourceFile.exists())
        {
            log.error("The source file doesn't exist: " + sourceFile.getAbsolutePath());
            return Collections.emptyList();
        }

        try
        {
            List<String> lines = FileUtils.readLines(sourceFile);
            if(lines.isEmpty())
            {
                log.error("The source file is empty: " + sourceFile.getAbsolutePath());
                return Collections.emptyList();
            }
            List<double[]> retList = new ArrayList<>(lines.size());
            int numDataPerLine = 0;
            for(String line : lines)
            {
                String[] strValues = line.trim().split(",");
                if(null == strValues || strValues.length < 1)
                    continue;
                if(strValues.length < 2)
                {
                    log.debug("Skipped!!! The given line has data less than 2: " + line);
                    continue;
                }
                if(0 == numDataPerLine)
                    numDataPerLine = strValues.length;
                else
                {
                    if(numDataPerLine != strValues.length)
                    {
                        log.error("The number of data per line is NOT consistent: " + numDataPerLine + ":" + strValues.length);
                        return Collections.emptyList();
                    }
                }
                double[] values = new double[numDataPerLine];
                int idx = 0;
                for(String strValue : strValues)
                {
                    double value = Double.parseDouble(strValue);
                    values[idx] = value;
                    ++idx;
                }
                retList.add(values);
            }

            return retList;

        } catch (IOException ioe)
        {
            log.error(ioe.getMessage());
        }
        return Collections.emptyList();
    }

}
