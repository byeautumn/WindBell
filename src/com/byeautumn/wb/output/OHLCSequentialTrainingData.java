package com.byeautumn.wb.output;

import com.byeautumn.wb.data.OHLCElement;
import com.byeautumn.wb.data.OHLCElementTable;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

/**
 * Created by qiangao on 5/17/2017.
 */
public class OHLCSequentialTrainingData {
    private static final double MILLION_FACTOR = 0.000001;
    private List<SequentialFlatRecord> flatData;

    private OHLCSequentialTrainingData() {}
    public OHLCSequentialTrainingData(List<OHLCElementTable> ohlcElementTables)
    {
        if(null == ohlcElementTables || ohlcElementTables.size() < 1)
        {
            System.err.println("Invalid input(s) for OHLCSequentialTrainingData generation.");
            return;
        }

        buildFlatData(ohlcElementTables);
    }

    public static OHLCSequentialTrainingData createInstance(List<SequentialFlatRecord> flatData)
    {
        if(null == flatData)
        {
            System.err.println("The sequence length should be consistent. OHLCSequentialTrainingData generation failed.");
            return null;
        }

        OHLCSequentialTrainingData ret = new OHLCSequentialTrainingData();
        ret.flatData = flatData;

        return ret;
    }

    public List<OHLCSequentialTrainingData> split(int numSequence)
    {
        if(numSequence > this.flatData.size())
        {
            System.err.println();
            return null;
        }

        List<OHLCSequentialTrainingData> trainDataSplits = new ArrayList<>(flatData.size() - numSequence + 1);
        for(int idx = 0; idx <= flatData.size() - numSequence; ++idx)
        {
            List<SequentialFlatRecord> subFlatData = new ArrayList<>(numSequence);
            for(int secIdx = idx; secIdx < idx + numSequence; ++secIdx)
            {
                SequentialFlatRecord flatRecord = this.flatData.get(secIdx).clone();
                subFlatData.add(flatRecord);
            }

            OHLCSequentialTrainingData subTrainData = OHLCSequentialTrainingData.createInstance(subFlatData);
            trainDataSplits.add(subTrainData);
        }

        return trainDataSplits;
    }

    private void buildFlatData(List<OHLCElementTable> tableList)
    {
        if(null == tableList || tableList.isEmpty())
        {
            System.err.println("The OHLCSequentialTrainingData instance is empty. Stop building flat data.");
            return;
        }

        if(null == flatData)
            flatData = new ArrayList<>(tableList.get(0).size());

        //Assume each OHLCElement has 5 values.
        int feartureSizePerOHLCElement = 5;
        int featureSize = tableList.size() * feartureSizePerOHLCElement;
        OHLCElementTable mainTable = tableList.get(0);
        List<OHLCElement> mainElemList = mainTable.getOHCLElementsSortedByDate();
        for (int timeSeriesIdx = 0; timeSeriesIdx < mainElemList.size(); ++timeSeriesIdx) {
            OHLCElement elem = mainElemList.get(timeSeriesIdx);
            Date date = elem.getDateValue();
            double[] flatValues = new double[featureSize];
            if(tableList.size() > 1) {
                boolean bMissingMatch = false;
                //Note: idx starts from 1
                for (int idx = 1; idx < tableList.size(); ++idx) {
                    OHLCElementTable table = tableList.get(idx);
                    OHLCElement matchElem = table.getOHLCElement(date);
                    if (null == matchElem) {
                        bMissingMatch = true;
                        break;
                    }

                    flatValues[feartureSizePerOHLCElement * idx] = matchElem.getOpenValue();
                    flatValues[feartureSizePerOHLCElement * idx + 1] = matchElem.getHighValue();
                    flatValues[feartureSizePerOHLCElement * idx + 2] = matchElem.getLowValue();
                    flatValues[feartureSizePerOHLCElement * idx + 3] = matchElem.getCloseValue();
                    flatValues[feartureSizePerOHLCElement * idx + 4] = matchElem.getVolumeValue();

                }
                if (bMissingMatch)
                    continue;
            }
            flatValues[0] = elem.getOpenValue();
            flatValues[1] = elem.getHighValue();
            flatValues[2] = elem.getLowValue();
            flatValues[3] = elem.getCloseValue();
            flatValues[4] = elem.getVolumeValue();

            int label = Integer.MIN_VALUE;
            if(timeSeriesIdx < mainElemList.size() - 1)
            {
                OHLCElement nextElem = mainElemList.get(timeSeriesIdx + 1);
                label = BasicLSTMLabelingManager.generateLabel_7(elem.getCloseValue(), nextElem.getCloseValue());
            }

            SequentialFlatRecord flatRecord = new SequentialFlatRecord(date, flatValues, label);
            flatData.add(flatRecord);
        }
//        System.out.println("flat data size: " + flatData.size());
    }

    public String printSelfAsCSV()
    {
        return printSelfAsCSV(0, flatData.size() - 1, true);
    }

    public String printSelfAsCSV(int startIdx, int endIdx, boolean bExcludeUnlabeledRecord)
    {
        if(startIdx < 0 || endIdx < startIdx)
        {
            System.err.println("Invalid input(s).");
            return "";
        }

        StringBuffer sb = new StringBuffer();
        for(int idx = startIdx; idx <= endIdx; ++idx)
        {
            SequentialFlatRecord flatRecord = flatData.get(idx);
            if(bExcludeUnlabeledRecord) {
                if (!BasicLSTMLabelingManager.isLabelValid(flatRecord.getLabel()))
                    continue;
            }

            if(bExcludeUnlabeledRecord)
                sb.append(flatRecord.printValuesAndLabelWithDateInfoAsCSV());
            else
                sb.append(flatRecord.printValuesWithDateInfoAsCSV()); //In this case don't include label in the file at all.
            sb.append("\n");
        }
        return sb.toString();
    }

    public void generateTrainingCSVFile(String outputFileName) {
        generateCSVFile(outputFileName, false);
    }

    public void generateCSVFile(String outputFileName, boolean bForPrediction) {
        if (null == flatData) {
            System.err.println("The flat data is empty. CSV file generation failed.");
            return;
        }

        File outputFile = new File(outputFileName);
        if (outputFile.exists()) {
            if (!outputFile.delete()) {
                System.err.println("CSV file " + outputFile.getAbsolutePath() + " cannot be deleted! CSV file generation failed.");
                return;
            }
        }

        FileWriter fileWriter = null;
        try
        {
            fileWriter = new FileWriter(outputFileName);
            if(bForPrediction)
                fileWriter.write(printSelfAsCSV(1, flatData.size() - 1, false));
            else
                fileWriter.write(printSelfAsCSV());

        } catch (IOException ioe)
        {
            ioe.printStackTrace();
        }finally
        {
            try
            {
                fileWriter.flush();
                fileWriter.close();
            }
            catch(IOException ioe2)
            {
                 ioe2.printStackTrace();
            }
        }
    }

    public void generatePredictionCSVFile(String outputFileName)
    {
        generateCSVFile(outputFileName, true);
    }
}
