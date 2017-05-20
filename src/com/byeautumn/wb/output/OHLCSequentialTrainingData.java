package com.byeautumn.wb.output;

import com.byeautumn.wb.data.OHLCElement;
import com.byeautumn.wb.data.OHLCElementTable;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

/**
 * Created by qiangao on 5/17/2017.
 */
public class OHLCSequentialTrainingData {
    private static final double MILLION_FACTOR = 0.000001;
    private List<OHLCElementTable> ohlcElementTableList;
    private int[] labels;
    private int timeSeriesLength;

    public OHLCSequentialTrainingData(List<OHLCElementTable> ohlcElementTables, int[] labels, int timeSeriesLength)
    {
        if(null == ohlcElementTables || null == labels || labels.length < 1 || timeSeriesLength < labels.length || ohlcElementTables.size() < 1 || ohlcElementTables.get(0).size() < timeSeriesLength)
        {
            System.err.println("Invalid input(s) for OHLCSequentialTrainingData generation.");
            return;
        }
        if(ohlcElementTables.size() > 1)
        {
            int sequenceLength = ohlcElementTables.get(0).size();
            for(OHLCElementTable table : ohlcElementTables)
            {
                if(table.size() != sequenceLength)
                {
                    System.err.println("The sequence length should be consistent. OHLCSequentialTrainingData generation failed.");
                    return;
                }
            }
        }

        this.ohlcElementTableList = ohlcElementTables;
        this.labels = labels;
        this.timeSeriesLength = timeSeriesLength;
    }

    public void toCSVFile(String outputFileName) {
        if (null == ohlcElementTableList) {
            System.err.println("The OHLCSequentialTrainingData instance is empty. CSV file generation failed.");
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
//            int timeSequenceLength = ohlcElementTableList.get(0).size();
            int labelLength = labels.length; // labellength could be smaller than parameterLength
            StringBuffer sb = new StringBuffer();
            for(int idx = 0; idx < this.timeSeriesLength; ++idx)
            {

                for (OHLCElementTable table : ohlcElementTableList)
                {
                    OHLCElement elem = table.getOHCLElementsSortedByDate().get(idx);
                    sb.append(elem.getOpenValue()).append(",");
                    sb.append(elem.getHighValue()).append(",");
                    sb.append(elem.getLowValue()).append(",");
                    sb.append(elem.getCloseValue()).append(",");
                    sb.append(elem.getVolumeValue() * MILLION_FACTOR).append(",");//Volume in million
                }
                if(idx < this.timeSeriesLength - labelLength)
                    sb.append(Integer.MIN_VALUE);
                else
                    sb.append(labels[idx]);
                sb.append(",\n");
            }
            fileWriter.write(sb.toString());

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
}
