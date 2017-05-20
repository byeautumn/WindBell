package com.byeautumn.wb.output;

import com.byeautumn.wb.data.OHLCElement;
import com.byeautumn.wb.data.OHLCElementTable;
import com.byeautumn.wb.data.OHLCUtils;
import com.byeautumn.wb.process.SeriesDataProcessUtil;

import java.io.File;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by qiangao on 5/17/2017.
 */
public class BasicLSTMDataGenerator {

    private static SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd");
    public static void generateLSTMTrainingData(String symbol, String sourceFile, int numOfExamplesPerSequence)
    {
        OHLCElementTable bigTable = OHLCUtils.readOHLCDataSourceFile(sourceFile);
        System.out.println("Big table size: " + bigTable.size());

        //NOte: here ere pass numOfExamplesPerSequence + 1 because for labeling we need 1 more time spot.
        List<OHLCElementTable> pieceList = SeriesDataProcessUtil.splitOHLCElementTableCasscade(bigTable, numOfExamplesPerSequence + 1);

        System.out.println("Number of pieces: " + pieceList.size());
        System.out.println("Last piece table: " + pieceList.get(pieceList.size() - 1).printSelf());

        File outputDir = new File("../../WindBell/WindBell/resources/training/BasicLSTMData/" + symbol);
        if(outputDir.exists())
        {
            for (File f : outputDir.listFiles())
            {
                if (!f.delete()) {
                    System.err.println("File " + f.getPath() + " cannot be deleted!");
                    return;
                }
            }
        }
        else if(!outputDir.mkdirs())
        {
            System.err.println("Folder " + outputDir.getPath() + " cannot be created!");
            return;
        }

        int pieceCount = 0;
        for(OHLCElementTable piece : pieceList)
        {
            //Skip the last piece since it won't have a valid label.
            if(++pieceCount == pieceList.size())
                break;

//            OHLCElementTable normalizedPiece = SeriesDataProcessUtil.normalize(piece);
//            if(normalizedPiece.size() <= 1)
//            {
//                System.err.println("The normalized OHLCElementTable piece doesn't have enough time series to be trained, skipped.");
//                continue;
//            }
            int[] labels = new int[piece.size()-1];
            List<OHLCElement> elemList = piece.getOHCLElementsSortedByDate();
            for(int idx = 1; idx < elemList.size(); ++idx)
                labels[idx-1] = generateLabel_7(elemList.get(idx-1), elemList.get(idx));
            List<OHLCElementTable> tableList = new ArrayList<>(1);
            tableList.add(piece);
            OHLCSequentialTrainingData trainDataElems = new OHLCSequentialTrainingData(tableList, labels, numOfExamplesPerSequence);

//            String outputFileName = Paths.get(outputDir.getAbsolutePath(), symbol + "_" + dateFormat.format(piece.getLatest().getDateValue()) + ".csv").toString();
            String outputFileName = Paths.get(outputDir.getAbsolutePath(), pieceCount + ".csv").toString();
            trainDataElems.toCSVFile(outputFileName);
            System.out.println("outputFileName: " + outputFileName);

        }


    }

    private static int generateLabel_9(OHLCElement lastElem, OHLCElement labelElem)
    {
        double lastClose = lastElem.getCloseValue();
        double labelClose = labelElem.getCloseValue();

        double move = (labelClose - lastClose) / lastClose;

        if(move <= -0.06)
            return 0;
        else if(move > -0.06 && move <= -0.03)
            return 1;
        else if(move > -0.03 && move <= -0.015)
            return 2;
        else if(move > -0.015 && move <= -0.003)
            return 3;
        else if(move > -0.003 && move <= 0.003)
            return 4;
        else if(move > 0.003 && move <= 0.015)
            return 5;
        else if(move > 0.015 && move <= 0.03)
            return 6;
        else if(move > 0.03 && move <= 0.06)
            return 7;
        else //(move > 0.06)
            return 8;

    }

    private static int generateLabel_7(OHLCElement lastElem, OHLCElement labelElem)
    {
        double lastClose = lastElem.getCloseValue();
        double labelClose = labelElem.getCloseValue();

        double move = (labelClose - lastClose) / lastClose;

        int label = 0;
        if(move <= -0.04)
            return label;
        else if(move > -0.04 && move <= -0.015)
            return label + 1;
        else if(move > -0.015 && move <= -0.005)
            return label + 2;
        else if(move > -0.005 && move <= 0.003)
            return label + 3;
        else if(move > 0.003 && move <= 0.01)
            return label + 4;
        else if(move > 0.01 && move <= 0.02)
            return label + 5;
        else //(move > 0.02)
            return label + 6;

    }
    private static int generateLabel_5(OHLCElement lastElem, OHLCElement labelElem)
    {
        double lastClose = lastElem.getCloseValue();
        double labelClose = labelElem.getCloseValue();

        double move = (labelClose - lastClose) / lastClose;

        int label = 0;
        if(move <= -0.025)
            return label;
        else if(move > -0.025 && move <= -0.008)
            return label + 1;
        else if(move > -0.008 && move <= 0.006)
            return label + 2;
        else if(move > 0.006 && move <= 0.015)
            return label + 3;
        else //(move > 0.015)
            return label + 4;

    }

    private static int generateLabel_3(OHLCElement lastElem, OHLCElement labelElem)
    {
        double lastClose = lastElem.getCloseValue();
        double labelClose = labelElem.getCloseValue();

        double move = (labelClose - lastClose) / lastClose;

        int label = 0;
        if(move <= -0.015)
            return label;
        else if(move > -0.015 && move <= 0.01)
            return label + 1;
        else //(move > 0.015)
            return label + 2;

    }

    private static int generateLabel_2(OHLCElement lastElem, OHLCElement labelElem)
    {
        double lastClose = lastElem.getCloseValue();
        double labelClose = labelElem.getCloseValue();

        double move = (labelClose - lastClose) / lastClose;

        int label = 0;
        if(move <= 0.0)
            return label;
        else
            return label + 1;
    }

    public static void main(String[] args)
    {
        generateLSTMTrainingData("SPX", "../../WindBell/WindBell/resources/source/Yahoo/SPX_Daily_All.csv", 56);
    }
}
