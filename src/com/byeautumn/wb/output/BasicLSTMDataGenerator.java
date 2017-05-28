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

//        //Force to reset the numOfExamplesPerSequence
//        numOfExamplesPerSequence = 56;

        //NOte: here ere pass numOfExamplesPerSequence + 1 because for labeling we need 1 more time spot.
        List<OHLCElementTable> pieceList = SeriesDataProcessUtil.splitOHLCElementTableCasscade(bigTable, numOfExamplesPerSequence + 1);

        System.out.println("Number of pieces: " + pieceList.size());
//        System.out.println("Last piece table: " + pieceList.get(pieceList.size() - 1).printSelf());

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

            int[] labels = new int[piece.size()-1];
            List<OHLCElement> elemList = piece.getOHCLElementsSortedByDate();
            for(int idx = 1; idx < elemList.size(); ++idx) {
                labels[idx - 1] = BasicLSTMLabelingManager.generateLabel_7(elemList.get(idx - 1).getCloseValue(), elemList.get(idx).getCloseValue());
            }

            List<OHLCElementTable> tableList = new ArrayList<>(1);
            tableList.add(piece);
            OHLCSequentialTrainingData trainDataElems = new OHLCSequentialTrainingData(tableList);


//            String outputFileName = Paths.get(outputDir.getAbsolutePath(), symbol + "_" + dateFormat.format(piece.getLatest().getDateValue()) + ".csv").toString();
            String outputFileName = Paths.get(outputDir.getAbsolutePath(), pieceCount + ".csv").toString();
            trainDataElems.generateTrainingCSVFile(outputFileName);
            System.out.println("outputFileName: " + outputFileName);

        }


    }

    private static OHLCSequentialTrainingData generateOHLCSequentialData(List<String> sourceFiles, int numOfExamplesPerSequence)
    {
        List<OHLCElementTable> rawTables = new ArrayList<>(sourceFiles.size());
        for(String sourceFile : sourceFiles) {
            OHLCElementTable bigTable = OHLCUtils.readOHLCDataSourceFile(sourceFile);
            //NOTE: normalize it here...
//            bigTable.normalizeDataByYearRange();
            System.out.println("Big table size: " + bigTable.size());

            rawTables.add(bigTable);
        }
        System.out.println("RawTables size: " + rawTables.size());
        OHLCSequentialTrainingData allTrainData = new OHLCSequentialTrainingData(rawTables);

        return allTrainData;
    }

    public static void generateLSTMTrainingData2(String symbol, List<String> sourceFiles, int numOfExamplesPerSequence)
    {
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

        OHLCSequentialTrainingData allTrainData = generateOHLCSequentialData(sourceFiles, numOfExamplesPerSequence);

        List<OHLCSequentialTrainingData> trainDataList = allTrainData.split(numOfExamplesPerSequence);

        int pieceCount = 0;
        for(OHLCSequentialTrainingData trainData : trainDataList)
        {
            String outputFileName = Paths.get(outputDir.getAbsolutePath(), pieceCount + ".csv").toString();
            trainData.generateTrainingCSVFile(outputFileName);
            ++pieceCount;
        }
        System.out.println("Total number of csv files generated: " + pieceCount);

    }

    public static void generateLSTMPredictionData(String symbol, List<String> sourceFiles, int numOfExamplesPerSequence)
    {
        File outputDir = new File("../../WindBell/WindBell/resources/predict/BasicLSTMData/" + symbol);
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

        OHLCSequentialTrainingData allTrainData = generateOHLCSequentialData(sourceFiles, numOfExamplesPerSequence);
        List<OHLCSequentialTrainingData> trainDataList = allTrainData.split(numOfExamplesPerSequence);



        OHLCSequentialTrainingData predictData = trainDataList.get(trainDataList.size() - 1);
        String outputFileName = Paths.get(outputDir.getAbsolutePath(), "0.csv").toString();
        predictData.generatePredictionCSVFile(outputFileName);

        System.out.println("The prediction data file has been generated: " + outputFileName);
    }

    public static void main(String[] args)
    {
        List<String> sourceFileNames = new ArrayList<>();
        sourceFileNames.add("../../WindBell/WindBell/resources/source/Yahoo/SPX_Daily_All.csv");
        sourceFileNames.add("../../WindBell/WindBell/resources/source/Yahoo/VIX_Daily.csv");
//        sourceFileNames.add("../../WindBell/WindBell/resources/source/Yahoo/TLT_Daily.csv");
//        sourceFileNames.add("../../WindBell/WindBell/resources/source/Yahoo/GLD_Daily.csv");
        generateLSTMTrainingData2("SPX", sourceFileNames, 56);//Two month for stocks and indices.
    }
}
