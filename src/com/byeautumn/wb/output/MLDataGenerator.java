package com.byeautumn.wb.output;

import java.io.File;
import java.nio.file.Files;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import com.byeautumn.wb.data.OHLCElement;
import com.byeautumn.wb.data.OHLCElementTable;
import com.byeautumn.wb.data.OHLCUtils;
import com.byeautumn.wb.process.SeriesDataProcessUtil;

public class MLDataGenerator
{
	private static SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd");
	public static void generateCNNTrainingData(String symbol, String sourceFile)
	{
		OHLCElementTable bigTable = OHLCUtils.readOHLCDataSourceFile(sourceFile);
		System.out.println("Big table size: " + bigTable.size());
		
		List<OHLCElementTable> pieceList = SeriesDataProcessUtil.splitOHLCElementTableCasscade(bigTable, 5);
		
		System.out.println("Number of pieces: " + pieceList.size());
		System.out.println("Last piece table: " + pieceList.get(pieceList.size() - 1).printSelf());
		
		File outputDir = new File("src/com/byeautumn/wb/output/result/20170507/");
		if(outputDir.exists())
		{
			if(!outputDir.delete())
			{
				System.err.println("Folder " + outputDir.getPath() + " cannot be deleted!");
				return;
			}
		}
		if(!outputDir.mkdirs())
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
			
			OHLCElementTable normalizedPiece = SeriesDataProcessUtil.normalize(piece);
			
			ArrayList<OHLCBarImageBW> imageArr = new ArrayList<OHLCBarImageBW>(normalizedPiece.size());
			for(OHLCElement elem : normalizedPiece.getOHCLElementsSortedByDate())
			{
				OHLCBarImageBW image = OHLCImageGenerator.generateBarImageBW(elem, 50);
//				System.out.println("++++++++++++++");
//				System.out.println(image.printSelf());
				imageArr.add(image);
			}
			OHLCBarImageBW bigImage = OHLCImageGenerator.combineSeriesImagesBW(imageArr, 2, true);
			
//			System.out.println("++++++++++++++");
//			System.out.println(bigImage.printSelf());
			
			OHLCBarImageBW biggerImage = OHLCImageGenerator.dialate(bigImage, 2);
//			System.out.println("++++++++++++++");
//			System.out.println(biggerImage.printSelf());
			OHLCElement lastElem = piece.getLatest();
			Date lastDate = lastElem.getDateValue();
			OHLCElement labelElem = bigTable.getOHLCElementAfter(lastDate);
			String label = generateCNNLabel(lastElem, labelElem);
			String outputFileName = outputDir.getAbsolutePath() + "/" + generateCNNDataFileName(symbol, labelElem, "5", label);
			System.out.println("The output file name is: " + outputFileName);
			
//			break;
			OHLCImageGenerator.saveOHLCBarImageBW(biggerImage, outputFileName);
		}


			
	}
	
	private static String generateCNNLabel(OHLCElement lastElem, OHLCElement labelElem)
	{
		double lastClose = lastElem.getCloseValue();
		double labelClose = labelElem.getCloseValue();
		
		double move = (labelClose - lastClose) / lastClose;
		
		if(move <= -0.06)
			return "1";
		else if(move > -0.06 && move <= -0.03)
			return "2";
		else if(move > -0.03 && move <= -0.015)
			return "3";
		else if(move > -0.015 && move <= -0.003)
			return "4";
		else if(move > -0.003 && move <= 0.003)
			return "5";
		else if(move > 0.003 && move <= 0.015)
			return "6";
		else if(move > 0.015 && move <= 0.03)
			return "7";
		else if(move > 0.03 && move <= 0.06)
			return "8";
		else //(move > 0.06)
			return "9";
		
	}
	
	private static String generateCNNDataFileName(String symbol, OHLCElement labelElem, String specialInfoStr, String label)
	{
		StringBuffer sb = new StringBuffer();
		sb.append(symbol).append("_").append(dateFormat.format(labelElem.getDateValue()));
		sb.append("_").append(specialInfoStr).append("_").append(label).append(".jpg");
		
		return sb.toString();
	}
	
	public static void main(String[] args)
	{
		String sourceFile = "src/com/byeautumn/wb/input/source/Yahoo/SPX_Daily_All.csv";
		MLDataGenerator.generateCNNTrainingData("SPX", sourceFile);

	}

}
