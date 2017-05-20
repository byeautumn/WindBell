package com.byeautumn.wb.process;

import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import com.byeautumn.wb.common.Constants;
import com.byeautumn.wb.data.OHLCElement;
import com.byeautumn.wb.data.OHLCElementTable;
import com.byeautumn.wb.data.OHLCUtils;

public class SeriesDataProcessUtil
{

	public static ArrayList<OHLCElementTable> splitOHLCElementTableCasscade(OHLCElementTable bigTable, int pieceSize)
	{
		ArrayList<OHLCElementTable> retArr = new ArrayList<>();
		if(bigTable.size() <= pieceSize || pieceSize <= 0)
		{
			retArr.add(bigTable);
			return retArr;
		}
		
		List<OHLCElement> elemArr = bigTable.getOHCLElementsSortedByDate();
		for(int idx = 0; idx <= elemArr.size() - pieceSize; ++idx)
		{
			OHLCElementTable pieceTable = new OHLCElementTable();
			for(int secIdx = idx; secIdx < idx + pieceSize; ++secIdx)
			{
				OHLCElement elem = elemArr.get(secIdx);
				pieceTable.addOHLCElement(elem);
			}
			retArr.add(pieceTable);
		}
		
		return retArr;
	}
	
	public static OHLCElementTable normalize(OHLCElementTable elemTable)
	{
		if(elemTable == null || elemTable.size() < 1)
			return null;
		
		OHLCElementTable retTable = new OHLCElementTable();
		//find the smallest "low" and "volume".
		double lowBase = Double.MAX_VALUE;
		double volumeMinBase = Double.MAX_VALUE;
		double highBase = 0.0;
		double volumeMaxBase = 0.0;
		for (OHLCElement elem : elemTable.getOHCLElementsSortedByDate())
		{
			lowBase = Math.min(lowBase, elem.getLowValue());
			highBase = Math.max(highBase, elem.getHighValue());
			volumeMinBase = Math.min(volumeMinBase, elem.getVolumeValue());
			volumeMaxBase = Math.max(volumeMaxBase, elem.getVolumeValue());
		}
		
		for (OHLCElement elem : elemTable.getOHCLElementsSortedByDate())
		{
			OHLCElement retElem = normalize(elem, lowBase, highBase, volumeMinBase, volumeMaxBase);
			if(null != retElem)
				retTable.addOHLCElement(retElem);
		}
		
		return retTable;
	}
	
	public static OHLCElement normalize(OHLCElement elem, double lowBase, double highBase, double volumeMinBase, double volumeMaxBase)
	{
		if(null == elem || Math.abs(lowBase) < Constants.EPSILON || Math.abs(highBase) < Constants.EPSILON 
				|| volumeMinBase < Constants.EPSILON  || volumeMaxBase < Constants.EPSILON  || lowBase > highBase || volumeMinBase > volumeMaxBase)
			return null;
		
		OHLCElement retElem = new OHLCElement();
		
		double pDiff = highBase - lowBase;
		double vDiff = volumeMaxBase - volumeMinBase;
		double open = (elem.getOpenValue() - lowBase) / pDiff;
		double high = (elem.getHighValue() - lowBase ) / pDiff;
		double low = (elem.getLowValue() - lowBase) / pDiff;
		double close = (elem.getCloseValue() - lowBase) / pDiff;
		double volume = (elem.getVolumeValue() - volumeMinBase) / vDiff;
		
		retElem.setDateValue(elem.getDateValue());
		retElem.setOpenValue(open);
		retElem.setHighValue(high);
		retElem.setLowValue(low);
		retElem.setCloseValue(close);
		retElem.setVolumeValue(volume);
		
		return retElem;
	}
	
	public static void main(String[] args)
	{
		System.out.println("user.dir:" + System.getProperty("user.dir"));
		String sourceFile = "../../WindBell/WindBell/resources/source/Yahoo/SPX_Daily_All.csv";
		OHLCElementTable bigTable = OHLCUtils.readOHLCDataSourceFile(sourceFile);
		System.out.println("Big table size: " + bigTable.size());
		
		List<OHLCElementTable> pieceList = splitOHLCElementTableCasscade(bigTable, 5);
		
		System.out.println("Number of pieces: " + pieceList.size());
		System.out.println("Last piece table: " + pieceList.get(pieceList.size() - 1).printSelf());
		
		OHLCElementTable lastPiece = pieceList.get(pieceList.size() - 1);
		OHLCElementTable normalizedPiece = normalize(lastPiece);
		System.out.println("Normalized last piece table: " + normalizedPiece.printSelf());

	}

}
