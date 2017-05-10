package com.byeautumn.wb.analysis;

import java.util.Date;
import java.util.TreeMap;

import com.byeautumn.wb.data.OHLCElement;
import com.byeautumn.wb.data.OHLCElementTable;
import com.byeautumn.wb.data.OHLCUtils;

public class OHLCDirectionalMovingSegmentChain
{
	private OHLCElementTable dataTable;
	private TreeMap<Date, OHLCDirectionalMovingSegment> segmentMap;
	public OHLCDirectionalMovingSegmentChain(OHLCElementTable dataTable)
	{
		//TODO check inputs...
		
		this.dataTable = dataTable;
		calculate(dataTable);
	}
	
	private void calculate(OHLCElementTable dataTable)
	{
		segmentMap = new TreeMap<Date, OHLCDirectionalMovingSegment>();
		OHLCElement currElem = dataTable.getLatest();
		while(null != currElem)
		{
			OHLCDirectionalMovingSegment segment = new OHLCDirectionalMovingSegment(currElem, dataTable);
			
			segmentMap.put(currElem.getDateValue(), segment);
			OHLCElement prevElem = segment.getEarliest();
			currElem = dataTable.getOHLCElementBefore(prevElem.getDateValue());	
		}
	}
	
	public String printSelf()
	{
		StringBuffer sb = new StringBuffer();
		for(OHLCDirectionalMovingSegment segment : segmentMap.values())
		{
			sb.append("||||||||||||||||||||||||||||||\n");
			sb.append(segment.printSelf());	
		}
		
		return sb.toString();
	}
	
	public static void main(String[] args)
	{
		// TODO Auto-generated method stub
		String sourceFile = "src/com/byeautumn/wb/input/source/Yahoo/TSLA_Daily.csv";//"/Users/byeautumn/Downloads/Z_Daily.csv";
		OHLCElementTable table = OHLCUtils.readOHLCDataSourceFile(sourceFile);
		
		OHLCDirectionalMovingSegmentChain movingSegChain = new OHLCDirectionalMovingSegmentChain(table);
		System.out.println(movingSegChain.printSelf());
	}

}
