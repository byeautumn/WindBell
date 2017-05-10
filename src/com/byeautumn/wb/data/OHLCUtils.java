package com.byeautumn.wb.data;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.List;

public class OHLCUtils
{
	public static OHLCElement buildOHLCElement(OHLCDataLine dataLine) throws ParseException
	{
		String[] strData = dataLine.getData();
		if(null == strData || strData.length < OHLCDataLineFormater.DATA_ITEM.values().length)
			return null;
		
		OHLCDataLineFormater formater = dataLine.getFormater();
		int dateIdx = formater.getDataItemIndex(OHLCDataLineFormater.DATA_ITEM.DATE);
		int openIdx = formater.getDataItemIndex(OHLCDataLineFormater.DATA_ITEM.OPEN);
		int highIdx = formater.getDataItemIndex(OHLCDataLineFormater.DATA_ITEM.HIGH);
		int lowIdx = formater.getDataItemIndex(OHLCDataLineFormater.DATA_ITEM.LOW);
		int closeIdx = formater.getDataItemIndex(OHLCDataLineFormater.DATA_ITEM.CLOSE);
		int volumeIdx = formater.getDataItemIndex(OHLCDataLineFormater.DATA_ITEM.VOLUME);
		SimpleDateFormat dateFormat = formater.getDateFormat();
		OHLCElement elem = new OHLCElement();
		elem.setDateValue(dateFormat.parse(strData[dateIdx]));
		elem.setOpenValue(Double.parseDouble(strData[openIdx]));
		elem.setHighValue(Double.parseDouble(strData[highIdx]));
		elem.setLowValue(Double.parseDouble(strData[lowIdx]));
		elem.setCloseValue(Double.parseDouble(strData[closeIdx]));
		elem.setVolumeValue(Long.parseLong(strData[volumeIdx]));
		
		return elem;
	}
	
	public static OHLCElementTable buildOHLCElementTable(List<String> textData, boolean isFirstLineTitle)
	{
		if(null == textData)
			return null;
		
		OHLCElementTable table = new OHLCElementTable();
		int idx = isFirstLineTitle ? 1 : 0;
		
		OHLCDataLineFormater formater = null;
		
		if(isFirstLineTitle)
			formater = OHLCDataLineFormater.parseFormaterString(textData.get(0), ",");
		
		if(null == formater)
			formater = OHLCDataLineFormater.getDefault();
		
		for(; idx < textData.size(); ++idx)
		{
			OHLCDataLine dataLine = new OHLCDataLine(textData.get(idx));
			dataLine.setFormater(formater);
			OHLCElement elem = null;
			try
			{
				elem = OHLCUtils.buildOHLCElement(dataLine);
			}catch(ParseException pe)
			{
				pe.printStackTrace();
				System.err.println("Error line: " + textData.get(idx));
				continue;
			} catch (NumberFormatException nfe)
			{
				nfe.printStackTrace();
				System.err.println("Error line: " + textData.get(idx));
				continue;
			}
			if(null != elem)
				table.addOHLCElement(elem);
		}
		return table;
	}
	
	public static OHLCElementTable readOHLCDataSourceFile(String fileName)
	{
		Path path = Paths.get(fileName);
		try
		{
			List<String> strList = Files.readAllLines(path, Charset.defaultCharset());
			return OHLCUtils.buildOHLCElementTable(strList, true);
		} catch (IOException e)
		{
			e.printStackTrace();
		}
		return null;
	}
	
	public static void main(String[] args)
	{
		String textLine = "2014-09-09,131.01,133.73,129.22,130.11,1599700,130.11";
		OHLCDataLine dataLine = new OHLCDataLine(textLine);
		try
		{
			OHLCElement elem = OHLCUtils.buildOHLCElement(dataLine);
			System.out.println(elem.printSelf());
		}catch(ParseException pe)
		{
			pe.printStackTrace();
		}
		System.out.println("++++++++++++++++++++++++");
		String sourceFile = "src/com/byeautumn/wb/input/source/Yahoo/SPX_Daily_All.csv";
		OHLCElementTable table = OHLCUtils.readOHLCDataSourceFile(sourceFile);
		System.out.println(table.getEarliest().printSelf());
	}

}
