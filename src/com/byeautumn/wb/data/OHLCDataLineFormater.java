package com.byeautumn.wb.data;

import java.text.SimpleDateFormat;
import java.util.HashMap;

public class OHLCDataLineFormater
{
	public enum DATA_ITEM 
	{
		DATE, 
		OPEN, 
		HIGH, 
		LOW, 
		CLOSE, 
		VOLUME;
		
		public String getName()
		{
			return name();
		}
	}
	
	private HashMap<DATA_ITEM, Integer> itemIndexMap;
	private String spliter = ",";
	private SimpleDateFormat dateFormat;
	public static OHLCDataLineFormater getDefault()
	{
		OHLCDataLineFormater formater = new OHLCDataLineFormater();
		formater.setDataItemIndex(DATA_ITEM.DATE, 0);
		formater.setDataItemIndex(DATA_ITEM.OPEN, 1);
		formater.setDataItemIndex(DATA_ITEM.HIGH, 2);
		formater.setDataItemIndex(DATA_ITEM.LOW, 3);
		formater.setDataItemIndex(DATA_ITEM.CLOSE, 4);
		formater.setDataItemIndex(DATA_ITEM.VOLUME, 5);
		formater.setDateFormat(new SimpleDateFormat("yyyy-MM-dd"));
		
		return formater;
	}
	
	public void setDataItemIndex(DATA_ITEM item, int index)
	{
		if(null == itemIndexMap)
			itemIndexMap = new HashMap<DATA_ITEM, Integer>();
		
		itemIndexMap.put(item, index);
	}
	
	public int getDataItemIndex(DATA_ITEM item)
	{
		Integer itemIdx = itemIndexMap.get(item);
		
		if(null == itemIdx)
			return -1;
		
		return itemIdx.intValue();
	}

	public String getSpliter()
	{
		return spliter;
	}

	public void setSpliter(String spliter)
	{
		this.spliter = spliter;
	}

	public SimpleDateFormat getDateFormat()
	{
		return dateFormat;
	}

	public void setDateFormat(SimpleDateFormat dateFormat)
	{
		this.dateFormat = dateFormat;
	}
	
	public boolean isAllSet()
	{
		if(null != itemIndexMap && itemIndexMap.size() >= DATA_ITEM.values().length)
			return true;
		
		return false;
	}
	
	public static OHLCDataLineFormater parseFormaterString(String formaterStr, String spliter)
	{
		if(null == formaterStr || null == spliter || formaterStr.isEmpty() || spliter.isEmpty())
			return OHLCDataLineFormater.getDefault();
		
		String[] strArr = formaterStr.split(spliter);
		if(null != strArr && strArr.length >= DATA_ITEM.values().length)
		{
			OHLCDataLineFormater formater = new OHLCDataLineFormater();
			int idx = 0;
			for(String s : strArr)
			{
				if(DATA_ITEM.OPEN.name().toLowerCase().equals(s.toLowerCase()))
				{
					formater.setDataItemIndex(DATA_ITEM.OPEN, idx);
				}
				else if(DATA_ITEM.LOW.name().toLowerCase().equals(s.toLowerCase()))
				{
					formater.setDataItemIndex(DATA_ITEM.LOW, idx);
				}
				else if(DATA_ITEM.HIGH.name().toLowerCase().equals(s.toLowerCase()))
				{
					formater.setDataItemIndex(DATA_ITEM.HIGH, idx);
				}
				else if(DATA_ITEM.CLOSE.name().toLowerCase().equals(s.toLowerCase()))
				{
					formater.setDataItemIndex(DATA_ITEM.CLOSE, idx);
				}
				else if(DATA_ITEM.VOLUME.name().toLowerCase().equals(s.toLowerCase()))
				{
					formater.setDataItemIndex(DATA_ITEM.VOLUME, idx);
				}
				else if(DATA_ITEM.DATE.name().toLowerCase().equals(s.toLowerCase()))
				{
					formater.setDataItemIndex(DATA_ITEM.DATE, idx);
				}
				
				++idx;
			}
			
			if(formater.isAllSet())
			{
				formater.setDateFormat(new SimpleDateFormat("yyyy-MM-dd"));
				return formater;
			}
			
		}

		
		return getDefault();
	}
}
