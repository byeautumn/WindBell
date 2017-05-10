package com.byeautumn.wb.data;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Date;
import java.util.List;
import java.util.Map.Entry;
import java.util.TreeMap;

public class OHLCElementTable
{
	private TreeMap<Date, OHLCElement> elementMap;
	
	public void addOHLCElement(OHLCElement elem)
	{
		if(null == elementMap)
			elementMap = new TreeMap<Date, OHLCElement>();
		
		elementMap.put(elem.getDateValue(), elem);
	}
	
	public OHLCElement getOHLCElement(Date date)
	{
		return elementMap.get(date);
	}
	
	public OHLCElement getLatest()
	{
		if(null == elementMap || elementMap.isEmpty())
			return null;
		
		return elementMap.lastEntry().getValue();
	}
	
	public OHLCElement getEarliest()
	{
		if(null == elementMap || elementMap.isEmpty())
			return null;
		
		return elementMap.firstEntry().getValue();
	}
	
	public OHLCElement getOHLCElementBefore(Date date)
	{
		Entry<Date, OHLCElement> lowerEntry = elementMap.lowerEntry(date);
		if(null == lowerEntry)
			return null;
		
		return lowerEntry.getValue();
	}
	
	public OHLCElement getOHLCElementAfter(Date date)
	{
		Entry<Date, OHLCElement> higherEntry = elementMap.higherEntry(date);
		if(null == higherEntry)
			return null;
		
		return higherEntry.getValue();
	}
	
	public List<OHLCElement> getOHCLElementsSortedByDate()
	{
		if(null == elementMap)
			return null;
		
		List<OHLCElement> retList = new ArrayList<>(elementMap.size());
		for(OHLCElement elem : elementMap.values())
			retList.add(elem);
		
		return retList;
	}
	
	public int size()
	{
		if(null == elementMap)
			return 0;
		
		return elementMap.size();
	}
	
	public String printSelf()
	{
		StringBuffer sb = new StringBuffer();
		for(OHLCElement elem : elementMap.values())
		{
			sb.append(elem.printSelf()).append("\n");
		}
		return sb.toString();
	}
}
