package com.byeautumn.wb.data;

import java.util.*;
import java.util.Map.Entry;

public class OHLCElementTable
{
	private TreeMap<Date, OHLCElement> elementMap;
//	private TreeMap<Date, OHLCElement> normalizedElementMap;
	
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

	public void normalizeDataByYearRange()
	{
		List<OHLCElement> elemList = getOHCLElementsSortedByDate();
		Calendar calendar = Calendar.getInstance();
		calendar.setTime(elemList.get(0).getDateValue());
		int firstYear = calendar.get(Calendar.YEAR);

		int yearCount = 0;
		int currYear = firstYear;
		for(OHLCElement elem : elemList)
		{
			calendar.setTime(elem.getDateValue());
			int elemYear = calendar.get(Calendar.YEAR);
			if(elemYear > currYear)
			{
				++yearCount;
				currYear = elemYear;
			}
		}

		if(yearCount < 1) {
			return;
		}

		List<double[]> yearMinMaxValuesList = new ArrayList<>(yearCount);
		currYear = firstYear;
		double minPrice = Double.MAX_VALUE, minVolume = Double.MAX_VALUE;
		double maxPrice = Double.MIN_VALUE, maxVolume = Double.MIN_VALUE;
		for(int idx = 0; idx < elemList.size(); ++idx)
		{
			OHLCElement elem = elemList.get(idx);
			calendar.setTime(elem.getDateValue());
			int elemYear = calendar.get(Calendar.YEAR);
			if(elemYear == currYear)
			{
				minPrice = Double.min(minPrice, elem.getLowValue());
				minVolume = Double.min(minVolume, elem.getVolumeValue());
				maxPrice = Double.max(maxPrice, elem.getHighValue());
				maxVolume = Double.max(maxVolume, elem.getVolumeValue());
			}
			if(elemYear > currYear || idx == elemList.size() - 1)
			{
				double[] yearMinMaxValues = {minPrice, maxPrice, minVolume, maxVolume};
				yearMinMaxValuesList.add(yearMinMaxValues);
				currYear = elemYear;
				minPrice = Double.MAX_VALUE; minVolume = Double.MAX_VALUE;
				maxPrice = Double.MIN_VALUE; maxVolume = Double.MIN_VALUE;
			}
		}

		elementMap = new TreeMap<>();
		for(OHLCElement elem : elemList)
		{
			calendar.setTime(elem.getDateValue());
			int year = calendar.get(Calendar.YEAR);
			double[] yearMinMaxValues = yearMinMaxValuesList.get(year - firstYear);
			double pDiff = yearMinMaxValues[1] - yearMinMaxValues[0];
			double vDiff = yearMinMaxValues[3] - yearMinMaxValues[2];

			OHLCElement nElem = new OHLCElement();
			nElem.setDateValue(elem.getDateValue());
			nElem.setOpenValue(pDiff == 0.0 ? 0.0 : (elem.getOpenValue() - yearMinMaxValues[0]) / pDiff);
			nElem.setHighValue(pDiff == 0.0 ? 0.0 : (elem.getHighValue() - yearMinMaxValues[0]) / pDiff);
			nElem.setLowValue(pDiff == 0.0 ? 0.0 : (elem.getLowValue() - yearMinMaxValues[0]) / pDiff);
			nElem.setCloseValue(pDiff == 0.0 ? 0.0 : (elem.getCloseValue() - yearMinMaxValues[0]) / pDiff);
			nElem.setVolumeValue(vDiff == 0.0 ? 0.0 : (elem.getVolumeValue() - yearMinMaxValues[2]) / vDiff);
			elementMap.put(elem.getDateValue(), nElem);
		}

	}

//	public List<OHLCElement> getNormalizedOHCLElementsSortedByDate()
//	{
//		if(null == normalizedElementMap)
//			normalizeDataByYearRange();
//
//		List<OHLCElement> retList = new ArrayList<>(normalizedElementMap.size());
//		for(OHLCElement elem : normalizedElementMap.values())
//			retList.add(elem);
//
//		return retList;
//	}

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
