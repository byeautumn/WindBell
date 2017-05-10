package com.byeautumn.wb.analysis;

import java.util.Collection;

import com.byeautumn.wb.data.OHLCElement;
import com.byeautumn.wb.data.OHLCElementTable;
import com.byeautumn.wb.data.OHLCUtils;

public class OHLCPercentageAnalyzer
{
	private OHLCElementTable dataTable;
	public OHLCPercentageAnalyzer(OHLCElementTable dataTable)
	{
		//TODO check inputs...
		
		this.dataTable = dataTable;
	}
	
	public String computeBidirectionSwingPercentages()
	{
		StringBuffer sb = new StringBuffer();
		
		Collection<OHLCElement> elements = dataTable.getOHCLElementsSortedByDate();
		if(null == elements)
			return sb.toString();
		
		sb.append("********** Bi-direction swing percentage report ************").append("\n");
		int bucketCount = 10;
		int[] countBuckets = new int[bucketCount]; //1%, 2%, 3%...10%;
		int elemCount = elements.size();
		sb.append("Total sample count: ").append(elemCount).append("\n");
		for(OHLCElement elem : elements)
		{
			double minValue = Math.min(Math.abs(elem.getHighValue() - elem.getOpenValue()), Math.abs(elem.getLowValue() - elem.getOpenValue()));
			double minSwingPercentage = minValue/elem.getOpenValue();
			int bucketIndex = (int)Math.floor(minSwingPercentage * 100);
			if(bucketIndex < bucketCount)
				countBuckets[bucketIndex]++;
			else
				countBuckets[9]++;
		}
		
		for(int ii = 0; ii < bucketCount; ++ii)
		{
			sb.append("").append(ii + 1).append("%: ").append(countBuckets[ii]).append(", ").append(((double)countBuckets[ii])/((double)elemCount) * 100.00).append("\n");
		}
		
		int subTotal = 0;
		for(int ii = 0; ii < bucketCount; ++ii)
		{
			subTotal += countBuckets[ii];
			sb.append("<=").append(ii + 1).append("%: ").append(subTotal).append(", ").append(((double)subTotal)/((double)elemCount) * 100.00).append("\n");
		}
		
		return sb.toString();
	}
	
	public String computeSingleDirectionExtremePercentages()
	{
		StringBuffer sb = new StringBuffer();
		
		Collection<OHLCElement> elements = dataTable.getOHCLElementsSortedByDate();
		if(null == elements)
			return sb.toString();
		
		sb.append("********** Single Direction Extreme Percentages Report ************").append("\n");
		int bucketCount = 10;
		int[] countBuckets = new int[bucketCount]; //1%, 2%, 3%...10%;
		int elemCount = elements.size();
		sb.append("Total sample count: ").append(elemCount).append("\n");
		for(OHLCElement elem : elements)
		{
			double maxValue = Math.max(Math.abs(elem.getHighValue() - elem.getOpenValue()), Math.abs(elem.getLowValue() - elem.getOpenValue()));
			double maxPercentage = maxValue/elem.getOpenValue();
			int bucketIndex = (int)Math.floor(maxPercentage * 100);
			if(bucketIndex < bucketCount)
				countBuckets[bucketIndex]++;
			else
				countBuckets[9]++;
		}
		
		for(int ii = 0; ii < bucketCount; ++ii)
		{
			sb.append("").append(ii + 1).append("%: ").append(countBuckets[ii]).append(", ").append(((double)countBuckets[ii])/((double)elemCount) * 100.00).append("\n");
		}
		
		int subTotal = 0;
		for(int ii = 0; ii < bucketCount; ++ii)
		{
			subTotal += countBuckets[ii];
			sb.append("<=").append(ii + 1).append("%: ").append(subTotal).append(", ").append(((double)subTotal)/((double)elemCount) * 100.00).append("\n");
		}
		
		return sb.toString();
	}
	
	public String printSelf()
	{
		return "";
	}
	
	public static void main(String[] args)
	{
		String sourceFile = "/Users/byeautumn/Downloads/Z_Weekly.csv";
		OHLCElementTable table = OHLCUtils.readOHLCDataSourceFile(sourceFile);
		OHLCPercentageAnalyzer analyzer = new OHLCPercentageAnalyzer(table);
		System.out.println(analyzer.computeBidirectionSwingPercentages());
		System.out.println(analyzer.computeSingleDirectionExtremePercentages());

	}

}
