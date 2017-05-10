package com.byeautumn.wb.analysis;

import com.byeautumn.wb.analysis.TrendAndDirectionBasic.MOVING_DIRECTION;
import com.byeautumn.wb.analysis.TrendAndDirectionBasic.TRENDING_CALCULATION_METHOD;
import com.byeautumn.wb.data.OHLCElement;
import com.byeautumn.wb.data.OHLCElementTable;
import com.byeautumn.wb.data.OHLCUtils;

public class OHLCDirectionalMovingSegment extends OHLCElementTable
{
	private MOVING_DIRECTION movingDirection = MOVING_DIRECTION.UNKNOWN;
	private TRENDING_CALCULATION_METHOD trendingByMethod = TRENDING_CALCULATION_METHOD.OHLC_RELATIONSHIP;
	public MOVING_DIRECTION getMovingDirection()
	{
		return movingDirection;
	}
	
	public TRENDING_CALCULATION_METHOD getTrendingByMethod()
	{
		return trendingByMethod;
	}
	
	public void setTrendingByMethod(TRENDING_CALCULATION_METHOD trendingByMethod)
	{
		this.trendingByMethod = trendingByMethod;
	}
	
	public OHLCDirectionalMovingSegment(OHLCElement currElem, OHLCElementTable dataSource)
	{
		//TODO check inputs...
		calculate(currElem, dataSource);
	}
	
	public OHLCDirectionalMovingSegment(OHLCElementTable dataSource)
	{
		//TODO check inputs...
		OHLCElement currElem = dataSource.getLatest();
		calculate(currElem, dataSource);
	}

	private void calculate(OHLCElement currElem, OHLCElementTable dataSource)
	{
		if(null == currElem)
			return;
		
		addOHLCElement(currElem);
		
		OHLCElement prevElem = dataSource.getOHLCElementBefore(currElem.getDateValue());
		if(null == prevElem)
		{
			movingDirection = MOVING_DIRECTION.SIDEWAYS;
			return;
		}
		
		MOVING_DIRECTION currentDirection = MOVING_DIRECTION.UNKNOWN;
		while(null != prevElem)
		{
			if(movingDirection == MOVING_DIRECTION.UNKNOWN)
			{
				if(prevElem.getOpenValue() < currElem.getCloseValue())
					movingDirection = MOVING_DIRECTION.UP;
				else if(prevElem.getOpenValue() > currElem.getCloseValue())
					movingDirection = MOVING_DIRECTION.DOWN;
				else
					movingDirection = MOVING_DIRECTION.SIDEWAYS;
			}
			else
			{
				if(prevElem.getOpenValue() < currElem.getCloseValue())
					currentDirection = MOVING_DIRECTION.UP;
				else if(prevElem.getOpenValue() > currElem.getCloseValue())
					currentDirection = MOVING_DIRECTION.DOWN;
				else
					currentDirection = MOVING_DIRECTION.SIDEWAYS;
				
				if(movingDirection != currentDirection)
					break;
			}

			addOHLCElement(prevElem);
			
			currElem = prevElem;
			prevElem = dataSource.getOHLCElementBefore(prevElem.getDateValue());
		}
	}
	
	public String printSelf()
	{
		String strDirection = "UNKNOWN";
		if(movingDirection == MOVING_DIRECTION.UP)
			strDirection = "UP";
		else if (movingDirection == MOVING_DIRECTION.DOWN)
			strDirection = "DOWN";
		else if (movingDirection == MOVING_DIRECTION.SIDEWAYS)
			strDirection = "SIDEWAYS";
		return "|| Direction: " + strDirection + "\n" + super.printSelf();
	}
	
	public static void main(String[] args)
	{
		// TODO Auto-generated method stub
		String sourceFile = "/Users/byeautumn/Downloads/Z_Weekly.csv";
		OHLCElementTable table = OHLCUtils.readOHLCDataSourceFile(sourceFile);
		
		OHLCDirectionalMovingSegment movingSeg = new OHLCDirectionalMovingSegment(table);
		System.out.println(movingSeg.printSelf());
	}

}
