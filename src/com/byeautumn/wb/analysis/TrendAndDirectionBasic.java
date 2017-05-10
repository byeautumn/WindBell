package com.byeautumn.wb.analysis;

public class TrendAndDirectionBasic
{
	public enum MOVING_DIRECTION {UNKNOWN, UP, DOWN, SIDEWAYS}
	/*
	 * PERCENTAGE: Using movement percentage to determine the trend (trending or NOT trending)
	 * OHLC_RELATIONSHIP: Using the relationship between close value and last open value to determine the trend (UP, DOWN or DIDEWAYS)
	 */
	public enum TRENDING_CALCULATION_METHOD {PERCENTAGE, OHLC_RELATIONSHIP}
}
