package com.byeautumn.wb.data;

import java.text.SimpleDateFormat;
import java.util.Date;

public class OHLCElement
{
	private Date dateValue;
	private double highValue;
	private double openValue;
	private double lowValue;
	private double closeValue;
	private double volumeValue;
	private SimpleDateFormat dateFormat = new SimpleDateFormat("MM/dd/yyyy");
	
	public Date getDateValue()
	{
		return dateValue;
	}
	public void setDateValue(Date dateValue)
	{
		this.dateValue = dateValue;
	}
	public double getHighValue()
	{
		return highValue;
	}
	public void setHighValue(double highValue)
	{
		this.highValue = highValue;
	}
	public double getOpenValue()
	{
		return openValue;
	}
	public void setOpenValue(double openValue)
	{
		this.openValue = openValue;
	}
	public double getLowValue()
	{
		return lowValue;
	}
	public void setLowValue(double lowValue)
	{
		this.lowValue = lowValue;
	}
	public double getCloseValue()
	{
		return closeValue;
	}
	public void setCloseValue(double closeValue)
	{
		this.closeValue = closeValue;
	}
	public double getVolumeValue()
	{
		return volumeValue;
	}
	public void setVolumeValue(double volumeValue)
	{
		this.volumeValue = volumeValue;
	}
	
	public void setDateFormat(SimpleDateFormat dateFormat)
	{
		this.dateFormat = dateFormat;
	}
	
	public String printSelf()
	{
		StringBuffer sb = new StringBuffer();
		sb.append("|| Date: ").append(dateFormat.format(dateValue)).append(" | ");
		sb.append("Open: ").append(openValue).append(" | ");
		sb.append("High: ").append(highValue).append(" | ");
		sb.append("Low: ").append(lowValue).append(" | ");
		sb.append("Close: ").append(closeValue).append(" | ");
		sb.append("Volume: ").append(volumeValue).append(" || ");
		
		return sb.toString();
	}

}
