package com.byeautumn.wb.data;

public class OHLCDataLine
{
	private String textLine;
	private String[] data;
	private OHLCDataLineFormater formater;
	public OHLCDataLine(String textLine)
	{
		if(null == textLine || textLine.isEmpty())
			return;
		formater = OHLCDataLineFormater.getDefault();
		this.textLine = textLine;
		
	}
	private void splitTextLine()
	{
		if(null == textLine || textLine.isEmpty())
			return;
		
		data = textLine.split(formater.getSpliter());
	}
	public OHLCDataLineFormater getFormater()
	{
		return formater;
	}
	public void setFormater(OHLCDataLineFormater formater)
	{
		this.formater = formater;
	}
	public String[] getData()
	{
		splitTextLine();
		return data;
	}
	
}
