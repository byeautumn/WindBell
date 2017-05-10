package com.byeautumn.wb.output;

import com.byeautumn.wb.data.OHLCElement;

public class OHLCBarImageBW
{
	private int[][] iMatrix;
	
	public OHLCBarImageBW(int[][] iMatrix)
	{
		this.iMatrix = iMatrix;
	}

	public int[][] getPixelMatrix()
	{
		return iMatrix;
	}
	
	public int getWidth()
	{
		if(null == iMatrix || iMatrix[0].length < 1)
			return 0;
		
		return iMatrix[0].length;
	}
	
	public int getHeight()
	{
		if(null == iMatrix)
			return 0;
		
		return iMatrix.length;
	}
	
	public String printSelf()
	{
		if(null == iMatrix)
			return "";
		int h = iMatrix.length;
		if(h < 1)
			return "";
		int w = iMatrix[0].length;
		
		StringBuffer sb = new StringBuffer();
		for(int i = 0; i < h; ++i)
		{
			for(int j = 0; j < w; ++j)
			{
				sb.append(iMatrix[i][j]).append(" ");
			}
			sb.append("\n");
		}
		
		return sb.toString();
	}
	public static void main(String[] args)
	{
		// TODO Auto-generated method stub

	}

}
