package com.byeautumn.wb.output;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

import com.byeautumn.wb.data.OHLCElement;
import com.byeautumn.wb.data.OHLCElementTable;
import com.byeautumn.wb.data.OHLCUtils;
import com.byeautumn.wb.process.SeriesDataProcessUtil;

public class OHLCImageGenerator
{
	public static OHLCBarImageBW generateBarImageBW(OHLCElement elem, int maxHeight)
	{
		int openBarWidth = maxHeight / 10;
		if(openBarWidth < 1)
			return null;
		
		double open = elem.getOpenValue();
		double high = elem.getHighValue();
		double low = elem.getLowValue();
		double close = elem.getCloseValue();
		
		int[][] matrix = new int[maxHeight][openBarWidth * 2 + 1];
		
		int lowPosY = (int)Math.min(maxHeight-1, Math.round(maxHeight * low));
		int highPosY = (int)Math.min(maxHeight-1, Math.round(maxHeight * high));
		int openPosY = (int)Math.min(maxHeight-1, Math.round(maxHeight * open));
		int closePosY = (int)Math.min(maxHeight-1, Math.round(maxHeight * close));
		
		
		int lowPosX = openBarWidth;
		for(int idx = maxHeight - highPosY - 1; idx < maxHeight- lowPosY; ++idx)
			matrix[idx][lowPosX] = 1;
		int closePosX = openBarWidth + 1;
		for(int idx = 0; idx < openBarWidth; ++idx)
		{			
			matrix[maxHeight - openPosY - 1][idx] = 1;
			matrix[maxHeight - closePosY - 1][idx + closePosX] = 1;
		}
		
		OHLCBarImageBW image = new OHLCBarImageBW(matrix);
		
		return image;
	}

	public static OHLCBarImageBW combineSeriesImagesBW(ArrayList<OHLCBarImageBW> imageArr, int distance, boolean isHorizontal)
	{
		int width = 0;
		int height = 0;
		int imageCount = 0; //this will ignore invalid images.
		
		//Determine the dimension of new image.
		for(OHLCBarImageBW image : imageArr)
		{
			if(image.getWidth() < 1 || image.getHeight() < 1)
				continue;
			
			if(isHorizontal)
			{
				width += image.getWidth();
				height = Math.max(height, image.getHeight());
			}
			else
			{
				width += Math.max(width, image.getWidth());
				height += image.getHeight();
			}
			++imageCount;
		}
		
		if(imageCount < 1)
			return null;
		
		if(isHorizontal)
			width += (imageCount - 1) * distance;
		else
			height += (imageCount - 1) * distance;
		
		int[][] matrix = new int[height][width];
		int startX = 0;
		int startY = 0;
		for(OHLCBarImageBW image : imageArr)
		{
			int w = image.getWidth();
			int h = image.getHeight();
			if(w < 1 || h < 1)
				continue;
			int[][] m = image.getPixelMatrix();
			for(int ii = 0; ii < h; ++ii)
			{
				for(int jj = 0; jj < w; ++jj)
				{
					if(isHorizontal)
					{
						matrix[ii][startX + jj] = m[ii][jj];
					}
					else
					{
						matrix[startY + ii][jj] = m[ii][jj];
					}
				}
			}

			if(isHorizontal)
				startX += w + distance;
			else
				startY += h + distance;
		}
		
		OHLCBarImageBW retImage = new OHLCBarImageBW(matrix);
		
		return retImage;
	}
	
	public static OHLCBarImageBW dialate(OHLCBarImageBW image, int times)
	{
		if(null == image)
			return null;
		if(times < 2)
			return image;
		int w = image.getWidth();
		int h = image.getHeight();
		if(w < 1 || h < 1)
			return image;
		
		int[][] m = image.getPixelMatrix();
		int[][] matrix = new int[h*times][w*times];
		for(int ii = 0; ii < h; ++ii)
		{
			for(int jj = 0; jj < w; ++jj)
			{
				if(m[ii][jj] > 0)
				{
					for(int kk = 0; kk < times; ++kk)
					{
						for(int ll = 0; ll < times; ++ll)
							matrix[ii*times+kk][jj*times+ll] = m[ii][jj];
					}
				}
			}
		}
		
		OHLCBarImageBW retImage = new OHLCBarImageBW(matrix);
		
		return retImage;
	}
	
	public static BufferedImage convertOHLCBarImageBWToBufferedImage(OHLCBarImageBW image)
	{
		if(null == image)
			return null;
		
		int w = image.getWidth();
		int h = image.getHeight();
		int[][] m = image.getPixelMatrix();
		
		BufferedImage bImage = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_BINARY);
		
		for(int ii = 0; ii < w; ++ii)
			for(int jj = 0; jj < h; ++jj)
			{
				if(m[jj][ii] > 0)
					bImage.setRGB(ii, jj, Color.WHITE.getRGB());
			}
		
		return bImage;
	}
	
	public static void saveOHLCBarImageBW(OHLCBarImageBW image, String fileName)
	{
		BufferedImage bImage = convertOHLCBarImageBWToBufferedImage(image);
		
		if(bImage == null)
			return;
		
		try
		{
			ImageIO.write(bImage, "jpg", new File(fileName));
		}
		catch (IOException ioe)
		{
			ioe.printStackTrace();
		}
	}
	
	public static void main(String[] args)
	{
		String sourceFile = "src/com/byeautumn/wb/input/source/Yahoo/SPX_Daily_All.csv";
		OHLCElementTable bigTable = OHLCUtils.readOHLCDataSourceFile(sourceFile);
		System.out.println("Big table size: " + bigTable.size());
		
		List<OHLCElementTable> pieceList = SeriesDataProcessUtil.splitOHLCElementTableCasscade(bigTable, 5);
		
		System.out.println("Number of pieces: " + pieceList.size());
		System.out.println("Last piece table: " + pieceList.get(pieceList.size() - 1).printSelf());
		
		OHLCElementTable lastPiece = pieceList.get(pieceList.size() - 1);
		OHLCElementTable normalizedPiece = SeriesDataProcessUtil.normalize(lastPiece);
		System.out.println("Normalized last piece table: " + normalizedPiece.printSelf());

		ArrayList<OHLCBarImageBW> imageArr = new ArrayList<OHLCBarImageBW>(normalizedPiece.size());
		for(OHLCElement elem : normalizedPiece.getOHCLElementsSortedByDate())
		{
			OHLCBarImageBW image = OHLCImageGenerator.generateBarImageBW(elem, 50);
			System.out.println("++++++++++++++");
			System.out.println(image.printSelf());
			imageArr.add(image);
		}
		
		OHLCBarImageBW bigImage = OHLCImageGenerator.combineSeriesImagesBW(imageArr, 2, true);
		
		System.out.println("++++++++++++++");
		System.out.println(bigImage.printSelf());
		
		OHLCBarImageBW biggerImage = OHLCImageGenerator.dialate(bigImage, 2);
		System.out.println("++++++++++++++");
		System.out.println(biggerImage.printSelf());
		
		OHLCImageGenerator.saveOHLCBarImageBW(biggerImage, "src/com/byeautumn/wb/output/result/SPX_test2.jpg");
	}

}
