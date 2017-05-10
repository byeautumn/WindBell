package com.byeautumn.wb.output;

import java.awt.Color;
import java.io.File;
import java.io.IOException;
import java.text.DateFormat;
import java.text.DecimalFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;
import java.util.List;
import java.util.TimeZone;

import javax.swing.JPanel;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.DateAxis;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.labels.StandardXYToolTipGenerator;
import org.jfree.chart.plot.CombinedDomainXYPlot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.CandlestickRenderer;
import org.jfree.chart.renderer.xy.XYBarRenderer;
import org.jfree.data.time.FixedMillisecond;
import org.jfree.data.time.RegularTimePeriod;
import org.jfree.data.time.TimeSeries;
import org.jfree.data.time.TimeSeriesCollection;
import org.jfree.data.time.ohlc.OHLCItem;
import org.jfree.data.time.ohlc.OHLCSeries;
import org.jfree.data.time.ohlc.OHLCSeriesCollection;
import org.jfree.data.xy.DefaultHighLowDataset;
import org.jfree.data.xy.DefaultOHLCDataset;
import org.jfree.data.xy.OHLCDataItem;
import org.jfree.data.xy.OHLCDataset;

import com.byeautumn.wb.data.OHLCElement;
import com.byeautumn.wb.data.OHLCElementTable;
//import com.fx.jfree.chart.candlestick.CustomHighLowItemLabelGenerator;
//import com.fx.jfree.chart.candlestick.JfreeCandlestickChart;
//import com.fx.jfree.chart.model.Trade;
//import com.fx.jfree.chart.utils.MathUtils;
//import com.fx.jfree.chart.utils.TimeUtils;

public class WDCandlestickChart extends JPanel
{

	/**
	 * 
	 */
	private static final long serialVersionUID = 7242788056826362241L;

	private static final DateFormat READABLE_TIME_FORMAT = new SimpleDateFormat("kk:mm:ss");
	private OHLCSeries ohlcSeries;
	private TimeSeries volumeSeries;
	private int timeInterval = 1;
//	private Trade candelChartIntervalFirstPrint = null;

	private DefaultHighLowDataset dataset;
//	private OHLCElementTable dataTable;
	private JFreeChart chart;
	
	
	public WDCandlestickChart(String title, OHLCElementTable dataTable)
	{
//		this.ohlcSeries = createOHLCSeries(dataTable);
		this.dataset = createHighLowDataset(dataTable);
//		this.chart = createChart(title, this.dataset);
		this.chart = createChart(title);
		this.ohlcSeries = createOHLCSeries(dataTable);
	}
	
//	public void loadTradeData(OHLCElementTable dataTable)
//	{
//		List<OHLCElement> elemArr = dataTable.getOHCLElementsSortedByDate();
//		
//		for(OHLCElement elem : elemArr)
//		{
//			// the first trade price in the day (day open price)
//			setOpen(MathUtils.roundDouble(elem.getOpenValue(), MathUtils.TWO_DEC_DOUBLE_FORMAT));
//			// the interval low
//			setLow(MathUtils.roundDouble(elem.getLowValue(), MathUtils.TWO_DEC_DOUBLE_FORMAT));
//			// the interval high
//			setHigh(MathUtils.roundDouble(elem.getHighValue(), MathUtils.TWO_DEC_DOUBLE_FORMAT));
//			// the interval close
//			setClose(MathUtils.roundDouble(elem.getCloseValue(), MathUtils.TWO_DEC_DOUBLE_FORMAT));
//			// set the initial volume
//			setVolume(elem.getVolumeValue());
//			
//			addCandel(elem);
//		}
//		
//	}
	
	private OHLCDataset createOHLCDataset(OHLCElementTable dataTable)
	{
		OHLCDataItem[] dataItemArr = new OHLCDataItem[dataTable.size()];

		int idx = 0;
		for(OHLCElement elem : dataTable.getOHCLElementsSortedByDate())
		{
			OHLCDataItem item = new OHLCDataItem(elem.getDateValue(), elem.getOpenValue(), elem.getHighValue(), elem.getLowValue(), elem.getCloseValue(), elem.getVolumeValue());
			
			dataItemArr[idx] = item;
			++idx;
		}
		
		OHLCDataset dataset = new DefaultOHLCDataset("SPX", dataItemArr);
		
		return dataset;
	}
	
	private DefaultHighLowDataset createHighLowDataset(OHLCElementTable dataTable)
	{
		Date[] dates = new Date[dataTable.size()];
		double[] opens = new double[dataTable.size()];
		double[] highs = new double[dataTable.size()];
		double[] lows = new double[dataTable.size()];
		double[] closes = new double[dataTable.size()];
		double[] volumes = new double[dataTable.size()];
		
		DefaultHighLowDataset dataset = new DefaultHighLowDataset("", dates, opens, highs, lows, closes, volumes);
		
		int idx = 0;
		for(OHLCElement elem : dataTable.getOHCLElementsSortedByDate())
		{
			dates[idx] = elem.getDateValue();
			opens[idx] = elem.getOpenValue();
			highs[idx] = elem.getHighValue();
			lows[idx] = elem.getLowValue();
			closes[idx] = elem.getCloseValue();
			volumes[idx] = elem.getVolumeValue();
			
			++idx;
		}
		
		return dataset;
	}
	
	public JFreeChart createChart(String chartTitle, OHLCDataset dataset)
	{
		this.chart = ChartFactory.createCandlestickChart(
	            "SPX",
	            null, 
	            null,
	            dataset, 
	            false
	        );
		chart.removeLegend();
		chart.setBorderVisible(false);
//		chart.setBackgroundPaint();
	        return chart;
	}
	
	public JFreeChart createChart(String chartTitle, DefaultHighLowDataset dataset)
	{
		this.chart = ChartFactory.createCandlestickChart(
	            "SPX",
	            null, 
	            null,
	            dataset, 
	            true
	        );
		chart.removeLegend();
		chart.setBorderVisible(false);
//		chart.setBackgroundPaint();
	        return chart;
	}
	
	public JFreeChart createChart2(String title)
	{
		OHLCSeriesCollection candlestickDataset = new OHLCSeriesCollection();
		candlestickDataset.addSeries(ohlcSeries);
		// Create candlestick chart priceAxis
		NumberAxis priceAxis = new NumberAxis("");
		priceAxis.setAutoRangeIncludesZero(false);
		// Create candlestick chart renderer
		CandlestickRenderer candlestickRenderer = new CandlestickRenderer(CandlestickRenderer.WIDTHMETHOD_AVERAGE,
				false, null);//new CustomHighLowItemLabelGenerator(new SimpleDateFormat("dd"), new DecimalFormat("0.0")));
		candlestickRenderer.removeAnnotations();

		// Create candlestickSubplot
		XYPlot candlestickSubplot = new XYPlot(candlestickDataset, null, priceAxis, candlestickRenderer);
		candlestickSubplot.setBackgroundPaint(Color.white);
		candlestickSubplot.setOrientation(PlotOrientation.VERTICAL);
		
		JFreeChart chart = new JFreeChart(title, JFreeChart.DEFAULT_TITLE_FONT, candlestickSubplot, true);
		chart.removeLegend();
		return chart;
	}
	
	private OHLCSeries createOHLCSeries(OHLCElementTable dataTable)
	{
		OHLCSeries series = new OHLCSeries("");
		for(OHLCElement elem : dataTable.getOHCLElementsSortedByDate())
		{
			FixedMillisecond period = new FixedMillisecond(elem.getDateValue());
			OHLCItem item = new OHLCItem(period, elem.getOpenValue(), elem.getHighValue(), elem.getLowValue(), elem.getCloseValue());
			series.add(item);
		}
		
		return series;
	}
	public JFreeChart createChart(String chartTitle) {

		/**
		 * Creating candlestick subplot
		 */
		// Create OHLCSeriesCollection as a price dataset for candlestick chart
		OHLCSeriesCollection candlestickDataset = new OHLCSeriesCollection();
		ohlcSeries = new OHLCSeries("");
		candlestickDataset.addSeries(ohlcSeries);
		// Create candlestick chart priceAxis
		NumberAxis priceAxis = new NumberAxis("");
		priceAxis.setAutoRangeIncludesZero(false);
		// Create candlestick chart renderer
		CandlestickRenderer candlestickRenderer = new CandlestickRenderer(CandlestickRenderer.WIDTHMETHOD_INTERVALDATA,
				false, null);//new CustomHighLowItemLabelGenerator(new SimpleDateFormat("dd"), new DecimalFormat("0.0")));
//		candlestickRenderer.removeAnnotations();

		// Create candlestickSubplot
		XYPlot candlestickSubplot = new XYPlot(candlestickDataset, null, priceAxis, candlestickRenderer);
		candlestickSubplot.setBackgroundPaint(Color.white);

		/**
		 * Creating volume subplot
		 */
		// creates TimeSeriesCollection as a volume dataset for volume chart
		TimeSeriesCollection volumeDataset = new TimeSeriesCollection();
		volumeSeries = new TimeSeries("");
		volumeDataset.addSeries(volumeSeries);
		// Create volume chart volumeAxis
		NumberAxis volumeAxis = new NumberAxis("");
		volumeAxis.setAutoRangeIncludesZero(false);
		// Set to no decimal
		volumeAxis.setNumberFormatOverride(new DecimalFormat("0"));
		// Create volume chart renderer
		XYBarRenderer timeRenderer = new XYBarRenderer();
		timeRenderer.setShadowVisible(false);
		timeRenderer.setBaseToolTipGenerator(new StandardXYToolTipGenerator("Volume--> Time={1} Size={2}",
				new SimpleDateFormat("dd"), new DecimalFormat("0")));
		// Create volumeSubplot
		XYPlot volumeSubplot = new XYPlot(volumeDataset, null, null, timeRenderer);
		volumeSubplot.setBackgroundPaint(Color.white);

		/**
		 * Create chart main plot with two subplots (candlestickSubplot,
		 * volumeSubplot) and one common dateAxis
		 */
		// Creating charts common dateAxis
		DateAxis dateAxis = new DateAxis("");
		dateAxis.setDateFormatOverride(new SimpleDateFormat("dd"));
		// reduce the default left/right margin from 0.05 to 0.02
		dateAxis.setLowerMargin(0.02);
		dateAxis.setUpperMargin(0.02);
		// Create mainPlot
		CombinedDomainXYPlot mainPlot = new CombinedDomainXYPlot(dateAxis);
		mainPlot.setGap(1.0);
		mainPlot.add(candlestickSubplot, 3);
		mainPlot.add(volumeSubplot, 1);
		mainPlot.setOrientation(PlotOrientation.VERTICAL);

		JFreeChart chart = new JFreeChart(chartTitle, JFreeChart.DEFAULT_TITLE_FONT, mainPlot, true);
		chart.removeLegend();
		return chart;
	}

	/**
	 * Fill series with data.
	 *
	 * @param t the t
	 */
//	public void addCandel(long time, double o, double h, double l, double c, long v) {
//		FixedMillisecond t = null;
//		try {
//			// Add bar to the data. Let's repeat the same bar
//			t = new FixedMillisecond(
//					READABLE_TIME_FORMAT.parse(TimeUtils.convertToReadableTime(time)));
//
//			ohlcSeries.add(t, o, h, l, c);
//			volumeSeries.add(t, v);
////			ohlcSeries.add(time, o, h, l, c);
////			volumeSeries.add(time, v);
//		} catch (ParseException e) {
//			e.printStackTrace();
//		} catch (Exception e)
//		{
//			e.printStackTrace();
//			System.err.println("Error: " + o + " | " + h + " | " + l + " | " + c + " | " + v + " | " + t + " | " + TimeUtils.convertToReadableTime(time));
//		}
//	}
	
	public void addCandel(OHLCElement elem) {
		FixedMillisecond t = null;
		try {
			// Add bar to the data. Let's repeat the same bar
			t = new FixedMillisecond(
					elem.getDateValue());

			ohlcSeries.add(t, elem.getOpenValue(), elem.getHighValue(), elem.getLowValue(), elem.getCloseValue());
			volumeSeries.add(t, elem.getVolumeValue());
//			ohlcSeries.add(time, o, h, l, c);
//			volumeSeries.add(time, v);
		} catch (Exception e)
		{
			e.printStackTrace();
			System.err.println("Error: " + elem.printSelf());
		}
	}
	
	public void saveChartAsJPEG(String fileName, int x, int y)
	{
		if(this.chart == null)
			return;
		try
		{
			ChartUtilities.saveChartAsJPEG(new File(fileName), chart, x, y);
		}
		catch(IOException ioe)
		{
			ioe.printStackTrace();
		}
	}
	

	public static void main(String[] args)
	{
		// TODO Auto-generated method stub

	}

}
