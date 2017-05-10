package com.byeautumn.wb.output;

import java.util.List;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;

import com.byeautumn.wb.data.OHLCElementTable;
import com.byeautumn.wb.data.OHLCUtils;
import com.byeautumn.wb.process.SeriesDataProcessUtil;
//import com.fx.jfree.chart.candlestick.JfreeCandlestickChart;

public class CandleStickChartGenerator extends JPanel
{
	public static WDCandlestickChart generate(OHLCElementTable dataTable, String fileName)
	{
		WDCandlestickChart chart = new WDCandlestickChart("SPX", dataTable);
		
		chart.saveChartAsJPEG(fileName, 800, 200);
		
		return chart;
	}
	
    private static void createAndShowGUI(WDCandlestickChart chart) {
        //Make sure we have nice window decorations.
        JFrame.setDefaultLookAndFeelDecorated(true);

        //Create and set up the window.
        JFrame frame = new JFrame("JfreeCandlestickChartDemo");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        //Create and set up the chart.
//        WDCandlestickChart jfreeCandlestickChart = new WDCandlestickChart("TWTR");
//        new FxMarketPxFeeder(jfreeCandlestickChart, "/twtr.csv", 2).run();
        frame.setContentPane(chart);

        //Disable the resizing feature
        frame.setResizable(false);
        //Display the window.
        frame.pack();
        frame.setVisible(true);
    }
    
	public static void main(String[] args)
	{
		String sourceFile = "src/com/byeautumn/wb/input/source/Yahoo/SPX_Daily_All.csv";
		OHLCElementTable bigTable = OHLCUtils.readOHLCDataSourceFile(sourceFile);
		System.out.println("Big table size: " + bigTable.size());
		
		List<OHLCElementTable> pieceList = SeriesDataProcessUtil.splitOHLCElementTableCasscade(bigTable, 5);
		
		System.out.println("Number of pieces: " + pieceList.size());
		System.out.println("Last piece table: " + pieceList.get(pieceList.size() - 1).printSelf());
		
		String fileName = "src/com/byeautumn/wb/output/result/SPX_test.jpg";
		WDCandlestickChart chart = CandleStickChartGenerator.generate(pieceList.get(pieceList.size() - 1), fileName);

        //creating and showing this application's GUI.
        SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                createAndShowGUI(chart);
            }
        });
		System.out.println("Done.");
	}

}
