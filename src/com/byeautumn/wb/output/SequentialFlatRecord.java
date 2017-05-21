package com.byeautumn.wb.output;

import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;

/**
 * Created by qiangao on 5/20/2017.
 */
public class SequentialFlatRecord {
    private static SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd");
    private final double[] values;
    private final Date date;
    private int label;

    public SequentialFlatRecord(Date date, double[] values, int label)
    {
        this.date = date;
        this.values = values;
        this.label = label;
    }

    public double getValueAt(int idx)
    {
        if(null == values)
        {
            System.err.println("The SequentialFlatRecord instance is null.");
            return Double.NaN;
        }
        if(idx >= values.length)
        {
            System.err.println("The given index is out of bound.");
            return Double.NaN;
        }

        return values[idx];
    }

    public int getLabel()
    {
        return this.label;
    }

    public SequentialFlatRecord clone()
    {
        double[] copyValues = new double[values.length];
        for(int idx = 0; idx < values.length; ++idx)
            copyValues[idx] = values[idx];

        Date copyDate = new Date(this.date.getTime());
        return new SequentialFlatRecord(copyDate, copyValues, this.label);
    }

    public String printValuesAndLabelAsCSV()
    {
        if(null == values || values.length < 1)
            return "";

        StringBuffer sb = new StringBuffer();
        for(double value : values)
            sb.append(value).append(",");

        sb.append(this.label);

        return sb.toString();
    }

    public String printValuesAndLabelWithDateInfoAsCSV()
    {
        if(null == values || values.length < 1)
            return "";

        StringBuffer sb = new StringBuffer();
        for(double value : values)
            sb.append(value).append(",");
        Calendar calendar = Calendar.getInstance();
        calendar.setTime(date);
        sb.append(calendar.get(Calendar.DAY_OF_WEEK)).append(",");
        sb.append(calendar.get(Calendar.MONTH)).append(",");
        sb.append(this.label);

        return sb.toString();
    }

    public String printSelf()
    {
        if(null == date || null == values || values.length < 1)
            return "";

        StringBuffer sb = new StringBuffer();
        sb.append(dateFormat.format(date));
        sb.append(printValuesAndLabelAsCSV()).append(",");
        return sb.toString();
    }
}
