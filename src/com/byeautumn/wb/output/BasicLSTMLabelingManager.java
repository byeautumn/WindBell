package com.byeautumn.wb.output;

import com.byeautumn.wb.data.OHLCElement;

/**
 * Created by qiangao on 5/20/2017.
 */
public class BasicLSTMLabelingManager {
    public static boolean isLabelValid(int label)
    {
        //Note: by default, label CANNOT be MAX_VALUE or MIN_VALUE.
        return label < Integer.MAX_VALUE && label > Integer.MIN_VALUE;
    }

    private static int generateLabel_9(double curr, double next)
    {
        double move = (next - curr) / curr;

        if(move <= -0.06)
            return 0;
        else if(move > -0.06 && move <= -0.03)
            return 1;
        else if(move > -0.03 && move <= -0.015)
            return 2;
        else if(move > -0.015 && move <= -0.003)
            return 3;
        else if(move > -0.003 && move <= 0.003)
            return 4;
        else if(move > 0.003 && move <= 0.015)
            return 5;
        else if(move > 0.015 && move <= 0.03)
            return 6;
        else if(move > 0.03 && move <= 0.06)
            return 7;
        else //(move > 0.06)
            return 8;

    }

    public static int generateLabel_7(double curr, double next)
    {
        double move = (next - curr) / curr;

        int label = 0;
        if(move <= -0.04)
            return label;
        else if(move > -0.04 && move <= -0.015)
            return label + 1;
        else if(move > -0.015 && move <= -0.005)
            return label + 2;
        else if(move > -0.005 && move <= 0.003)
            return label + 3;
        else if(move > 0.003 && move <= 0.01)
            return label + 4;
        else if(move > 0.01 && move <= 0.02)
            return label + 5;
        else //(move > 0.02)
            return label + 6;

    }
    private static int generateLabel_5(double curr, double next)
    {
        double move = (next - curr) / curr;

        int label = 0;
        if(move <= -0.025)
            return label;
        else if(move > -0.025 && move <= -0.008)
            return label + 1;
        else if(move > -0.008 && move <= 0.006)
            return label + 2;
        else if(move > 0.006 && move <= 0.015)
            return label + 3;
        else //(move > 0.015)
            return label + 4;

    }

    private static int generateLabel_3(double curr, double next)
    {
        double move = (next - curr) / curr;

        int label = 0;
        if(move <= -0.015)
            return label;
        else if(move > -0.015 && move <= 0.01)
            return label + 1;
        else //(move > 0.015)
            return label + 2;

    }

    private static int generateLabel_2(double curr, double next)
    {
        double move = (next - curr) / curr;

        int label = 0;
        if(move <= 0.0)
            return label;
        else
            return label + 1;
    }
}
