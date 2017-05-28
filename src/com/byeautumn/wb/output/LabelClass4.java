package com.byeautumn.wb.output;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by qiangao on 5/26/2017.
 */
public class LabelClass4 implements ILabelClass {

    private static final int NUMBER_LABEL = 4;
    List<String> labelDescList;
    public LabelClass4()
    {
        this.labelDescList = new ArrayList<>(NUMBER_LABEL);
        labelDescList.add(" down 0.2% or more");
        labelDescList.add(" down between 0.0% and 0.2%");
        labelDescList.add(" up between 0.0% and 0.2%");
        labelDescList.add(" up 0.2% or more");
    }

    @Override
    public int getNumLabels() {
        return NUMBER_LABEL;
    }

    @Override
    public List<String> getLabels() {
        return this.labelDescList;
    }

    @Override
    public int getLabel(double percentage) {
        int label = 0;
        if(percentage <= -0.002)
            return label;
        else if(percentage > -0.002 && percentage <= 0.0)
            return label + 1;
        else if(percentage > 0.0 && percentage <= 0.02)
            return label + 2;
        else //(percentage > 0.002)
            return label + 3;
    }
}
