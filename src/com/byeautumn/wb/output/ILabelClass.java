package com.byeautumn.wb.output;

import java.util.List;

/**
 * Created by qiangao on 5/26/2017.
 */
public interface ILabelClass {
    int getNumLabels();
    List<String> getLabels();
    int getLabel(double percentage);
}
