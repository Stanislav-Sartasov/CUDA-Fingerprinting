using System.Collections.Generic;
using System.Drawing;
namespace CUDAFingerprinting.Common.ConvexHull
{
    public class WorkingArea
    {
        public static bool[,] BuildWorkingArea(List<Minutia> Minutiae, int radius, int rows, int columns)
        {
            List<Point> MyMinutiae = new List<Point>();
            for (int i = 0; i < Minutiae.Count; i++)
            {
                Point x = new Point(Minutiae[i].Y, Minutiae[i].X);
                MyMinutiae.Add(x);
            }
            bool[,] primaryField = FieldFilling.GetFieldFilling(rows, columns, MyMinutiae);
            bool[,] resField = new bool[rows,columns];
            for (int i = 0; i < rows; i ++) 
                for (int j = 0; j < columns; j++)
                {
                    resField[i, j] = false;
                    if (primaryField[i, j])
                        resField[i, j] = true;
                    else
                        for (int iP = System.Math.Max(i - radius,0); iP <= System.Math.Min(rows-1,i + radius) && (!resField[i, j]); iP++)
                            for (int jP = System.Math.Max(j - radius,0); (jP <= System.Math.Min(columns-1,j + radius)) && (!resField[i, j]); jP++) 
                                if ((iP - i)*(iP - i) + (jP - j)*(jP - j) <= radius*radius)
                                    if (primaryField[iP, jP])
                                        resField[i, j] = true;
                }
            return resField;
        }
    }
}
