using System.Collections.Generic;
using System.Drawing;

namespace CUDAFingerprinting.Common.ConvexHull
{
    public static class FieldFilling  {
        public static bool[,] GetFieldFilling(int rows, int columns,List<Point> Minutiae)   {
            bool[,] Field = new bool[rows,columns];
            List<Point> Hull = ConvexHull.GetConvexHull(Minutiae);
            for (int i = 0 ; i< rows; i++)
                for (int j = 0; j < columns; j++)
                {
                    Point CurPoint = new Point(i, j);
                    bool GoodPoint = true;
                    for (int k = Hull.Count - 1; (k > 0) && GoodPoint; k--)
                    {
                        if (ActionsWithVectors.VectorProduct(ActionsWithVectors.SubtractOf(Hull[k - 1], Hull[k]),
                           ActionsWithVectors.SubtractOf(CurPoint, Hull[k])) > 0)
                            GoodPoint = false;

                    }
                    if (GoodPoint)
                        if (ActionsWithVectors.VectorProduct(ActionsWithVectors.SubtractOf(Hull[Hull.Count-1], Hull[0]),
                           ActionsWithVectors.SubtractOf(CurPoint, Hull[0])) > 0)
                            GoodPoint = false;
                    if (GoodPoint)
                        Field[i, j] = true;
                    else
                        Field[i, j] = false;
                }
            return Field;
        }
    }
}
