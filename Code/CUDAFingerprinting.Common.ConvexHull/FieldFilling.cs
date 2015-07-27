using System.Collections.Generic;
using System.Drawing;

namespace CUDAFingerprinting.Common.ConvexHull
{
    public static class FieldFilling
    {
        public static bool[,] GetFieldFilling(int rows, int columns, List<Point> Minutiae)
        {
            bool[,] field = new bool[rows, columns];
            List<Point> hull = ConvexHull.GetConvexHull(Minutiae);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    Point curPoint = new Point(i, j);
                    bool goodPoint = true;
                    for (int k = hull.Count - 1; (k > 0) && goodPoint; k--)
                    {
                        if (VectorHelper.VectorProduct(
                                VectorHelper.Difference(hull[k - 1], hull[k]),
                                VectorHelper.Difference(curPoint, hull[k]))
                            > 0)
                        {
                            goodPoint = false;
                        }
                    }

                    if (goodPoint &&
                        (VectorHelper.VectorProduct(
                            VectorHelper.Difference(hull[hull.Count - 1], hull[0]),
                            VectorHelper.Difference(curPoint, hull[0]))
                         > 0))
                    {
                        goodPoint = false;
                    }

                    field[i, j] = goodPoint ? true : false;
                }
            }

            return field;
        }
    }
}