using System;
using System.Collections.Generic;
using System.Drawing;

namespace CUDAFingerprinting.Common.ConvexHull
{
    public class ConvexHullModified
    {
        public static List<PointF> ExtendHull(List<PointF> hull, double omega)
        {
            List<PointF> extendedHull = new List<PointF>();

            int n = hull.Count;

            PointF fst, snd; 
            int i = 0;
            while (i < n)
            {
                fst = hull[i];
                snd = hull[(i + 1) % n];

                PointF diff = VectorHelper.Difference(snd, fst);

                PointF orthogonalDiff = new PointF(diff.Y, -diff.X); // Right-hand side orthogonal vector
                double orthogonalDiffNorm = VectorHelper.Norm(orthogonalDiff);

                PointF moveVector = new PointF(
                    (float)(orthogonalDiff.X / orthogonalDiffNorm * omega),
                    (float)(orthogonalDiff.Y / orthogonalDiffNorm * omega));

                PointF fstMoved = new PointF(
                    fst.X + moveVector.X,
                    fst.Y + moveVector.Y);

                PointF sndMoved = new PointF(
                    snd.X + moveVector.X,
                    snd.Y + moveVector.Y);
                
                extendedHull.Add(fstMoved);
                extendedHull.Add(sndMoved);

                i++;
            }

            return extendedHull;
        }

        public static bool[,] GetRoundedFieldFilling(int rows, int columns, double omega, List<PointF> hull, List<PointF> extendedHull)
        {
            bool[,] field = FieldFiller.GetFieldFilling(rows, columns, extendedHull);

            for (int i = 1; i < extendedHull.Count; i += 2)
            {
                PointF hullPoint = hull[((i + 1) / 2) % hull.Count]; // All these modulos for border cases (i + 1 = extendedHull.Count)

                int jMin = Math.Max((int)Math.Round(hullPoint.X - omega), 0);
                int jMax = Math.Min((int)Math.Round(hullPoint.X + omega), rows);
                int kMin = Math.Max((int)Math.Round(hullPoint.Y - omega), 0);
                int kMax = Math.Min((int)Math.Round(hullPoint.Y + omega), columns);
                for (int j = jMin; j < jMax; j++)
                {
                    for (int k = kMin; k < kMax; k++)
                    {
                        PointF curPoint = new PointF(j, k);
                        if (VectorHelper.PointDistance(curPoint, hullPoint) < omega)
                        {
                            field[j, k] = true;
                        }
                    }
                }
            }

            return field;
        }
    }
}
