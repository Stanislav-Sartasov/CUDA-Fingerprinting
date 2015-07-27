using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;

namespace CUDAFingerprinting.Common.ConvexHull
{
    public class ConvexHullMCCExtension
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

        public static bool[,] GetRoundedFieldFilling(int rows, int columns, float omega, List<PointF> hull, List<PointF> extendedHull)
        {
            bool[,] field = FieldFiller.GetFieldFilling(rows, columns, extendedHull);

            for (int i = 1; i < extendedHull.Count; i += 2)
            {
                PointF fst = extendedHull[i];
                PointF snd = extendedHull[(i + 1) % extendedHull.Count]; // All these modulos for border cases (i + 1 = extendedHull.Count)

                PointF hullPoint = hull[((i + 1) / 2) % hull.Count];

                for (int j = (int)Math.Round(hullPoint.X - omega); j < (int)Math.Round(hullPoint.X + omega); j++)
                {
                    for (int k = (int)Math.Round(hullPoint.Y - omega); k < (int)Math.Round(hullPoint.Y + omega); k++)
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
