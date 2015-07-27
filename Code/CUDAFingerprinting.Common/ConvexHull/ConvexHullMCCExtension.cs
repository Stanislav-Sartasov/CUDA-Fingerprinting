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
    }
}
