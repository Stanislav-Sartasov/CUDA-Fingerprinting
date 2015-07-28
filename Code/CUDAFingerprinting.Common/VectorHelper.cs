using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;

namespace CUDAFingerprinting.Common
{
    // Vector & point geometry helper class
    public static class VectorHelper
    {
        public static double PointDistance(PointF A, PointF B)
        {
            return Math.Sqrt(Math.Pow(B.X - A.X, 2) + Math.Pow(B.Y - A.Y, 2));
        }

        public static double Norm(PointF v)
        {
            return Math.Sqrt(v.X * v.X + v.Y * v.Y);
        }

        // Vector product of 2 vectors (only z coordinate, given vectors are supposed to be arranged on a plane)
        public static double VectorProduct(PointF v1, PointF v2)
        {
            return v1.X * v2.Y - v1.Y * v2.X;
        }

        public static PointF Difference(PointF v1, PointF v2)
        {
            PointF x = new PointF(v1.X - v2.X, v1.Y - v2.Y);
            return x;
        }

        // Helper function for 3 points 
        // A, B, C -> going from A to B, where is C, to the left or to the right?
        // > 0 - left (positive rotation)
        // = 0 - all 3 points are collinear
        // < 0 - right
        public static double Rotate(PointF A, PointF B, PointF C)
        {
            return VectorProduct(Difference(B, A), Difference(C, B));
        }

        // Segment intersection 
        public static bool Intersect(PointF A, PointF B, PointF C, PointF D)
        {
            // <= in the 1st case and < in the second are appropriate for the specific use of this helper (localization problem)
            bool a = Rotate(A, B, C) * Rotate(A, B, D) <= 0;
            bool b = Rotate(C, D, A) * Rotate(C, D, B) < 0;
            return a && b;
        }
    }
}
