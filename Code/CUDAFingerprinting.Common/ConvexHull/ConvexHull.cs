using System.Collections.Generic;
using System.Linq;
using System.Drawing;
using System;

namespace CUDAFingerprinting.Common.ConvexHull
{
    public static class VectorHelper
    {
        // Vector product of 2 vectors (only z coordinate, given vectors are supposed to be arranged on a plane)
        public static int VectorProductInt(PointF v1, PointF v2)
        {
            return (int)(v1.X * v2.Y - v1.Y * v2.X);
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
        public static int Rotate(PointF A, PointF B, PointF C)
        {
            return VectorProductInt(Difference(B, A), Difference(C, B));
        }

        // Segment intersection 
        public static bool Intersect(PointF A, PointF B, PointF C, PointF D)
        {
            // <= in the 1st case and < in the second are appropriate for the specific use of this helper
            return Rotate(A, B, C) * Rotate(A, B, D) <= 0 && Rotate(C, D, A) * Rotate(C, D, B) < 0;
        }

        public static double Norm(PointF v)
        {
            return Math.Sqrt(v.X * v.X + v.Y * v.Y);
        }
    }

    // Special comparer for our points (radial with respect to the starting point)
    public class RadialComparer : IComparer<PointF>
    {
        private PointF FirstPoint;

        public RadialComparer(PointF fp)
        {
            FirstPoint = fp;
        }

        public int Compare(PointF v1, PointF v2)
        {
            int result = 1;
            if (VectorHelper.Rotate(FirstPoint, v1, v2) > 0)
            {
                result = -1;
            }
            else if ((v1.X == v2.X) && (v1.Y == v2.Y)) // <=> VectorProductInt == 0
            {
                result = 0;
            }

            return result;
        }
    }

    public static class ConvexHull
    {
        // Sort points on the plane
        private static void Sort(List<PointF> list)
        {
            PointF firstPoint = new PointF(list[0].X, list[0].Y);
            for (int i = 0; i < list.Count; i++)
            {
                if (list[i].X < firstPoint.X)
                {
                    firstPoint = list[i];
                }
            }

            RadialComparer comparer = new RadialComparer(firstPoint);
            list.Sort(comparer);
        }

        // Build convex hull using Graham Scan
        private static Stack<PointF> Build(List<PointF> points)
        {
            PointF firstPoint = points[0];

            Stack<PointF> hullStack = new Stack<PointF>();
            hullStack.Push(points[0]);
            hullStack.Push(points[1]);

            PointF top = points[1];
            PointF nextToTop = points[0];

            for (int i = 2; i < points.Count; i++)
            {
                while (VectorHelper.Rotate(nextToTop, top, points[i]) < 0)
                {
                    hullStack.Pop();
                    top = nextToTop;
                    nextToTop = hullStack.ElementAt(1);
                }

                hullStack.Push(points[i]);
                nextToTop = top;
                top = points[i];
            }

            return hullStack;
        }


        public static List<PointF> GetConvexHull(List<PointF> points)
        {
            Sort(points);

            Stack<PointF> hullStack = Build(points);

            List<PointF> hullList = new List<PointF>();
            for (int i = 0; i < hullStack.Count(); i++)
            {
                hullList.Add(hullStack.ElementAt(i));
            }
            hullList.Reverse();

            return hullList;
        }
    }

    public static class FieldFiller
    {
        // Algorithm for any convex area (and even for some not convex)
        public static bool IsPointInsideHull(PointF point, List<PointF> hull)
        {
            int n = hull.Count;

            // If point is outside the segment (n - 1, 0, 1), it's always outside the hull
            if (VectorHelper.Rotate(hull[0], hull[1], point) < 0 || VectorHelper.Rotate(hull[0], hull[n - 1], point) > 0)
            {
                return false;
            }

            // Binary search
            int p = 1, r = n - 1;
            while (r - p > 1)
            {
                int q = (p + r) / 2;
                if (VectorHelper.Rotate(hull[0], hull[q], point) < 0)
                {
                    r = q;
                }
                else
                {
                    p = q;
                }
            }

            return !VectorHelper.Intersect(hull[0], point, hull[p], hull[r]);
        }

        public static bool[,] GetFieldFilling(int rows, int columns, List<PointF> minutiae)
        {
            bool[,] field = new bool[rows, columns];
            List<PointF> hull = ConvexHull.GetConvexHull(minutiae);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    PointF curPoint = new PointF(i, j);

                    field[i, j] = IsPointInsideHull(curPoint, hull) ? true : false;
                }
            }

            return field;
        }
    }
}