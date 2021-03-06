﻿using System.Collections.Generic;
using System.Linq;
using System.Drawing;

namespace CUDAFingerprinting.Common.ConvexHull
{
    // Special comparer for our points (radial with respect to the starting point)
    public class RadialComparer : IComparer<PointF>
    {
        private PointF FirstPoint;

        public RadialComparer(PointF fp)
        {
            FirstPoint = fp;
        }

        public int Compare(PointF A, PointF B)
        {
            double direction = VectorHelper.Rotate(FirstPoint, A, B);
            return
                direction >= 0 ? -1 : 1;
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

        public static bool[,] GetFieldFilling(int rows, int columns, List<PointF> hull)
        {
            bool[,] field = new bool[rows, columns];
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