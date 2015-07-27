﻿using System.Collections.Generic;
using System.Linq;
using System.Drawing;

namespace CUDAFingerprinting.Common.ConvexHull
{
    // Product of vectors, difference betweeen vectors, distance between Points
    public static class VectorHelper
    {
        // Vector product of 2 vectors (only z coordinate, given vectors are supposed to be arranged on a plane)
        public static int VectorProduct(Point v1, Point v2)
        {
            return v1.X * v2.Y - v1.Y * v2.X;
        }

        public static Point Difference(Point v1, Point v2)
        {
            Point x = new Point(v1.X - v2.X, v1.Y - v2.Y);
            return x;
        }

        public static int Distance(Point v1, Point v2)
        {
            return (v1.X - v2.X) * (v1.X - v2.X) + (v1.Y - v2.Y) * (v1.Y - v2.Y);
        }
    }

    // Special comparer for our points (radial with respect to the starting point)
    public class RadialComparer : IComparer<Point>
    {
        private Point FirstPoint;

        public RadialComparer(Point fp)
        {
            FirstPoint = fp;
        }

        public int Compare(Point v1, Point v2)
        {
            int result = 1;
            if (VectorHelper.VectorProduct(
                    VectorHelper.Difference(v1, FirstPoint),
                    VectorHelper.Difference(v2, FirstPoint))
                 < 0)
            {
                result = -1;
            }
            else if ((v1.X == v2.X) && (v1.Y == v2.Y)) // <=> VectorProduct == 0
            {
                result = 0;
            }

            return result;
        }
    }

    public static class ConvexHull
    {
        // Sort points on the plane
        private static void Sort(List<Point> list)
        {
            Point firstPoint = new Point(list[0].X, list[0].Y);
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
        private static Stack<Point> Build(List<Point> points)
        {
            Point firstPoint = points[0];

            Stack<Point> hullStack = new Stack<Point>();
            hullStack.Push(points[0]);
            hullStack.Push(points[1]);

            Point top = points[1];
            Point nextToTop = points[0];

            for (int i = 2; i < points.Count; i++)
            {
                while (VectorHelper.VectorProduct(
                            VectorHelper.Difference(top, nextToTop),
                            VectorHelper.Difference(points[i], nextToTop))
                        < 0)
                {
                    hullStack.Pop();
                    top = nextToTop;
                    nextToTop = hullStack.Peek();
                }

                hullStack.Push(points[i]);
                nextToTop = top;
                top = points[i];
            }

            return hullStack;
        }


        public static List<Point> GetConvexHull(List<Point> points)
        {
            Sort(points);

            Stack<Point> hullStack = Build(points);

            List<Point> hullList = new List<Point>();
            for (int i = 0; i < hullStack.Count(); i++)
            {
                hullList.Add(hullStack.ElementAt(i));
            }

            return hullList;
        }
    }
}