using System.Collections.Generic;
using System.Linq;
using System.Drawing;

namespace CUDAFingerprinting.Common.ConvexHull
{
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

        // Helper function for 3 points 
        // A, B, C -> going from A to B, where is C, to the left or to the right?
        // > 0 - left (positive rotation)
        // = 0 - all 3 points are collinear
        // < 0 - right
        public static int Rotate(Point A, Point B, Point C)
        {
            return VectorProduct(Difference(B, A), Difference(C, B));
        }
        
        // Segment intersection 
        public static bool Intersect(Point A, Point B, Point C, Point D)
        {
            // <= in the 1st case and < in the second are appropriate for the specific use of this helper
            return Rotate(A, B, C) * Rotate(A, B, D) <= 0 && Rotate(C, D, A) * Rotate(C, D, B) < 0;
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
            if (VectorHelper.Rotate(FirstPoint, v1, v2) > 0)
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


        public static List<Point> GetConvexHull(List<Point> points)
        {
            Sort(points);

            Stack<Point> hullStack = Build(points);

            List<Point> hullList = new List<Point>();
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
        public static bool IsPointInsideHull(Point point, List<Point> hull)
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
        
        public static bool[,] GetFieldFilling(int rows, int columns, List<Point> minutiae)
        {
            bool[,] field = new bool[rows, columns];
            List<Point> hull = ConvexHull.GetConvexHull(minutiae);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    Point curPoint = new Point(i, j);

                    field[i, j] = IsPointInsideHull(curPoint, hull) ? true : false;
                }
            }

            return field;
        }
    }
}