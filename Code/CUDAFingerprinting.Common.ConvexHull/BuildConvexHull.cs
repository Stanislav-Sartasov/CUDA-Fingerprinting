using System.Collections.Generic;
using System.Linq;
using System.Drawing;

namespace CUDAFingerprinting.Common.ConvexHull    {
    //Product, Subtract of Vectors and Distance between Points
    public static class ActionsWithVectors
    {
        public static int Distance(Point v1, Point v2)
        {
            return (v1.X - v2.X) * (v1.X - v2.X) + (v1.Y - v2.Y) * (v1.Y - v2.Y);
        }

        public static int VectorProduct(Point v1, Point v2)
        {
            return v1.X * v2.Y - v1.Y * v2.X;
        }

        public static Point SubtractOf(Point v1, Point v2)
        {
            Point x = new Point(v1.X - v2.X, v1.Y - v2.Y);
            return x;
        }
    }

    //Special comparer for our points
    public class SpecialComparer : IComparer<Point>   {
        private Point FirstPoint;

        public SpecialComparer(Point fp)    {
            FirstPoint = fp;
        }

        public int Compare(Point v1, Point v2)    {
            int result = 1;
            if ((ActionsWithVectors.VectorProduct(ActionsWithVectors.SubtractOf(v1, FirstPoint), ActionsWithVectors.SubtractOf(v2, FirstPoint)) < 0) ||
                (ActionsWithVectors.VectorProduct(ActionsWithVectors.SubtractOf(v1, FirstPoint), ActionsWithVectors.SubtractOf(v2, FirstPoint)) == 0) &&
                (ActionsWithVectors.Distance(v1, FirstPoint) < ActionsWithVectors.Distance(v2, FirstPoint)))
                result = -1;
            if ((v1.X == v2.X) && (v1.Y == v2.Y))
                result = 0;
            return result;
        }
    }

    //Main
    public static class ConvexHull     {
        

        //Sort points on the plane
        private static void Sort(List<Point> arr)       {
            Point firstPoint = new Point(arr[0].X, arr[0].Y);
            for (int i = 0; i < arr.Count; i++)    {
                if (arr[i].Y > firstPoint.Y)
                    firstPoint = arr[i];
                if ((arr[i].Y == firstPoint.Y) && (arr[i].X < firstPoint.X))
                    firstPoint = arr[i];
            }
            SpecialComparer comparer = new SpecialComparer(firstPoint);
            arr.Sort(comparer);
        }

        //Build Convex Hull using Graham Scan
        private static Stack<Point> Build(List<Point> arr)    {
            Point firstPoint = new Point(arr[0].X, arr[0].Y);
            Stack<Point> hullStack = new Stack<Point>();
            hullStack.Push(arr[0]);
            hullStack.Push(arr[1]);
            Point top = arr[1];
            Point nextToTop = arr[0];
            for (int i = 2; i < arr.Count; i++)    {
                while ((ActionsWithVectors.VectorProduct(ActionsWithVectors.SubtractOf(arr[i], nextToTop),
                       ActionsWithVectors.SubtractOf(top, nextToTop)) <= 0) && (!Equals(top,firstPoint)))
                {
                    top = nextToTop;
                    hullStack.Pop();
                    hullStack.Pop();
                    if (Equals(top, firstPoint))
                        nextToTop = top;
                    else
                        nextToTop = hullStack.Peek();
                    hullStack.Push(top);
                }
                hullStack.Push(arr[i]);
                nextToTop = top;
                top = arr[i];
            }
            return hullStack;
        }

        
        public static List<Point> GetConvexHull(List<Point> arr)   {
            Sort(arr);
            Stack<Point> st = Build(arr);
            List<Point> hullArr = new List<Point>();
            for (int i = 0; i < st.Count(); i++)
                hullArr.Add(st.ElementAt(i));
            return hullArr;
        }
    }
}
