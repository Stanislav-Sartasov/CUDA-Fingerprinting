using System;
using System.Collections.Generic;
using System.Drawing;

namespace TriangulationWithAfineTransformation.Classes
{
    public class Point: IComparable<Point>
    {
        public double X
        {
            get;
            private set;
        }

        public double Y
        {
            get;
            private set;
        }

        public Point(double x, double y)
        {
            X = x;
            Y = y;
        }

        public double GetDistance(Point point)
        {
            return Math.Sqrt((X - point.X) * (X - point.X) + (Y - point.Y) * (Y - point.Y));
        }

        public Point GetNearestPointFrom(ICollection<Point> points)
        {
            double nearestDistance = -1;
            Point result = null;
            foreach (Point point in points)
            {
                double distance = GetDistance(point);
                if ((distance > 0) && (nearestDistance == -1 || distance < nearestDistance))
                {
                    result = point;
                    nearestDistance = distance;
                }
            }

            return result;
        }

        public override bool Equals(object obj)
        {
            if (obj == null)
                return false;
            if (obj.GetType() != this.GetType())
                return false;

            return (X == ((Point)obj).X && Y == ((Point)obj).Y);
        }

        public override string ToString()
        {
            return "(" + X + ";" + Y + ")";
        }

        public static Point GetMassCenter(ICollection<Point> points)
        {
            double x = 0;
            double y = 0;

            foreach (Point point in points)
            {
                x += point.X;
                y += point.Y;
            }

            x /= points.Count;
            y /= points.Count;

            return new Point(x, y);
        }

        public static Point GetMassCenter(params Point[] points)
        {
            double x = 0;
            double y = 0;

            foreach (Point point in points)
            {
                x += point.X;
                y += point.Y;
            }

            x /= points.Length;
            y /= points.Length;

            return new Point(x, y);
        }

        public int CompareTo(Point other)
        {
            if (Math.Abs(X) + Math.Abs(Y) > Math.Abs(other.X) + Math.Abs(other.Y))
                return 1;
            if (Math.Abs(X) + Math.Abs(Y) < Math.Abs(other.X) + Math.Abs(other.Y))
                return -1;
            return 0;
        }
        
        public void Paint(Graphics g, Pen p, int height)
        {
            g.DrawEllipse(p, (int)X - 1, height - (int)Y + 1, 3, 3);
        }
    }
}
