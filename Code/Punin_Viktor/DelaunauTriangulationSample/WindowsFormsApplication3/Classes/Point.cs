using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DelaunauTriangulationSample.Classes
{
    public class Point
    {
        private double x;

        public double X
        {
            get
            {
                return x;
            }
        }

        private double y;

        public double Y
        {
            get
            {
                return y;
            }
        }

        public Point(double x, double y)
        {
            this.x = x;
            this.y = y;
        }

        public double getDistance(Point point)
        {
            return Math.Sqrt((x - point.x) * (x - point.x) + (y - point.y) * (y - point.y));
        }

        public Point getNearestPointFrom(ICollection<Point> points)
        {
            var nearestPoint =
                points.Select(x => new { Point = x, Distance = getDistance(x) }).Where(x => x.Distance != 0).OrderBy(x => x.Distance).Select(x => x.Point).FirstOrDefault();
            double nearestDistance = -1;
            Point result = null;
            foreach (Point point in points)
            {
                double distance = getDistance(point);
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

            return (x == ((Point)obj).x && y == ((Point)obj).y);
        }

        public override string ToString()
        {
            return "(" + x + ";" + y + ")";
        }

        public static Point getMassCenter(ICollection<Point> points)
        {
            double x = 0;
            double y = 0;

            foreach (Point point in points)
            {
                x += point.x;
                y += point.y;
            }

            x /= points.Count;
            y /= points.Count;

            return new Point(x, y);
        }

        public void Paint(Graphics g, Pen p, int formHeight)
        {
            g.DrawEllipse(p, (int)x - 1, formHeight - (int)y + 1, 3, 3);
        }
    }
}
