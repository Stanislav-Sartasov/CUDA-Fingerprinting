using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DelaunauTriangulationSample.Classes
{
    public class Vector
    {
        private Point start;
        private Point end;

        public Point Start
        {
            get
            {
                return start;
            }
        }

        public Point End
        {
            get
            {
                return end;
            }
        }

        public Point Center
        {
            get
            {
                return new Point(start.X + 0.5 * dx, start.Y + 0.5 * dy);
            }
        }

        private double dx;
        private double dy;

        public double Dx
        {
            get
            {
                return dx;
            }
        }

        public double Dy
        {
            get
            {
                return dy;
            }
        }

        public Vector(Point start, Point end)
        {
            this.start = start;
            this.end = end;

            dx = end.X - start.X;
            dy = end.Y - Start.Y;

            length = Math.Sqrt(dx * dx + dy * dy);
        }

        private double length;

        public double Length
        {
            get
            {
                return length;
            }
        }

        public Vector getSum(Vector vector)
        {
            return new Vector(start, new Point(start.X + dx + vector.dx, start.Y + dy + vector.dy));
        }

        public double getDotMultiplication(Vector vector)
        {
            return dx * vector.dx + dy * vector.dy;
        }

        public double getVectorMultiplication(Vector vector)
        {
            return dx * vector.dy - dy * vector.dx;
        }

        public Vector getRealMultiplication(double k)
        {
            return new Vector(start, new Point(start.X + k * dx, start.Y + k * dy));
        }

        public double getSumOfAngles(Point point)
        {
            Vector vecRight = new Vector(start, point);
            Vector vecLeft = new Vector(end, point);

            double angleRight = 180 * (Math.Acos(getDotMultiplication(vecRight) / (length * vecRight.Length))) / Math.PI;
            double angleLeft = 180 * (Math.Acos((new Vector(end, start).getDotMultiplication(vecLeft)) / (length * vecLeft.Length))) / Math.PI;
            return angleLeft + angleRight;
        }

        public override bool Equals(object obj)
        {
            if (obj == null)
                return false;
            if (!(obj is Vector))
                return false;

            return ((Vector)obj).Start.Equals(Start) && ((Vector)obj).End.Equals(End);
        }

        public override string ToString()
        {
            return "Vector fRhom: " + start + " to " + end;
        }
    }
}
