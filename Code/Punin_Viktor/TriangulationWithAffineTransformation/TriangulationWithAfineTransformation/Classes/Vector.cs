using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TriangulationWithAfineTransformation.Classes
{
    public class Vector
    {

        public Point Start
        {
            get;
            private set;
        }

        public Point End
        {
            get;
            private set;
        }

        public double Dx
        {
            get;
            private set;
        }

        public double Dy
        {
            get;
            private set;
        }
        
        public double Length
        {
            get;
            private set;
        }

        public Vector(Point start, Point end)
        {
            Start = start;
            End = end;

            Dx = end.X - start.X;
            Dy = end.Y - Start.Y;

            Length = Math.Sqrt(Dx * Dx + Dy * Dy);
        }

       
        public Vector GetSum(Vector vector)
        {
            return new Vector(Start, new Point(Start.X + Dx + vector.Dx, Start.Y + Dy + vector.Dy));
        }

        public double GetDotMultiplication(Vector vector)
        {
            return Dx * vector.Dx + Dy * vector.Dy;
        }

        public double GetVectorMultiplication(Vector vector)
        {
            return Dx * vector.Dy - Dy * vector.Dx;
        }

        public Vector GetRealMultiplication(double k)
        {
            return new Vector(Start, new Point(Start.X + k * Dx, Start.Y + k * Dy));
        }

        public double GetSumOfAngles(Point point)
        {
            Vector vecRight = new Vector(Start, point);
            Vector vecLeft = new Vector(End, point);

            double angleRight = 180 * (Math.Acos(GetDotMultiplication(vecRight) / (Length * vecRight.Length))) / Math.PI;
            double angleLeft = 180 * (Math.Acos((new Vector(End, Start).GetDotMultiplication(vecLeft)) / (Length * vecLeft.Length))) / Math.PI;
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
            return "Vector fRhom: " + Start + " to " + End;
        }
    }
}