using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TriangulationWithAfineTransformation.Classes
{
    public class PolarPoint : IComparable<PolarPoint>
    {
        private Point pole;

        public double CosPhi
        {
            get;
            private set;
        }

        public double SinPhi
        {
            get;
            private set;
        }

        public double Angle
        {
            get;
            private set;
        }

        public double Rho
        {
            get;
            private set;
        }

        private double CountAngle()
        {
            //вернет угол [-pi;pi]

            //вычисляем арккосинусы
            double t1 = Math.Acos(CosPhi);
            double t2 = -t1;

            //вычисляем арксинусы
            double t3 = Math.Asin(SinPhi);
            double t4;
            if (t3 > 0)
                t4 = Math.PI - t3;
            else
                t4 = -Math.PI - t3;

            if (Math.Abs(t1 - t3) < 0.01 || Math.Abs(t2 - t3) < 0.01)
                return t3;
            else
                return t4;
        }

        public PolarPoint(Point pole, Point thisPoint)
        {
            this.pole = pole;
            Rho = pole.GetDistance(thisPoint);
            CosPhi = (thisPoint.X - pole.X) / Rho;
            SinPhi = (thisPoint.Y - pole.Y) / Rho;
            Angle = CountAngle();
        }

        public Point ToCartesian()
        {
            return new Point(pole.X + Rho * CosPhi, pole.Y + Rho * SinPhi);
        }

        public override bool Equals(object obj)
        {
            if (obj == null)
                return false;

            if (obj.GetType() != this.GetType())
                return false;

            PolarPoint polPoint = (PolarPoint)obj;
            return polPoint.pole.Equals(pole) && polPoint.Rho == Rho && polPoint.CosPhi == CosPhi && polPoint.SinPhi == SinPhi;
        }

        public int CompareTo(PolarPoint other)
        {
            if (Angle > other.Angle)
                return 1;
            if (Angle == other.Angle && Rho > other.Rho)
                return 1;
            if (Angle == other.Angle && Rho == other.Rho)
                return 0;
            return -1;
        }
    }
}
