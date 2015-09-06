using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DelaunauTriangulationSample.Classes
{
    public class PolarPoint : IComparable<PolarPoint>
    {
        private Point pole;

        public double cosPhi
        {
            get;
            set;
        }

        public double sinPhi
        {
            get;
            set;
        }

        public double angle
        {
            get;
            set;
        }

        private double countAngle()
        {
            //вернет угол [-pi;pi]

            //вычисляем арккосинусы
            double t1 = Math.Acos(cosPhi);
            double t2 = -t1;

            //вычисляем арксинусы
            double t3 = Math.Asin(sinPhi);
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

        public double Rho
        {
            get;
            set;
        }

        public PolarPoint(Point pole, Point thisPoint)
        {
            this.pole = pole;
            Rho = pole.getDistance(thisPoint);
            cosPhi = (thisPoint.X - pole.X) / Rho;
            sinPhi = (thisPoint.Y - pole.Y) / Rho;
            angle = countAngle();
        }

        public Point toDecart()
        {
            return new Point(pole.X + Rho * cosPhi, pole.Y + Rho * sinPhi);
        }

        public override bool Equals(object obj)
        {
            if (obj == null)
                return false;

            if (obj.GetType() != this.GetType())
                return false;

            PolarPoint polPoint = (PolarPoint)obj;
            if (polPoint.pole.Equals(pole) && polPoint.Rho == Rho && polPoint.cosPhi == cosPhi && polPoint.sinPhi == sinPhi)
                return true;
            else
                return false;
        }

        public int CompareTo(PolarPoint other)
        {
            if (angle > other.angle)
                return 1;
            if (angle == other.angle && Rho > other.Rho)
                return 1;
            if (angle == other.angle && Rho == other.Rho)
                return 0;
            return -1;
        }
    }
}
