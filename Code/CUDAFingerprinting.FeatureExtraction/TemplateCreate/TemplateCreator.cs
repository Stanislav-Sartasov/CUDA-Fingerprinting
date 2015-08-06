using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using CUDAFingerprinting.Common;
using CUDAFingerprinting.Common.ConvexHull;
using CUDAFingerprinting.TemplateMatching.MCC;


namespace CUDAFingerprinting.FeatureExtraction.TemplateCreate
{
    class TemplateCreator
    {
        private const byte Radius = 70;
        private const byte BaseCuboid = 16;
        private const byte HeightCuboid = 6;
        private const uint NumberCell = BaseCuboid * BaseCuboid * HeightCuboid;
        private const double BaseCell = (2.0d * Radius) / BaseCuboid;
        private const double HeightCell = (2 * Math.PI) / HeightCuboid;
        private const double SigmaLocation = 28.0d / 3;
        private const double SigmaDirection = 2 * Math.PI / 9;
        private const double SigmoidParametrPsi = 0.01;
        private const byte Omega = 50;

        private readonly List<PointF> _convexHull;
        private readonly List<Minutia> _minutiaeList;

        public TemplateCreator(List<Minutia> minutiaeList)
        {
            _minutiaeList = minutiaeList;
            var pointList = _minutiaeList.Select(x => new PointF(x.X, x.Y)).ToList();
            _convexHull = Common.ConvexHull.ConvexHullModified.ExtendHull(pointList, Omega);
        }

        private double AngleHeight(int k)
        {
            return (-Math.PI + (k - 0.5) * HeightCell);
        }

        private PointF GetPoint(int i, int j, Minutia minutia)
        {
            return new PointF(
                (float)
                    (minutia.X + BaseCell *
                     (Math.Cos(minutia.Angle) * (i - (BaseCuboid + 1) / 2.0d) +
                      Math.Sin(minutia.Angle) * (j - (BaseCuboid + 1) / 2.0d))),
                (float)
                    (minutia.Y + BaseCell *
                     (-Math.Sin(minutia.Angle) * (i - (BaseCuboid + 1) / 2.0d) +
                      Math.Cos(minutia.Angle) * (j - (BaseCuboid + 1) / 2.0d)))
                    );
        }

        private double GaussianLocation(Minutia minutia, PointF point)
        {
            return Gaussian.Gaussian1D(VectorHelper.PointDistance(new PointF(minutia.X, minutia.Y), point), SigmaLocation);
        }

        private double GaussianDirection(Minutia middleMinutia, Minutia minutia, double anglePoint)
        {
            double common = Math.Sqrt(2) * SigmaDirection;
            double angle = CylinderHelper.GetAngleDiff(anglePoint,
                CylinderHelper.GetAngleDiff(middleMinutia.Angle, minutia.Angle));
            double first = Erf(((angle + HeightCell / 2)) / common);
            double second = Erf(((angle - HeightCell / 2)) / common);
            return (first - second) / 2;
        }

        private double Erf(double x)
        {
            // constants
            double a1 = 0.254829592;
            double a2 = -0.284496736;
            double a3 = 1.421413741;
            double a4 = -1.453152027;
            double a5 = 1.061405429;
            double p = 0.3275911;

            // Save the sign of x
            int sign = 1;
            if (x < 0)
                sign = -1;
            x = Math.Abs(x);

            // A&S formula 7.1.26
            double t = 1.0 / (1.0 + p * x);
            double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.Exp(-x * x);

            return sign * y;
        }

        private List<Minutia> GetNeighborhood(PointF point, Minutia middleMinutia)
        {
            List<Minutia> neighborhood = new List<Minutia>();
            foreach (var minutia in _minutiaeList)
            {
                if (VectorHelper.PointDistance(new PointF(minutia.X, minutia.Y), point) < 3 * SigmaLocation &&
                    EqualsMinutae(minutia, middleMinutia)) ;
                {
                    neighborhood.Add(minutia);
                }
            }
            return neighborhood;
        }

        private bool EqualsMinutae(Minutia firstMinutia, Minutia secondMinutia)
        {
            return (
                firstMinutia.X.Equals(secondMinutia.X) &&
                firstMinutia.Y.Equals(secondMinutia.Y) &&
                firstMinutia.Angle.Equals(secondMinutia.Angle)
                );
        }

        private double Sum(PointF point, double anglePoint,  List<Minutia> neighborhood, Minutia middleMinutia)
        {
            double Sum = 0;
            foreach (var minutia in neighborhood)
            {
                Sum += GaussianLocation(minutia, point)*GaussianDirection(middleMinutia, minutia, anglePoint);
            }
            return Sum;
        }
        private bool IsValidPoint(PointF point, Minutia middleMinutia)
        {
            return VectorHelper.PointDistance(new PointF(middleMinutia.X, middleMinutia.Y), point) < Radius &&
                   FieldFiller.IsPointInsideHull(point, _convexHull);
        }

    }
}
