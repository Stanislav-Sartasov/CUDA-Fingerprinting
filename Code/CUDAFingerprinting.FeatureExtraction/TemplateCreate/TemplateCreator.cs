using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using CUDAFingerprinting.Common;
using CUDAFingerprinting.Common.ConvexHull;
using CUDAFingerprinting.TemplateMatching.MCC;


namespace CUDAFingerprinting.FeatureExtraction.TemplateCreate
{
    public class TemplateCreator
    {
        public const byte Radius = 70;
        public const byte BaseCuboid = 16;
        public const byte HeightCuboid = 6;
        public const uint NumberCell = BaseCuboid * BaseCuboid * HeightCuboid;
        public const double BaseCell = (2.0d * Radius) / BaseCuboid;
        public const double HeightCell = (2 * Math.PI) / HeightCuboid;
        public const double SigmaLocation = 28.0d / 3;
        public const double SigmaDirection = 2 * Math.PI / 9;
        public const double SigmoidParametrPsi = 0.01;
        public const byte Omega = 50;
        public const byte MinNumberMinutiae = 2;

        private readonly List<PointF> _convexHull;
        private readonly List<Minutia> _minutiaeList;

        public TemplateCreator(List<Minutia> minutiaeList)
        {
            _minutiaeList = minutiaeList;
            var pointList = _minutiaeList.Select(x => new PointF(x.X, x.Y)).ToList();
            _convexHull = ConvexHullModified.ExtendHull(pointList, Omega);
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
            return Gaussian.Gaussian1D(VectorHelper.PointDistance(new PointF(minutia.X, minutia.Y), point),
                SigmaLocation);
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
            {
                sign = -1;
            }
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
                    !EqualsMinutae(minutia, middleMinutia))
                {
                    neighborhood.Add(minutia);
                }
            }
            return neighborhood;
        }

        private bool EqualsMinutae(Minutia firstMinutia, Minutia secondMinutia)
        {
            return (
                firstMinutia.X == secondMinutia.X &&
                firstMinutia.Y == secondMinutia.Y &&
                Math.Abs(firstMinutia.Angle - secondMinutia.Angle) < double.Epsilon
                );
        }

        private double Sum(PointF point, double anglePoint, List<Minutia> neighborhood, Minutia middleMinutia)
        {
            double sum = 0;
            foreach (var minutia in neighborhood)
            {
                sum += GaussianLocation(minutia, point) * GaussianDirection(middleMinutia, minutia, anglePoint);
            }
            return sum;
        }

        private byte StepFunction(double value)
        {
            return (byte)(value >= SigmoidParametrPsi ? 1 : 0);
        }

        public Template CreateTemplate()
        {
            List<Cylinder> listCylinders = new List<Cylinder>();
            foreach (var middleMinutia in _minutiaeList)
            {
                int count = GetCountMinutia(middleMinutia);
                if (count >= MinNumberMinutiae)
                {
                    Cylinder[] cylinders = CreateCylinders(middleMinutia);
                    listCylinders.Add(cylinders[0]);
                    listCylinders.Add(cylinders[1]);
                }
            }
            uint maxCount = GetMaxCount(listCylinders);
            for (int i = 1; i < listCylinders.Count; i += 2)
            {
                if (CylinderHelper.GetOneBitsCount(listCylinders[i].Values) >= 0.75 * maxCount)
                {
                    continue;
                }
                listCylinders.RemoveAt(i--);
                listCylinders.RemoveAt(i--);
            }
            return new Template(listCylinders.ToArray());
        }

        private int GetCountMinutia(Minutia middleMinutia)
        {
            int sum = 0;
            foreach (var minutia in _minutiaeList)
            {
                if (VectorHelper.PointDistance(
                    new PointF(minutia.X, minutia.Y),
                    new PointF(middleMinutia.X, middleMinutia.Y)) <= Radius + 3 * SigmaLocation &&
                    !EqualsMinutae(minutia, middleMinutia))
                {
                    sum++;
                }
            }
            return sum;
        }

        private uint GetMaxCount(List<Cylinder> listCylinders)
        {
            uint maxCount = 0;
            for (int i = 1; i < listCylinders.Count; i += 2)
            {
                uint count = CylinderHelper.GetOneBitsCount(listCylinders[i].Values);
                maxCount = count > maxCount ? count : maxCount;
            }
            return maxCount;
        }

        private Cylinder[] CreateCylinders(Minutia minutia)
        {
            Cylinder3D value = new Cylinder3D();
            Cylinder3D mask = new Cylinder3D();

            for (int i = 1; i <= BaseCuboid; i++)
            {
                for (int j = 1; j <= BaseCuboid; j++)
                {
                    if (IsValidPoint(GetPoint(i, j, minutia), minutia))
                    {
                        for (int k = 1; k <= HeightCuboid; k++)
                        {
                            mask.SetValue(i, j, k, 1);
                            value.SetValue(i, j, k, StepFunction(Sum(
                                GetPoint(i, j, minutia),
                                AngleHeight(k),
                                GetNeighborhood(GetPoint(i, j, minutia), minutia),
                                minutia
                                )));
                        }
                    }
                }
            }
            return new[]
            {
                new Cylinder(value.Cylinder, minutia.Angle, Math.Sqrt(CylinderHelper.GetOneBitsCount(value.Cylinder))), 
                new Cylinder(mask.Cylinder, minutia.Angle, Math.Sqrt(CylinderHelper.GetOneBitsCount(mask.Cylinder)))
            };
        }

        private bool IsValidPoint(PointF point, Minutia middleMinutia)
        {
            return VectorHelper.PointDistance(new PointF(middleMinutia.X, middleMinutia.Y), point) < Radius &&
                   FieldFiller.IsPointInsideHull(point, _convexHull);
        }
    }
}
