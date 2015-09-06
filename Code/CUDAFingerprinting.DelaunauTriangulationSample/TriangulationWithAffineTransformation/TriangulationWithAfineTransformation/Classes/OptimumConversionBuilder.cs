using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TriangulationWithAfineTransformation.Classes
{
    internal class OptimumConversionBuilder : IComparable<OptimumConversionBuilder>
    {
        public Triangle result
        {
            get;
            private set;
        }

        public Triangle SrcTriangle
        {
            get;
            private set;
        }

        public Triangle DestTriangle
        {
            get;
            private set;
        }

        private double[] transformation;

        public double Dx
        {
            get
            {
                return transformation[0];
            }
        }

        public double Dy
        {
            get
            {
                return transformation[1];
            }
        }

        public double Phi
        {
            get
            {
                return transformation[2];
            }
        }

        public double Distance
        {
            get; private set;
        }

        public OptimumConversionBuilder(Triangle src, Triangle dest, bool canChangeVertexes = false, bool canReflect = false)
        {
            SrcTriangle = src;
            DestTriangle = dest;

            result = src;
            transformation = new double[3];

            int n = 90;
            double nearestDistance = src.GetDistanceTo(dest);
            for (int i = 0; i < n; i++)
            {
                double phi = Math.PI * i * 2.0 / n; //(Math.PI / 180) * i * (360 / n);
                OptimumConversionOperator oco = new OptimumConversionOperator(src.GetTransformation(0, 0, phi), dest);
                Triangle tmpResult = oco.GetOptimumTriangle();
                double distanceNew = dest.GetDistanceTo(tmpResult);
                if (distanceNew < nearestDistance)
                {
                    nearestDistance = distanceNew;
                    result = tmpResult;
                    transformation = oco.transformation;
                    transformation[2] += phi;
                }
            }

            if (canReflect)
            {
                Triangle src_change = new Triangle(src.A, src.C, src.B);
                for (int i = 0; i < n; i++)
                {
                    double phi = Math.PI * i * 2.0 / n; //(Math.PI / 180) * i * (360 / n);
                    OptimumConversionOperator oco = new OptimumConversionOperator(src_change.GetTransformation(0, 0, phi), dest);
                    Triangle tmpResult = oco.GetOptimumTriangle();
                    double distance = dest.GetDistanceTo(tmpResult);
                    if (nearestDistance == -1 || distance < nearestDistance)
                    {
                        nearestDistance = distance;
                        result = tmpResult;
                        transformation = oco.transformation;
                        transformation[2] += phi;
                    }
                }
            }

            if (canChangeVertexes)
            {
                Triangle src_change = new Triangle(src.B, src.C, src.A);
                for (int i = 0; i < n; i++)
                {
                    double phi = Math.PI * i * 2.0 / n; //(Math.PI / 180) * i * (360 / n);
                    OptimumConversionOperator oco = new OptimumConversionOperator(src_change.GetTransformation(0, 0, phi), dest);
                    Triangle tmpResult = oco.GetOptimumTriangle();
                    double distance = dest.GetDistanceTo(tmpResult);
                    if (nearestDistance == -1 || distance < nearestDistance)
                    {
                        nearestDistance = distance;
                        result = tmpResult;
                        transformation = oco.transformation;
                        transformation[2] += phi;
                    }
                }

                if (canReflect)
                {
                    src_change = new Triangle(src.B, src.A, src.C);
                    for (int i = 0; i < n; i++)
                    {
                        double phi = Math.PI * i * 2.0 / n; //(Math.PI / 180) * i * (360 / n);
                        OptimumConversionOperator oco = new OptimumConversionOperator(src_change.GetTransformation(0, 0, phi), dest);
                        Triangle tmpResult = oco.GetOptimumTriangle();
                        double distance = dest.GetDistanceTo(tmpResult);
                        if (nearestDistance == -1 || distance < nearestDistance)
                        {
                            nearestDistance = distance;
                            result = tmpResult;
                            transformation = oco.transformation;
                            transformation[2] += phi;
                        }
                    }
                }
                src_change = new Triangle(src.C, src.A, src.B);
                for (int i = 0; i < n; i++)
                {
                    double phi = Math.PI * i * 2.0 / n; //(Math.PI / 180) * i * (360 / n);
                    OptimumConversionOperator oco = new OptimumConversionOperator(src_change.GetTransformation(0, 0, phi), dest);
                    Triangle tmpResult = oco.GetOptimumTriangle();
                    double distance = dest.GetDistanceTo(tmpResult);
                    if (nearestDistance == -1 || distance < nearestDistance)
                    {
                        nearestDistance = distance;
                        result = tmpResult;
                        transformation = oco.transformation;
                        transformation[2] += phi;
                    }
                }

                if (canReflect)
                {
                    src_change = new Triangle(src.C, src.B, src.A);
                    for (int i = 0; i < n; i++)
                    {
                        double phi = Math.PI * i * 2.0 / n; //(Math.PI / 180) * i * (360 / n);
                        OptimumConversionOperator oco = new OptimumConversionOperator(src_change.GetTransformation(0, 0, phi), dest);
                        Triangle tmpResult = oco.GetOptimumTriangle();
                        double distance = dest.GetDistanceTo(tmpResult);
                        if (nearestDistance == -1 || distance < nearestDistance)
                        {
                            nearestDistance = distance;
                            result = tmpResult;
                            transformation = oco.transformation;
                            transformation[2] += phi;
                        }
                    }
                }

                Distance = nearestDistance;
            }
        }

        public class OptimumConversionOperator
        {
            public Triangle ABC
            {
                get;
                private set;
            }

            private double ABCx;
            private double ABCy;

            public Triangle ABC_
            {
                get;
                private set;
            }

            private double ABC_x;
            private double ABC_y;

            public double[] transformation
            {
                get;
                private set;
            }

            private double vectorsProductions;
            private double scalarProductions;

            public OptimumConversionOperator(Triangle src, Triangle dest)
            {
                ABC = dest;
                Point ABCcenter = Point.GetMassCenter(dest.A, dest.B, dest.C);
                ABCx = ABCcenter.X;
                ABCy = ABCcenter.Y;

                ABC_ = src;
                Point ABC_center = Point.GetMassCenter(src.A, src.B, src.C);
                ABC_x = ABC_center.X;
                ABC_y = ABC_center.Y;

                vectorsProductions = ABC.A.X * ABC_.A.Y - ABC.A.Y * ABC_.A.X +
                        +ABC.B.X * ABC_.B.Y - ABC.B.Y * ABC_.B.X +
                        +ABC.C.X * ABC_.C.Y - ABC.C.Y * ABC_.C.X;

                scalarProductions = ABC.A.X * ABC_.A.X + ABC.A.Y * ABC_.A.Y +
                        +ABC.B.X * ABC_.B.X + ABC.B.Y * ABC_.B.Y +
                        +ABC.C.X * ABC_.C.X + ABC.C.Y * ABC_.C.Y;
            }

            public Triangle GetOptimumTriangle()
            {
                transformation = SolveNonLinearSystem(0.0000000001, 10000);

                return ABC_.GetTransformation(transformation[0], transformation[1], transformation[2]);
            }

            private double f1(double dx, double dy, double phi)
            {
                return ABC_x * Math.Cos(phi) - ABC_y * Math.Sin(phi) + dx - ABCx;
            }

            private double f2(double dx, double dy, double phi)
            {
                return ABC_x * Math.Sin(phi) + ABC_y * Math.Cos(phi) + dy - ABCy;
            }

            private double f3(double dx, double dy, double phi)
            {
                return Math.Cos(phi) * (3 * dx * ABC_y - 3 * dy * ABC_x + vectorsProductions) +
                    Math.Sin(phi) * (3 * dx * ABC_x + 3 * dy * ABC_y - scalarProductions);
            }

            private double df1dx(double dx, double dy, double phi)
            {
                return 1;
            }

            private double df1dy(double dx, double dy, double phi)
            {
                return 0;
            }

            private double df1dphi(double dx, double dy, double phi)
            {
                return -ABC_x * Math.Sin(phi) - ABC_y * Math.Cos(phi);
            }

            private double df2dx(double dx, double dy, double phi)
            {
                return 0;
            }

            private double df2dy(double dx, double dy, double phi)
            {
                return 1;
            }

            private double df2dphi(double dx, double dy, double phi)
            {
                return ABC_x * Math.Cos(phi) - ABC_y * Math.Sin(phi);
            }

            private double df3dx(double dx, double dy, double phi)
            {
                return 3 * ABC_y * Math.Cos(phi) + 3 * ABC_x * Math.Sin(phi);
            }

            private double df3dy(double dx, double dy, double phi)
            {
                return -3 * ABC_x * Math.Cos(phi) - 3 * ABC_y * Math.Sin(phi);
            }

            private double df3dphi(double dx, double dy, double phi)
            {
                return -Math.Sin(phi) * (3 * dx * ABC_y - 3 * dy * ABC_x + vectorsProductions) +
                        Math.Cos(phi) * (3 * dx * ABC_x + 3 * dy * ABC_y - scalarProductions);
            }

            private double[,] getJacobian(double dx, double dy, double phi)
            {
                double[,] matrix = new double[3, 3];
                matrix[0, 0] = df1dx(dx, dy, phi);
                matrix[0, 1] = df1dy(dx, dy, phi);
                matrix[0, 2] = df1dphi(dx, dy, phi);

                matrix[1, 0] = df2dx(dx, dy, phi);
                matrix[1, 1] = df2dy(dx, dy, phi);
                matrix[1, 2] = df2dphi(dx, dy, phi);

                matrix[2, 0] = df3dx(dx, dy, phi);
                matrix[2, 1] = df3dy(dx, dy, phi);
                matrix[2, 2] = df3dphi(dx, dy, phi);
                return matrix;
            }

            private double[] getF(double dx, double dy, double phi)
            {
                double[] result = new double[3];
                result[0] = -f1(dx, dy, phi);
                result[1] = -f2(dx, dy, phi);
                result[2] = -f3(dx, dy, phi);
                return result;
            }

            private double[] SolveNonLinearSystem(double e, int maxItarations = 10)
            {
                double[] xk = new double[3];
                int iteration = 0;
                while (iteration < maxItarations)
                {
                    double[] dxk = LinearSystem.SolveLinearSystem(getJacobian(xk[0], xk[1], xk[2]), getF(xk[0], xk[1], xk[2]));
                    if (dxk[2] * dxk[2] < e)
                        break;
                    else
                    {
                        xk[0] += dxk[0];
                        xk[1] += dxk[1];
                        xk[2] += dxk[2];
                    }
                    iteration++;
                }

                return xk;
            }
        }

        public int CompareTo(OptimumConversionBuilder other)
        {
            if (Distance > other.Distance)
                return 1;
            if (Distance < other.Distance)
                return -1;
            return 0;
        }
    }
}