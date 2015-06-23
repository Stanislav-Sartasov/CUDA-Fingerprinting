using System;
using System.Numerics;
using System.Threading.Tasks;

namespace CUDAFingerprinting.Common
{
    public static class KernelHelper
    {
        public static int GetKernelSizeForGaussianSigma(double sigma)
        {
            return 2*(int) Math.Ceiling(sigma*3.0f) + 1;
        }   

        public static Complex[,] MakeComplexKernel(Func<int, int, double> realFunction,
                                                   Func<int, int, double> imaginaryFunction, int size)
        {
            var realPart = MakeKernel(realFunction, size);
            var imPart = MakeKernel(imaginaryFunction, size);
            return MakeComplexFromDouble(realPart, imPart);
        }

        public static double Max2d(double[,] arr)
        {
            double max = double.NegativeInfinity;
            for (int x = 0; x < arr.GetLength(0); x++)
            {
                for (int y = 0; y < arr.GetLength(1); y++)
                {
                    if (arr[x, y] > max) max = arr[x, y];
                }
            }
            return max;
        }

        public static double Min2d(double[,] arr)
        {
            double min = double.PositiveInfinity;
            for (int x = 0; x < arr.GetLength(0); x++)
            {
                for (int y = 0; y < arr.GetLength(1); y++)
                {
                    if (arr[x, y] < min) min = arr[x, y];
                }
            }
            return min;
        }

        public static Tuple<int, int> Max2dPosition(double[,] arr)
        {
            double max = double.NegativeInfinity;
            int x=0;
            int y=0;
            
            for (int i = 0; i < arr.GetLength(0); i++)
            {
                for (int j = 0; j < arr.GetLength(1); j++)
                {
                    if (arr[i, j] > max)
                    {
                        max = arr[i, j];
                        x = i;
                        y = j;
                    }
                }
            }

            return new Tuple<int,int>(x,y);
        }

        public static double[,] MakeKernel(Func<int, int, double> function, int size)
        {
            double[,] kernel = new double[size,size];
            int center = size/2;
            double sum = 0;
            for (int x = -center; x <= center; x++)
            {
                for (int y = -center; y <= center; y++)
                {
                    sum += kernel[center + x, center + y] = function(x, y);
                }
            }
            // normalization
            if (Math.Abs(sum) >0.0000001)
                for (int x = -center; x <= center; x++)
                {
                    for (int y = -center; y <= center; y++)
                    {
                        kernel[center + x, center + y] /= sum;
                    }
                }
            return kernel;
        }

    public static Complex[,] MakeComplexFromDouble(double[,] real, double[,] imaginary)
        {
            int maxX = real.GetLength(0);
            int maxY = real.GetLength(1);
            Complex[,] result = new Complex[maxX, maxY];
            for (int x = 0; x <maxX; x++)
            {
                for (int y = 0; y <maxY; y++)
                {
                    result[x, y] = new Complex(real[x,y],imaginary[x,y]);
                }
            }
            return result;
        }

        public static double[,] Subtract(double[,] source, double[,] value)
        {
            var maxX = source.GetLength(0);
            var maxY = source.GetLength(1);
            var result = new double[maxX, maxY];
            for (int x = 0; x < maxX; x++)
            {
                for (int y = 0; y < maxY; y++)
                {
                    result[x, y] = source[x, y] - value[x, y];
                 }
            }
            return result;
        }

        public static double[,] Zip2D(double[,] arr1, double[,] arr2, Func<double,double,double> f)
        {
            var result = new double[arr1.GetLength(0), arr1.GetLength(1)];
            for (int x = 0; x < arr1.GetLength(0); x++)
            {
                for (int y = 0; y < arr1.GetLength(1); y++)
                {
                    result[x, y] = f(arr1[x, y], arr2[x, y]);
                }
            }
            return result;
        }

        public static V[,] Zip2D<T, U, V>(this T[,] arr1, U[,] arr2, Func<T, U, V> f)
        {
            var result = new V[arr1.GetLength(0), arr1.GetLength(1)];
            for (int x = 0; x < arr1.GetLength(0); x++)
            {
                for (int y = 0; y < arr1.GetLength(1); y++)
                {
                    result[x, y] = f(arr1[x, y], arr2[x, y]);
                }
            }
            return result;
        }

        public static double[,] Add(double[,] source, double[,] value)
        {
            var maxX = source.GetLength(0);
            var maxY = source.GetLength(1);
            var result = new double[maxX, maxY];
            for (int x = 0; x < maxX; x++)
            {
                for (int y = 0; y < maxY; y++)
                {
                    result[x, y] = source[x, y] + value[x, y];
                }
            }
            return result;
        }

        public static U[,] Select2D<T, U>(this T[,] array, Func<T, U> f)
        {
            var result = new U[array.GetLength(0),array.GetLength(1)];
            Parallel.For(0, array.GetLength(0), new ParallelOptions(){MaxDegreeOfParallelism = Environment.ProcessorCount}, (x,state) =>
            {

                for (int y = 0; y < array.GetLength(1); y++)
                {
                    result[x, y] = f(array[x, y]);
                }
            });

            return result;
        }

        public static U[,] Select2DParallel<T, U>(this T[,] array, Func<T, int, int, U> f)
        {
            var result = new U[array.GetLength(0), array.GetLength(1)];

            Parallel.For(0, array.GetLength(0),
                new ParallelOptions() { MaxDegreeOfParallelism = Environment.ProcessorCount }, (row, state) =>
                {
                    for (int column = 0; column < array.GetLength(1); column++)
                    {
                        result[row, column] = f(array[row, column], row, column);
                    }
                });

            return result;
        }

        public static U[,] Select2D<T, U>(this T[,] array, Func<T, int, int, U> f)
        {
            var result = new U[array.GetLength(0), array.GetLength(1)];

            for (int row = 0; row < array.GetLength(0); row++)
            {
                for (int column = 0; column < array.GetLength(1); column++)
                {
                    result[row, column] = f(array[row, column], row, column);
                }
            }

            return result;
        }

        public static U[] Select2D<T, U>(this T[] array, int rows, int columns, Func<T, int, int, U> f)
        {
            var result = new U[rows*columns];

            for (int row = 0; row < rows; row++)
            {
                for (int column = 0; column < columns; column++)
                {
                    result[row*columns + column] = f(array[row*columns + column], row, column);
                }
            }

            return result;
        }

        public static T[] Make1D<T>(this T[,] arr)
        {
            var rows = arr.GetLength(0);
            var columns = arr.GetLength(1);

            var result = new T[rows * columns];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    result[i * columns + j] = arr[i, j];
                }
            }
            return result;
        }

        public static T[,] Make2D<T>(this T[] arr, int rows, int columns)
        {
            var result = new T[rows, columns];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    result[i, j] = arr[i * columns + j];
                }
            }
            return result;
        }

        public static double Average(double[,] arr)
        {
            double sum = 0;

            foreach (double d in arr)
            {
                sum += d;
            }

            return sum / (arr.GetLength(0) * arr.GetLength(1));
        }
    }
}
