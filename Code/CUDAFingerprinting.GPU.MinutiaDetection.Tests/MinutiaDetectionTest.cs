using System;
using System.Runtime.InteropServices;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CUDAFingerprinting.Common;
using System.IO;
using System.Collections.Generic;

namespace CUDAFingerprinting.GPU.MinutiaDetection.Tests
{
    [TestClass]
    public class MinutiaDetectionTest
    {
        [DllImport("..\\..\\..\\Debug\\CUDAFingerprinting.GPU.MinutiaDetection.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "GetMinutias")]
        public static extern int GetMinutias(float[] dest, int[] data, double[] orientation, int width, int height);

        [TestMethod]
        public void MinutiaDetectorBasicTest()
        {
            var image = Resources.skeleton;
            var bytes = ImageHelper.LoadImageAsInt(image);
            PixelwiseOrientationField field = new PixelwiseOrientationField(bytes, 16);

            float[] minutiasArray = new float[bytes.GetLength(0) * bytes.GetLength(1) * 3];
            
            int minutiasCount = GetMinutias(
                minutiasArray,
                array2Dto1D(bytes), 
                array2Dto1D(field.Orientation), 
                bytes.GetLength(1), 
                bytes.GetLength(0)
            );
            
            List<Minutia> minutias = MinutiasArrayToList(minutiasArray, minutiasCount);

            //field.SaveAboveToFile(image, Path.GetTempPath() + "//minutiaDetectionOrientationField.bmp", true);

            System.Console.WriteLine(Path.GetTempPath());//result path
            System.Console.WriteLine(minutias.Count);
            System.Console.WriteLine(minutiasCount);

            ImageHelper.MarkMinutiae(
                image,
                minutias,
                Path.GetTempPath() + "//minutiaDetectionMinutiae.png"
            );
            ImageHelper.MarkMinutiaeWithDirections(
                image,
                minutias,
                Path.GetTempPath() + "//minutiaDetectionMinutiaeWithDirections.png"
            );
        }

        private static List<Minutia> MinutiasArrayToList(float[] minutiasArray, int size)
        {
            List<Minutia> list = new List<Minutia>();
            for (int i = 0; i < size; i++)
            {
                Minutia m = new Minutia();
                m.X = Convert.ToInt32(minutiasArray[i * 3]);
                m.Y = Convert.ToInt32(minutiasArray[i * 3 + 1]);
                m.Angle = Convert.ToDouble(minutiasArray[i * 3 + 2]);
                list.Add(m);
            }
            return list;
        }

        private static int[] array2Dto1D(int[,] source)
        {
            int[] res = new int[source.GetLength(0) * source.GetLength(1)];
            for (int y = 0; y < source.GetLength(0); y++)
            {
                for (int x = 0; x < source.GetLength(1); x++)
                {
                    res[y * source.GetLength(1) + x] = source[y, x];
                }
            }
            return res;
        }

        private static double[] array2Dto1D(double[,] source)
        {
            double[] res = new double[source.GetLength(0) * source.GetLength(1)];
            for (int y = 0; y < source.GetLength(0); y++)
            {
                for (int x = 0; x < source.GetLength(1); x++)
                {
                    res[y * source.GetLength(1) + x] = source[y, x];
                }
            }
            return res;
        }
    }
}
