using System;
using System.Runtime.InteropServices;
using CUDAFingerprinting.Common.OrientationField;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CUDAFingerprinting.Common;
using System.IO;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using CUDAFingerprinting.Common.OrientationField;

namespace CUDAFingerprinting.GPU.MinutiaDetection.Tests
{
    [TestClass]
    public class MinutiaDetectionTest
    {
        [DllImport("..\\..\\..\\Debug\\CUDAFingerprinting.GPU.MinutiaDetection.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "GetMinutias")]
        public static extern int GetMinutias(IntPtr dest, int[] data, float[] orientation, int width, int height);
        /*
        public static Bitmap Fimage = Resources.skeleton;
        public static int[,] Fbytes = ImageHelper.LoadImageAsInt(Fimage);
        public static int[] Fbytes1D = array2Dto1D(Fbytes);
        public static double[,] Ffield = (new PixelwiseOrientationField(Fbytes, 16)).Orientation;
        public static float[] Ffield1D = array2Dto1D(Ffield);
        public static Minutia[] FminutiasArray = new Minutia[Fbytes.GetLength(0) * Fbytes.GetLength(1)];

        [TestMethod]
        public void FMinutiaDetectorBasicTest()
        {
            int minutiasCount = GetMinutias(
                    FminutiasArray,
                    Fbytes1D,
                    Ffield1D,
                    Fbytes.GetLength(1),
                    Fbytes.GetLength(0)
                );
        }
        */
        [TestMethod]
        public void MinutiaDetectorBasicTest()
        {
            var image = Resources.skeleton;
            var bytes = ImageHelper.LoadImage<int>(image);
            PixelwiseOrientationField field = new PixelwiseOrientationField(bytes, 16);

            int minutiaSize = Marshal.SizeOf(typeof(Minutia));
            IntPtr minutiasArrayPtr = Marshal.AllocHGlobal(minutiaSize * bytes.GetLength(0) * bytes.GetLength(1));

            int minutiasCount = GetMinutias(
                minutiasArrayPtr,
                array2Dto1D(bytes),
                array2Dto1D(field.Orientation),
                bytes.GetLength(1),
                bytes.GetLength(0)
            );

            List<Minutia> minutias = new List<Minutia>(minutiasCount);

            for (int i = 0; i < minutiasCount; i++)
            {
                IntPtr ptr = new IntPtr(minutiasArrayPtr.ToInt32() + minutiaSize * i);
                minutias.Add(
                    (Minutia)Marshal.PtrToStructure(
                        new IntPtr(minutiasArrayPtr.ToInt32() + minutiaSize * i), 
                        typeof(Minutia)
                    )
                );
            }

            Marshal.FreeHGlobal(minutiasArrayPtr);

            //List<Minutia> minutias = new List<Minutia>(minutiasArray.Take(minutiasCount));

            //field.SaveAboveToFile(image, Path.GetTempPath() + "//minutiaDetectionOrientationField.bmp", true);

            //System.Console.WriteLine(Path.GetTempPath());//result path
            //System.Console.WriteLine(minutias.Count);
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

        private static float[] array2Dto1D(double[,] source)
        {
            float[] res = new float[source.GetLength(0) * source.GetLength(1)];
            for (int y = 0; y < source.GetLength(0); y++)
            {
                for (int x = 0; x < source.GetLength(1); x++)
                {
                    float result = (float)source[y, x];
                    if (float.IsPositiveInfinity(result))
                    {
                        result = float.MaxValue;
                    }
                    else if (float.IsNegativeInfinity(result))
                    {
                        result = float.MinValue;
                    }
                    res[y * source.GetLength(1) + x] = result;
                }
            }
            return res;
        }
    }
}
