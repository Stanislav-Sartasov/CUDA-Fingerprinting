using System;
using CUDAFingerprinting.Common.OrientationField;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Runtime.InteropServices;
using CUDAFingerprinting.Common;
using System.IO;

namespace CUDAFingerprinting.GPU.SingularityRegionDetection.Test
{
    [TestClass]
    public class SingularityRegionDetectionTests
    {
        [DllImport("CUDAFingerprinting.GPU.SingularityRegionDetection.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "Detect")]
        public static extern void Detect(float[] orient, int width, int height, float[] target);
        [TestMethod]
        public void SingularityRegionDetectionTest()
        {
            var intBmp = ImageHelper.LoadImage<int>(Properties.Resources._1_1);
            int width = intBmp.GetLength(0);
            int height = intBmp.GetLength(1);

            PixelwiseOrientationField field = new PixelwiseOrientationField(intBmp, 8);

            var orient = field.Orientation;
            float[] linOrient = new float [width * height];

            for (int i = 0; i < width; i++ )
            {
                for (int j = 0; j < height; j++)
                {
                    linOrient[i * height + j] = (float)orient[i, j];
                }
            }

            float[] target = new float [width * height];

            Detect(linOrient, width, height, target);

            /*int[,] result = new int[width, height];
            for (int i = 0; i < width; ++i)
            {
                for (int j = 0; j < height; ++j)
                {
                    result[i, j] = (int)(target[j * width + i] * 255);
                }
            }

            ImageHelper.SaveArrayToBitmap(result).Save("Result.jpg");*/
        }
    }
}
