using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Runtime.InteropServices;
using CUDAFingerprinting.Common;
using CUDAFingerprinting.Common.OrientationField;
using System;
using System.IO;

namespace CUDAFingerprinting.GPU.OrientationField.Test
{
    [TestClass]
    public class OrientationFieldRegularizationTest
    {
        [DllImport("CUDAFingerprinting.GPU.OrientationField.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "OrientationRegularizationPixels")]
        public static extern void OrientationRegularization(float[] outpOrient, float[] inpOrient, int height, int width, int FilSize);

        [TestMethod]
        public void OrientationRegularizationAngleTest()
        {
            var image = Resources._1;
            var bytes = ImageHelper.LoadImage<int>(Resources._1);
            int height = bytes.GetLength(0);
            int width = bytes.GetLength(1);
            float[] sourceBytes = new float[height * width];
            float[] orientInp = new float[height * width];
            float[] orientOut = new float[height * width];
            PixelwiseOrientationField field1 = new PixelwiseOrientationField(bytes, 16);
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    sourceBytes[i * width + j] = (float)bytes[i, j];
                    orientInp[i * width + j] = (float)field1.Orientation[i, j];
                    orientOut[i * width + j] = 0.0f;
                }
            }
            OrientationRegularization(orientOut, orientInp, height, width, 25);
            double[,] orient_2D = new double[height, width];
            for (int i = 0; i < height; i++)
                for (int j = 0; j < width; j++)
                    orient_2D[i, j] = orientOut[i * width + j];
            //field1.NewOrientation(orient_2D);
            field1.SaveAboveToFile(image);
        }
    }
}
