using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Runtime.InteropServices;
using CUDAFingerprinting.Common;
using System.IO;

namespace CUDAFingerprinting.GPU.Segmentation.Test
{
    [TestClass]
    public class SegmentationTests
    {
        [DllImport("CUDAFingerprinting.GPU.Segmentation.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "MakeMatrix")]
        public static extern void MakeMatrix(float[] fPic, int picWidth, int picHeight, int[] matrix);
        [TestMethod]
        public void SegmentationTest()
        {
            float [,] fPic = ImageHelper.LoadImage<float>(Properties.Resources._1_8);
            int width = fPic.GetLength(0);
            int height = fPic.GetLength(1);

            float[] fPicLin = new float [width * height];
            for (int i = 0; i < width; ++i)
            {
                for (int j = 0; j < height; ++j)
                {
                    fPicLin[j * width + i] = fPic[i, j];
                }
            }

            int[] matrix = new int[width * height];
            // In this matrix 1 means light shade of gray, and 0 means dark shade of gray 

            MakeMatrix(fPicLin, width, height, matrix);

            int[,] result = new int[width, height];
            for (int i = 0; i < width; ++i)
            {
                for (int j = 0; j < height; ++j)
                {
                    result[i, j] = matrix[j * width + i] * 255;
                }
            }

            ImageHelper.SaveArrayToBitmap(result).Save("Result.bmp");
        }
    }
}
